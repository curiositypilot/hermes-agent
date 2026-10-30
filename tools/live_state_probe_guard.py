"""Refuse kanban-worker Python probes that import live cron writers (t_90c6fa5e).

``cron.jobs`` anchors its store at ``get_hermes_home()``, so an ad-hoc
``hermes-python -c "from cron.jobs import create_job; create_job(...)"`` run by
a kanban worker writes the live ``cron/jobs.json`` and the gateway ticker then
fires the junk job (ErrJob* in t_93b0fd0a, Brief in t_90c6fa5e). A prose rule
failed twice, and the probes imported snapshot trees' own copy of ``cron.jobs``,
so the fence sits where every probe passes: the terminal pre-exec chain and
``execute_code``. ``home-probe`` runs the same code against a throwaway home.

Accident guard, not a sandbox: obfuscated imports (string concatenation) pass.
"""

import os
import re
import shlex
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# State-writing cron modules; a submodule of one counts as the module.
_TARGETS = (
    "cron.jobs",
    "cron.scheduler",
    "tools.cronjob_tools",
    "hermes_cli.cron",
    "hermes_cli.cli_commands_mixin",
)

_FROM_RE = re.compile(r"(?<![\w.])from\s+([A-Za-z_][\w.]*)\s+import\s+(\([^)]*\)|[^;\n]*)")
_IMPORT_RE = re.compile(r"(?<![\w.])import\s+([^;\n]*)")
_DYNAMIC_RE = re.compile(r"""(?:import_module|__import__)\(\s*['"]([A-Za-z_][\w.]*)['"]""")
_NAME_RE = re.compile(r"\s*([A-Za-z_][\w.]*)")
# Package names that re-export a target's writers (cron/__init__.py): ``from cron import
# create_job`` / ``from cron import *`` / bare ``import cron`` reach them without naming
# the module. Re-exported readers (get_job, list_jobs, JOBS_FILE) are left out on purpose.
_REEXPORTS = {
    "cron": {
        **dict.fromkeys(("*", "create_job", "remove_job", "update_job", "pause_job",
                         "resume_job", "trigger_job", "rearm_oneshot"), "cron.jobs"),
        "tick": "cron.scheduler",
    },
}
_PACKAGE_IMPORTS = {"cron": "cron.jobs"}
# Cheap prefilter: no target leaf word, no import of one.
_LEAF_RE = re.compile(r"\b(?:jobs|scheduler|cronjob_tools|cron|cli_commands_mixin)\b")

_INTERPRETER_RE = re.compile(r"^(?:python[\d.]*|pypy[\d.]*|hermes-python)$")
_INTERPRETER_WORD_RE = re.compile(r"(?:python[\d.]*|pypy[\d.]*|hermes-python)\b")
_PROBE_HELPERS = frozenset({"home-probe", "kanban-probe"})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash"})
_SHELL_C_FLAG_RE = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")
_WRAPPERS = frozenset({"env", "exec", "command", "nice", "nohup", "time", "timeout", "stdbuf", "sudo"})
_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
_HEREDOC_OPEN_RE = re.compile(r"(?<!<)<<(-?)\s*(['\"]?)([A-Za-z_][\w.-]*)\2")
_SEPARATOR_CHARS = frozenset(";&|()\n")
_REDIRECT_CHARS = frozenset("<>&")
_MAX_SCRIPT_BYTES = 1 << 20
_MAX_SHELL_DEPTH = 3


def guard_active() -> bool:
    """True in a kanban worker (the agent process carries ``HERMES_KANBAN_TASK``)."""
    return os.environ.get("HERMES_KANBAN_TASK", "") != ""


def _hit(module: str) -> Optional[str]:
    for target in _TARGETS:
        if module == target or module.startswith(target + "."):
            return target
    return None


def _names(clause: str) -> Iterator[str]:
    for item in clause.strip().strip("()").split(","):
        m = _NAME_RE.match(item)
        if item.strip() == "*":
            yield "*"
        elif m:
            yield m.group(1)


def detect_live_state_probe(text: str) -> Optional[str]:
    """The cron writer module *text* imports (``import X [as Y], ...``, ``from X import ...``,
    ``from pkg import X``, a package re-export such as ``from cron import create_job``,
    ``import_module("X")``), or None."""
    if not text or not _LEAF_RE.search(text):
        return None
    for m in _FROM_RE.finditer(text):
        base = m.group(1)
        names = list(_names(m.group(2)))
        reexports = _REEXPORTS.get(base, {})
        for name in names:
            if name in reexports:
                return reexports[name]
        for candidate in (base, *(f"{base}.{n}" for n in names)):
            if _hit(candidate):
                return _hit(candidate)
    # A bare package import (`import cron`, `import_module("cron")`) binds the re-exports.
    imported = [n for m in _IMPORT_RE.finditer(text) for n in _names(m.group(1))]
    for name in imported + [m.group(1) for m in _DYNAMIC_RE.finditer(text)]:
        hit = _PACKAGE_IMPORTS.get(name) or _hit(name)
        if hit:
            return hit
    return None


def block_message(module: str) -> str:
    return (
        f"Blocked: this kanban-worker probe imports `{module}`, which writes the LIVE cron store "
        "(cron/jobs.json; the gateway ticker would fire any job it creates). Run it with "
        "`~/.hermes/scripts/home-probe -c '...'` (or `home-probe script.py`; throwaway HERMES_HOME); "
        "read live cron state with read_file or `hermes cron list`."
    )


def _split_heredocs(command: str) -> tuple[str, list[str]]:
    """``command`` with heredoc bodies removed, plus the bodies in opener order."""
    out: list[str] = []
    bodies: list[str] = []
    pending: list[tuple[str, bool]] = []
    current: list[str] = []
    for line in command.split("\n"):
        if pending:
            delim, strip_tabs = pending[0]
            if (line.lstrip("\t") if strip_tabs else line).strip() == delim:
                bodies.append("\n".join(current))
                current = []
                pending.pop(0)
            else:
                current.append(line)
            continue
        out.append(line)
        pending = [(m.group(3), m.group(1) == "-") for m in _HEREDOC_OPEN_RE.finditer(line)]
    if pending:  # unterminated: the shell reads to EOF
        bodies.append("\n".join(current))
    return "\n".join(out), bodies


def _segments(command: str) -> Iterator[tuple[list[str], list[str], list[str], str]]:
    """Simple commands of *command*: (argv words, stdin texts, stdin file paths, the
    separator token that ends the segment, "" at the end)."""
    text, bodies = _split_heredocs(command)
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    lex.whitespace = " \t\r"  # newline separates commands, like ';'
    words: list[str] = []
    stdin_texts: list[str] = []
    stdin_files: list[str] = []
    tokens = iter(lex)
    for tok in tokens:
        if tok and set(tok) <= _SEPARATOR_CHARS:
            yield words, stdin_texts, stdin_files, tok
            words, stdin_texts, stdin_files = [], [], []
        elif tok and set(tok) <= _REDIRECT_CHARS and set(tok) & {"<", ">"}:
            operand = next(tokens, "")
            if tok in ("<<", "<<-"):
                if bodies:
                    stdin_texts.append(bodies.pop(0))
            elif tok == "<<<":
                stdin_texts.append(operand)
            elif tok == "<":
                stdin_files.append(operand)
        else:
            words.append(tok)
    yield words, stdin_texts, stdin_files, ""


def _pipe_state(sep: str) -> str:
    """How *sep* treats the text piped so far: ``pipe`` feeds it on, ``group`` (a bare
    subshell paren) keeps it, anything else (``;``, ``&&``, ``||``, newline) drops it."""
    if "|" in sep and "||" not in sep:
        return "pipe"
    return "group" if set(sep) <= {"(", ")"} else "drop"


def _strip_prefix(words: list[str]) -> list[str]:
    """Drop leading ``VAR=val`` assignments and wrapper commands (``env``, ``timeout 5``,
    ``uv run [opts]``) so ``words[0]`` is the program."""
    i = 0
    while i < len(words):
        w = words[i]
        base = os.path.basename(w)
        if _ASSIGN_RE.match(w):
            i += 1
        elif base in _WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or _ASSIGN_RE.match(words[i])
                                      or (base == "timeout" and words[i][:1].isdigit())):
                i += 1
        elif base == "uv" and i + 1 < len(words) and words[i + 1] == "run":
            rest = words[i + 2:]
            j = next((k for k, x in enumerate(rest)
                      if _INTERPRETER_RE.match(os.path.basename(x)) or x.endswith(".py")), None)
            return [] if j is None else (["python"] if rest[j].endswith(".py") else []) + rest[j:]
        else:
            break
    return words[i:]


def _interpreter_inputs(args: list[str]) -> tuple[list[str], Optional[str], bool]:
    """(inline code, script path, reads stdin) for a Python interpreter's argv tail."""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            return [], (args[i + 1] if i + 1 < len(args) else None), i + 1 >= len(args)
        if a == "-":
            return [], None, True
        if not a.startswith("-"):
            return [], a, False
        if a.startswith("--"):
            i += 1
            continue
        flags = a[1:]
        for k, ch in enumerate(flags):
            if ch == "c":
                code = flags[k + 1:] or (args[i + 1] if i + 1 < len(args) else "")
                return [code], None, False
            if ch == "m":
                return [], None, False
            if ch in "WX":
                if not flags[k + 1:]:
                    i += 1
                break
        i += 1
    return [], None, True


def _read_local_script(path: str, cwd: str) -> Optional[str]:
    """Bounded host read (local backend only); None for missing, oversized or binary files."""
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(cwd) / p
        if not p.is_file() or p.stat().st_size > _MAX_SCRIPT_BYTES:
            return None
        data = p.read_bytes()
    except OSError:
        return None
    return None if b"\x00" in data else data.decode("utf-8", errors="replace")


_Producer = tuple[str, list[str], list[str], list[str], str]


def _produced(prog: str, argv: list[str], stdin_texts: list[str], stdin_files: list[str],
              cwd: str) -> list[str]:
    """Text a pipeline producer segment may write to the next one's stdin: its arguments
    (``echo``/``printf``, escapes expanded), its here-docs and, for ``cat``, its files
    (read only once a stdin-reading interpreter consumes them)."""
    texts = [" ".join(argv[1:]).replace("\\n", "\n"), *stdin_texts]
    if prog == "cat":
        files = [a for a in argv[1:] if not a.startswith("-")] + stdin_files
        texts.extend(_read_local_script(f, cwd) or "" for f in files)
    return texts


def _shell_command_string(args: list[str]) -> Optional[str]:
    """The ``-c`` command string of a shell's argv tail (``-c``, ``-lc``, ``-ec``...), or None."""
    for i, a in enumerate(args[:-1]):
        if _SHELL_C_FLAG_RE.match(a):
            return args[i + 1]
    return None


def _segment_hit(prog: str, argv: list[str], stdin_texts: list[str], stdin_files: list[str],
                 piped: list[_Producer], cwd: str, depth: int) -> Optional[str]:
    if prog in _SHELLS and depth < _MAX_SHELL_DEPTH:
        body = _shell_command_string(argv[1:])
        return None if body is None else detect_in_command(body, cwd, depth + 1)
    if prog in _PROBE_HELPERS or not _INTERPRETER_RE.match(prog):
        return None
    codes, script, reads_stdin = _interpreter_inputs(argv[1:])
    if script:
        codes.append(_read_local_script(script, cwd) or "")
    if reads_stdin:
        codes.extend(stdin_texts)
        codes.extend(_read_local_script(f, cwd) or "" for f in stdin_files)
        if not stdin_texts and not stdin_files:
            for producer in piped:
                codes.extend(_produced(*producer))
    for code in codes:
        hit = detect_live_state_probe(code)
        if hit:
            return hit
    return None


def detect_in_command(command: str, cwd: str, _depth: int = 0) -> Optional[str]:
    """The cron writer module a Python interpreter in *command* would import, or None.

    Only interpreter invocations count: their ``-c`` code, here-doc/here-string stdin,
    ``< file`` stdin, stdin piped from earlier segments of the pipeline (``echo``/``printf``
    arguments, ``cat`` here-docs and files) or script argument; ``-m`` runs are not
    inspected. ``home-probe`` / ``kanban-probe`` pass because they are the program of their
    own segment; a chained bare-interpreter segment is still checked. ``bash -c`` bodies are
    scanned recursively.
    """
    if not command or not _INTERPRETER_WORD_RE.search(command):
        return None
    try:
        segments = list(_segments(command))
    except ValueError:  # unbalanced quotes: the shell rejects the command too
        return None
    piped: list[_Producer] = []
    for words, stdin_texts, stdin_files, sep in segments:
        argv = _strip_prefix(words)
        produced: list[_Producer] = []
        if argv:
            prog = os.path.basename(argv[0])
            if prog == "cd" and len(argv) > 1:
                cwd = str(Path(cwd) / Path(argv[1]).expanduser())
            else:
                hit = _segment_hit(prog, argv, stdin_texts, stdin_files, piped, cwd, _depth)
                if hit:
                    return hit
                produced = [(prog, argv, stdin_texts, stdin_files, cwd)]
        piped = [] if _pipe_state(sep) == "drop" else piped + produced
    return None
