"""security-guidance plugin — fast pattern-matched security warnings on file writes.

Scans content written by ``write_file`` / ``patch`` / ``skill_manage`` for known dangerous patterns
and appends a ``⚠️ Security guidance`` block to the tool result; the file is still written and the
model self-corrects next turn. ``skill_manage`` is scanned per operation (the ``operations[]`` batch,
or one legacy flat op). Warn (not block) by default because patterns have a real false-positive
rate (``eval(`` in a tokenizer, ECB in a test fixture); ``SECURITY_GUIDANCE_BLOCK=1`` refuses the write
instead, ``SECURITY_GUIDANCE_DISABLE=1`` is a kill switch. Pattern data: ``patterns.py`` (Apache-2.0 fork).

Content rules skip doc/prose files by default. On ``.py``/``.pyi`` writes the code rules see a copy with
string literals and comments blanked (``_mask_python_literals``); a rule's ``scan_literals`` arm still
reads the raw text. A V4A ``patch`` blob is scanned per ``*** Update/Add File:`` section, keyed by that path.
"""

from __future__ import annotations

import bisect
import io
import json
import logging
import os
import re
import tokenize
from typing import Any, Dict, List, Optional, Tuple

from . import patterns as _patterns

logger = logging.getLogger(__name__)

# tool name -> (path_arg_name, content_arg_names). Every populated content field is scanned
# (patch's new_string, or its V4A ``patch`` text per file section: _v4a_writes). skill_manage is not here: its schema is operations[],
# scanned per op by _skill_manage_writes (legacy flat args are one op).
_TARGET_TOOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "write_file": ("path", ("content",)),
    "patch": ("path", ("new_string", "patch")),
}

# Above this we skip: matching a multi-MB blob has poor signal and slows the agent loop.
_MAX_SCAN_BYTES = 256 * 1024

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUTHY


def _compile_rules() -> list[dict[str, Any]]:
    """Pre-compile regexes once; substrings stay plain (``in`` beats a literal regex)."""
    compiled: list[dict[str, Any]] = []
    for rule in _patterns.SECURITY_PATTERNS:
        try:
            regex, scan_literals = (re.compile(rule[k]) if rule.get(k) else None for k in ("regex", "scan_literals"))
        except re.error as err:
            logger.warning("security-guidance: skipping rule %s — invalid regex: %s", rule["ruleName"], err)
            continue
        compiled.append({
            "ruleName": rule["ruleName"], "reminder": rule["reminder"], "path_filter": rule.get("path_filter"),
            "path_check": rule.get("path_check"), "substrings": tuple(rule.get("substrings", ())), "regex": regex,
            "scan_literals": scan_literals,
        })
    return compiled


_COMPILED: list[dict[str, Any]] = _compile_rules()


# Token types whose text is data, not code. FSTRING_MIDDLE/TSTRING_MIDDLE are the literal parts of
# f-/t-strings (3.12+/3.14); their ``{expr}`` parts are ordinary code tokens and stay visible.
_LITERAL_TOKENS = frozenset(
    t for t in (getattr(tokenize, n, None) for n in ("STRING", "COMMENT", "FSTRING_MIDDLE", "TSTRING_MIDDLE")) if t is not None
)
_MASKED_EXTS = (".py", ".pyi")  # not .ipynb: that is JSON


def _literal_spans(content: str) -> tuple[list[tuple[int, int]], bool]:
    """(spans, clean): ``(start, end)`` offsets of every literal/comment token in ``content``, and
    False when tokenize raised ``SyntaxError`` (the spans are then the ones found before it). An
    unterminated string spans from its start to the end of its line (single-quoted) or of the text
    (triple-quoted)."""
    starts = [0] + [i + 1 for i, ch in enumerate(content) if ch == "\n"]
    spans: list[tuple[int, int]] = []
    last_end = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(content).readline):
            end = starts[tok.end[0] - 1] + tok.end[1] if tok.end[0] <= len(starts) else len(content)
            if tok.type in _LITERAL_TOKENS:
                spans.append((starts[tok.start[0] - 1] + tok.start[1], end))
            if tok.type != tokenize.ENDMARKER:
                last_end = max(last_end, end)
    except tokenize.TokenError as err:
        # tokenize reports a bad string before emitting it, so everything after the last token is it.
        msg = str(err.args[0]) if err.args else ""
        if "string" in msg:
            stop = len(content) if "multi-line" in msg or "triple" in msg else content.find("\n", last_end)
            spans.append((last_end, len(content) if stop < 0 else stop))
    except SyntaxError:  # IndentationError, TabError included
        return spans, False
    return spans, True


def _unindented_literal_spans(content: str) -> tuple[list[tuple[int, int]], bool]:
    """``_literal_spans`` of ``content`` with leading `` \\t\\f`` stripped from every line, mapped back
    to offsets in ``content``. Indentation never decides which characters are string or comment
    tokens, and a flat copy cannot raise an indentation error. Lines split on ``\\n`` only, as
    tokenize's readline does, so rows match the original."""
    flat_parts: list[str] = []
    flat_starts: list[int] = []
    shift: list[int] = []  # characters stripped up to and including each line
    flat_len = stripped = 0
    for line in content.split("\n"):
        body = line.lstrip(" \t\f")
        stripped += len(line) - len(body)
        flat_starts.append(flat_len)
        shift.append(stripped)
        flat_parts.append(body)
        flat_len += len(body) + 1
    spans, clean = _literal_spans("\n".join(flat_parts))

    def to_original(offset: int) -> int:
        return offset + shift[bisect.bisect_right(flat_starts, offset) - 1]

    return [(to_original(b), to_original(e)) for b, e in spans], clean


def _mask_python_literals(content: str) -> str:
    """``content`` with every string literal and comment overwritten by spaces (line breaks kept, so
    same-line regexes and offsets still work). Odd dedents (patch fragments that start deeper than
    they end) are masked in full by re-tokenizing an indentation-free copy; other fragments that do
    not tokenize (open brackets) keep the masks found so far; an unterminated string is blanked from
    its start to the end of its line (single-quoted) or of the text (triple-quoted). Never raises."""
    spans, clean = _literal_spans(content)
    if not clean:
        retry, retry_clean = _unindented_literal_spans(content)
        if retry_clean:
            spans = retry
    if not spans:
        return content
    chars = list(content)
    for begin, end in spans:
        for i in range(begin, min(end, len(chars))):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


def _rule_matches(entry: dict[str, Any], path: str, content: str, code: str) -> bool:
    """One rule against one write; a raising path predicate is a non-match. path_check rules fire on
    the path ALONE and never scan content; path_filter gates content rules to relevant file types.
    ``code`` is ``content`` with literals masked (Python) or ``content`` itself; ``scan_literals``
    arms read the raw ``content``."""
    try:
        if entry["path_check"] is not None:
            return bool(entry["path_check"](path))
        if entry["path_filter"] is not None and not entry["path_filter"](path):
            return False
    except Exception:
        return False
    return (
        any(sub in code for sub in entry["substrings"])
        or (entry["regex"] is not None and bool(entry["regex"].search(code)))
        or (entry["scan_literals"] is not None and bool(entry["scan_literals"].search(content)))
    )


def _scan_content(path: str, content: str) -> list[tuple[str, str]]:
    """Return [(ruleName, reminder), ...]; each rule fires at most once per call."""
    if not content or len(content.encode("utf-8", errors="ignore")) > _MAX_SCAN_BYTES:
        return []
    path = path or ""
    code = _mask_python_literals(content) if path.endswith(_MASKED_EXTS) else content
    return [(e["ruleName"], e["reminder"]) for e in _COMPILED if _rule_matches(e, path, content, code)]


# V4A file headers (tools/patch_parser.py): Update/Add sections carry content; Delete/Move and the
# Begin/End markers close the current section.
_V4A_CONTENT_HEADER = re.compile(r"^\*\*\*\s*(?:Update|Add)\s+File:\s*(.+?)\s*$")
_V4A_OTHER_HEADER = re.compile(r"^\*\*\*\s*(?:(?:Delete|Move)\s+File:|(?:Begin|End)\s+Patch\s*$)")


def _v4a_writes(path: str, patch_text: str) -> list[tuple[str, str]]:
    """``(path, text)`` per file section of a V4A patch; the text is the section's added and
    context lines without their 1-char prefix (removed lines are not written). A blob without
    file headers is one write at ``path``."""
    sections: list[tuple[str, list[str]]] = []
    current: Optional[list[str]] = None
    for line in patch_text.removesuffix("\n").split("\n"):
        line = line.removesuffix("\r")
        if header := _V4A_CONTENT_HEADER.match(line):
            current = []
            sections.append((header.group(1), current))
        elif _V4A_OTHER_HEADER.match(line):
            current = None
        elif current is not None and not line.startswith(("-", "@@", "\\")):
            current.append(line[1:] if line[:1] in ("+", " ") else line)
    if not sections:
        return [(path, patch_text)]
    return [(p, "\n".join(lines) + "\n") for p, lines in sections if lines]


# Per-op content keys on skill_manage. create/edit/full-rewrite use ``content``;
# write_file uses ``file_content``; a SKILL.md or supporting-file patch uses ``new_string``.
_SKILL_CONTENT_KEYS = ("content", "file_content", "new_string")


def _skill_manage_writes(args: Any) -> list[tuple[str, str]]:
    """``(path, content)`` pairs from one skill_manage call.

    The advertised schema is ``operations[]``. A missing key is the legacy flat
    shape (one op). A string value is JSON from a caller that stringified the
    array; bad JSON is ignored. ``file_path`` is the path inside the skill dir;
    create / full-rewrite / a SKILL.md patch omit it and target ``SKILL.md``.
    """
    if not isinstance(args, dict):
        return []
    if "operations" not in args:
        ops: Any = [args]
    else:
        ops = args["operations"]
        if isinstance(ops, str):
            try:
                ops = json.loads(ops)
            except (ValueError, TypeError):
                return []
        if not isinstance(ops, list):
            return []
    writes: list[tuple[str, str]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        raw_path = op.get("file_path")
        path = raw_path if isinstance(raw_path, str) and raw_path else "SKILL.md"
        for key in _SKILL_CONTENT_KEYS:
            val = op.get(key)
            if isinstance(val, str) and val:
                writes.append((path, val))
    return writes


def _scan_args(tool_name: str, args: Any) -> list[tuple[str, str]]:
    """Shared scan for both hooks (block mode via pre_tool_call, warn mode via transform)."""
    if _env_flag("SECURITY_GUIDANCE_DISABLE") or not isinstance(args, dict):
        return []
    if tool_name == "skill_manage":
        writes = _skill_manage_writes(args)
    elif (spec := _TARGET_TOOLS.get(tool_name)) is not None:
        path_key, content_keys = spec
        path = raw_path if isinstance(raw_path := args.get(path_key), str) else ""
        writes: list[tuple[str, str]] = []
        for key in content_keys:
            val = args.get(key)
            if isinstance(val, str) and val:
                writes += _v4a_writes(path, val) if key == "patch" else [(path, val)]
    else:
        return []
    # One call can hit the same rule in several writes (ops, patch sections); name each rule once.
    seen: set[str] = set()
    findings: list[tuple[str, str]] = []
    for path, content in writes:
        for finding in _scan_content(path, content):
            if finding[0] not in seen:
                seen.add(finding[0])
                findings.append(finding)
    return findings


def _format_warning_block(findings: list[tuple[str, str]]) -> str:
    """Render findings into the Markdown block appended to the tool result."""
    names = ", ".join(name for name, _ in findings)
    lines = ["", "---", f"⚠️ Security guidance — {len(findings)} pattern{'s' if len(findings) != 1 else ''} matched ({names})", ""]
    for _, reminder in findings:
        lines += [reminder, ""]
    lines.append(
        "Pattern matches can be false positives. If the construct is safe in this "
        "context, briefly document why in a code comment and continue. Otherwise, "
        "fix the code before moving on."
    )
    return "\n".join(lines)


def _on_pre_tool_call(tool_name: str = "", args: Any = None, **_: Any) -> Optional[dict[str, str]]:
    """Block mode only: refuse the write if any pattern matches (None = let it through)."""
    findings = _scan_args(tool_name, args) if _env_flag("SECURITY_GUIDANCE_BLOCK") else []
    if not findings:
        return None
    return {
        "action": "block",
        "message": "security-guidance refused this write: " + _format_warning_block(findings) + "\n\nTo override, unset SECURITY_GUIDANCE_BLOCK and retry.",
    }


def _on_transform_tool_result(tool_name: str = "", args: Any = None, result: Any = None, **_: Any) -> Optional[str]:
    """Warn mode: append the warning block to the result string (None = unchanged)."""
    # In block mode pre_tool_call already handled it — the tool didn't run, no result to wrap.
    findings = [] if _env_flag("SECURITY_GUIDANCE_BLOCK") or not isinstance(result, str) else _scan_args(tool_name, args)
    if not findings:
        return None
    # Don't decorate error results — the model already has bigger problems.
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and "error" in parsed and len(parsed) <= 2:
            return None
    except (ValueError, TypeError):
        pass
    return result + "\n\n" + _format_warning_block(findings)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
