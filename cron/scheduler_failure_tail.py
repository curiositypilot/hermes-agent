"""The error text a cron failure notice quotes: head of an exception, tail of a script's output.

A failed script's error is ``Script exited with code N\\nstderr:\\n<stderr>[\\nstdout:\\n<stdout>]``
(``scheduler_script.py``). A script that logs progress to stderr ends with its cause (and a
Python traceback ends with the exception line), so keeping the head showed only progress lines
and three different causes read the same (t_69291d34). Exception strings keep the head: there the
first line is the informative one.
"""

from __future__ import annotations

import re

_SCRIPT_EXIT_PREFIX = "Script exited with code"
_STDERR_MARK = "\nstderr:\n"
_STDOUT_MARK = "\nstdout:\n"
# Two lines of the 2026-11-05 marketplace-watch cause (source name + HTTP status) are ~230 chars.
_TAIL_BUDGET = 300
_HEAD_BUDGET = 180
_HEAD_BOUND = 2000  # bound the head path's regex input; the tail path slices, no regex


def _collapse(line: str) -> str:
    return " ".join(line.split())


def _stream_section(text: str) -> tuple[str, str] | None:
    """``(stream name, section)``: stderr when present, else stdout, else None."""
    start = text.find(_STDERR_MARK)
    if start >= 0:
        start += len(_STDERR_MARK)
        end = text.find(_STDOUT_MARK, start)
        return "stderr", text[start:end] if end >= 0 else text[start:]
    start = text.find(_STDOUT_MARK)
    if start >= 0:
        return "stdout", text[start + len(_STDOUT_MARK):]
    return None


def _tail(section: str) -> str:
    """Last whole non-empty lines within the budget, ``…``-prefixed when anything was dropped."""
    lines = [c for c in (_collapse(raw) for raw in section.splitlines()) if c]
    if not lines:
        return ""
    last = lines[-1]
    if len(last) > _TAIL_BUDGET:
        cut = last[-_TAIL_BUDGET:]
        if last[-_TAIL_BUDGET - 1] != " " and " " in cut:
            cut = cut[cut.index(" ") + 1:]
        return "…" + cut
    kept = [last]
    size = len(last)
    for line in reversed(lines[:-1]):
        if size + 1 + len(line) > _TAIL_BUDGET:
            break
        kept.append(line)
        size += 1 + len(line)
    joined = " ".join(reversed(kept))
    return joined if len(kept) == len(lines) else "…" + joined


def _script_exit_text(text: str) -> str:
    newline = text.find("\n")
    head = (text if newline < 0 else text[:newline]).strip()
    stream, section = _stream_section(text) or ("", "")
    tail = _tail(section)
    return f"{head}, last {stream} lines: {tail}" if tail else head


def cleaned_failure_text(text: str) -> str:
    """The one-line error a failure notice quotes. Never ends in ``.``: the notice adds one."""
    if text.startswith(_SCRIPT_EXIT_PREFIX):
        return _script_exit_text(text).rstrip(".")
    cleaned = re.sub(r"^(RuntimeError|Exception|ValueError|HTTPStatusError):\s*", "", text[:_HEAD_BOUND])
    cleaned = _collapse(cleaned).rstrip(".")
    if len(cleaned) > _HEAD_BUDGET:
        cleaned = cleaned[:_HEAD_BUDGET - 3].rstrip().rstrip(".") + "…"
    return cleaned
