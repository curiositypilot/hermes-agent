"""Result dataclasses and pure text helpers shared by ``tools.file_operations``
and its search/lint mixins. Leaf module (imports nothing from ``tools`` at module
scope) so nothing cycles. The ``to_dict`` output of every class here IS tool
behavior — key names, order and omission rules are pinned by tests and read by
the model.
"""

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple


@dataclass
class ReadResult:
    """Result from reading a file."""
    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    truncated_lines: Optional[bool] = None
    hint: Optional[str] = None
    is_binary: bool = False
    is_image: bool = False
    base64_content: Optional[str] = None
    mime_type: Optional[str] = None
    dimensions: Optional[str] = None  # For images: "WIDTHxHEIGHT"
    error: Optional[str] = None
    #: True only when the path is genuinely absent. An error with this False is a read that
    #: FAILED (transport down, no byte transport installed); callers deciding whether a path
    #: is free must not read that as "absent". See patch_parser._apply_add.
    not_found: bool = False
    similar_files: List[str] = field(default_factory=list)
    _snapshot: Optional[tuple] = None
    # sha256 of the exact on-disk bytes ``read_file_raw`` decoded (BOM included).
    _content_sha256: Optional[str] = None
    # Codec the content was transcoded from (UTF-16 rescue); None = read as UTF-8.
    _source_encoding: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and v is not None and v != []}


@dataclass
class WriteResult:
    """Result from writing a file."""
    bytes_written: int = 0
    dirs_created: bool = False
    # True when the on-disk sha256 matched the intended content; None when the
    # backend couldn't verify (no sha256sum). A mismatch is a hard error, never a flag.
    verified: Optional[bool] = None
    _content_sha256: Optional[str] = None
    lint: Optional[Dict[str, Any]] = None
    # LSP semantic diagnostics, kept separate from ``lint`` (syntax) so the model
    # reads the two as independent signals. None when LSP is off/inapplicable.
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and v is not None}


@dataclass
class PatchResult:
    """Result from patching a file."""
    success: bool = False
    diff: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    lint: Optional[Dict[str, Any]] = None
    lsp_diagnostics: Optional[str] = None  # see WriteResult.lsp_diagnostics
    error: Optional[str] = None
    # Success-shaped no-op: the edit was already present, nothing written; ``note`` says why.
    no_change: bool = False
    note: Optional[str] = None
    # One ``(path, read_sha256, written_sha256)`` per file write, in order: the sha256 of the
    # bytes the edit read ("" when it created the file, None when unknown) and of the bytes it
    # wrote (None when unknown). patch_tool carries a task's full-file baseline across them.
    _writes: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)

    # Emission order is part of the output contract.
    _DICT_FIELDS: ClassVar[tuple] = (
        "diff", "files_modified", "files_created", "files_deleted",
        "lint", "lsp_diagnostics", "error",
    )

    def to_dict(self) -> dict:
        result: Dict[str, Any] = {"success": self.success}
        if self.no_change:
            result["no_change"] = True
        if self.note:
            result["note"] = self.note
        for key in self._DICT_FIELDS:
            value = getattr(self, key)
            if value:
                result[key] = value
        return result


@dataclass
class SearchMatch:
    """A single search match."""
    path: str
    line_number: int
    content: str
    mtime: float = 0.0  # Modification time for sorting


@dataclass
class SearchResult:
    """Result from searching."""
    matches: List[SearchMatch] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    limit_reason: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None

    # Below this many matches the verbose array is already compact enough that
    # a path-grouping header would cost more tokens than it saves.
    _DENSIFY_MIN_MATCHES: ClassVar[int] = 5

    def _densify_matches(self) -> Optional[str]:
        """Lossless path-grouped text block: path once, then ``  <line>: <content>``
        rows. Relies on rg/grep emitting a file's hits consecutively. None when
        too few matches to be worth it."""
        if len(self.matches) < self._DENSIFY_MIN_MATCHES:
            return None
        lines: list[str] = []
        current_path: Optional[str] = None
        for m in self.matches:
            if m.path != current_path:
                lines.append(m.path)
                current_path = m.path
            # rstrip only: leading indentation is meaningful code and kept verbatim.
            lines.append(f"  {m.line_number}: {m.content.rstrip()}")
        return "\n".join(lines)

    def to_dict(self, densify: bool = False) -> dict:
        result: dict[str, object] = {"total_count": self.total_count}
        if self.matches:
            dense = self._densify_matches() if densify else None
            if dense is not None:
                # Self-describing so the model never guesses the block's shape.
                result["matches_format"] = (
                    "path-grouped: each file path on its own line, followed by "
                    "indented '<line>: <content>' rows for matches in that file"
                )
                result["matches_text"] = dense
            else:
                result["matches"] = [
                    {"path": m.path, "line": m.line_number, "content": m.content} for m in self.matches
                ]
        if self.files:
            result["files"] = self.files
        if self.counts:
            result["counts"] = self.counts
        if self.truncated:
            result["truncated"] = True
            result["total_count_is_lower_bound"] = True
        for key in ("limit_reason", "warning", "error"):
            value = getattr(self, key)
            if value:
                result[key] = value
        return result


@dataclass
class LintResult:
    """Result from linting a file."""
    success: bool = True
    skipped: bool = False
    output: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        if self.skipped:
            return {"status": "skipped", "message": self.message}
        result = {"status": "ok" if self.success else "error", "output": self.output}
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class ExecuteResult:
    """Result from executing a shell command."""
    stdout: str = ""
    exit_code: int = 0
    # Set when the backend's own ``builtin cd -- <cwd> || exit 126`` wrapper failed:
    # the command never ran, so the verdict is about the working directory, not
    # the requested path (callers must neither cache it nor blame the path).
    cwd_error: str = ""


# ---------------------------------------------------------------------------
# Pure text helpers (no I/O)
# ---------------------------------------------------------------------------

_OSC_SEQUENCE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_FENCE_MARKER_RE = re.compile(r"'?\x07?__HERMES_FENCE_[A-Za-z0-9]+__\x07?'?")


_CONFLICT_OPEN = re.compile(r"^\s*\d+\|<<<<<<< ", re.M)
_CONFLICT_CLOSE = re.compile(r"^\s*\d+\|>>>>>>> ", re.M)


def count_conflict_blocks(formatted_content: str) -> int:
    """Unresolved git merge-conflict blocks in a ``LINE|CONTENT`` read; 0 when the page has no
    balanced ``<<<<<<< `` / ``>>>>>>> `` pair (a lone marker in prose or a test fixture is not a
    conflict). Reported on read so the model resolves the conflict instead of editing around it."""
    opens = len(_CONFLICT_OPEN.findall(formatted_content))
    return min(opens, len(_CONFLICT_CLOSE.findall(formatted_content))) if opens else 0


def _strip_terminal_fence_leaks(text: str) -> str:
    """Strip leaked terminal fence wrappers (OSC sequences, fence markers) from
    command output; drops lines that were nothing but wrapper."""
    if not text:
        return text
    cleaned_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        had_terminal_wrapper = "__HERMES_FENCE_" in line or "\x1b]" in line
        cleaned = _FENCE_MARKER_RE.sub("", _OSC_SEQUENCE_RE.sub("", line)).replace("\x07", "")
        if had_terminal_wrapper and cleaned.strip("'\r\n\t ") == "":
            continue
        cleaned_lines.append(cleaned)
    return "".join(cleaned_lines)


def _detect_line_ending(sample: str) -> Optional[str]:
    """Dominant line ending of ``sample`` (``\\r\\n`` if any CRLF in the first 4KB,
    else ``\\n``), or None for empty/single-line content. Preserves a file's
    endings across write_file/patch: bare-LF tool args would otherwise silently
    normalize CRLF files, and patch would produce mixed endings."""
    head = sample[:4096] if sample else ""
    if "\r\n" in head:
        return "\r\n"
    if "\n" in head:
        return "\n"
    return None


def _normalize_line_endings(text: str, target: str) -> str:
    """Convert every line ending (CRLF, lone CR, LF) in ``text`` to ``target``.
    Idempotent. Collapses to LF first — separate replacements would
    double-convert CRLF → LFLF."""
    lf_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return lf_normalized
    if target == "\r\n":
        return lf_normalized.replace("\n", "\r\n")
    return text


# UTF-8 BOM (EF BB BF == U+FEFF), prepended by some Windows editors. Stripped on
# read so the model never sees a phantom first character (and patch's first-line
# match works), restored on write when the on-disk file had one.
_UTF8_BOM = "\ufeff"


def _strip_bom(text: str) -> tuple[str, bool]:
    """Return (text-without-leading-BOM, had_bom). Only a leading BOM is
    stripped; mid-content U+FEFF is legitimate data."""
    if _has_bom(text):
        return text[len(_UTF8_BOM):], True
    return text, False


def _has_bom(text: Optional[str]) -> bool:
    """True if ``text`` begins with a UTF-8 BOM."""
    return bool(text) and text.startswith(_UTF8_BOM)


# ---------------------------------------------------------------------------
# Display-placeholder write-back guard (patch replace + V4A update)
# ---------------------------------------------------------------------------

# read_file/search_files and tool results show secrets masked (agent/redact.py): the file-read
# sentinel ``«redacted:sk-…»``, a bare ``***``, or a head/tail ``sk-ant...ghij``. Redaction is
# display-only, so a model that copies a masked line into patch would write the mask over the
# real value; the similarity strategies in fuzzy_match still match the masked block.
_TOKEN_DELIMS = frozenset("\"'`()[]{}<>,;")
_TOKEN_CHAR = r"[^\s\"'`()\[\]{}<>,;]"
_PLACEHOLDER_RE = re.compile(
    rf"«redacted[^»]*»|\*{{3}}|{_TOKEN_CHAR}{{2,}}(?:\.\.\.|…){_TOKEN_CHAR}{{2,}}")


def _is_token_char(ch: str) -> bool:
    return not ch.isspace() and ch not in _TOKEN_DELIMS


def _placeholder_candidates(text: str) -> set:
    """Mask-shaped tokens in ``text``: each match, plus the match widened to its run of
    non-delimiter chars (the ``***`` in ``KEY=***`` also yields ``KEY=***``)."""
    tokens = set()
    for m in _PLACEHOLDER_RE.finditer(text):
        start, end = m.start(), m.end()
        while start > 0 and _is_token_char(text[start - 1]):
            start -= 1
        while end < len(text) and _is_token_char(text[end]):
            end += 1
        tokens.update((m.group(0), text[start:end]))
    return tokens


def _changed_lines(original: str, new_content: str) -> Tuple[List[int], str]:
    """(indices of ``original`` lines the edit replaced or deleted, text of the lines it
    inserted). Unchanged lines are excluded, so a secret line the edit left alone never counts."""
    import difflib
    old, new = original.split("\n"), new_content.split("\n")
    lo = 0
    while lo < min(len(old), len(new)) and old[lo] == new[lo]:
        lo += 1
    hi_old, hi_new = len(old), len(new)
    while hi_old > lo and hi_new > lo and old[hi_old - 1] == new[hi_new - 1]:
        hi_old, hi_new = hi_old - 1, hi_new - 1
    removed: List[int] = []
    inserted: List[str] = []
    matcher = difflib.SequenceMatcher(None, old[lo:hi_old], new[lo:hi_new], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            removed.extend(range(lo + i1, lo + i2))
            inserted.extend(new[lo + j1:lo + j2])
    return removed, "\n".join(inserted)


def _removed_views(original: str, removed: List[int], path: str) -> Tuple[str, List[str]]:
    """(raw text of the removed lines, those lines as the model could have seen them): the
    read_file/search_files view and the tool-result/diff view, each cut from the whole-file view
    (the context the model read), or redacted as a block when that view changed the line count."""
    from agent.redact import _is_secret_file_arg, redact_sensitive_text
    old = original.split("\n")
    removed_text = "\n".join(old[i] for i in removed)
    views = []
    for kwargs in ({"file_read": True, "secret_file": _is_secret_file_arg(path)}, {}):
        full = redact_sensitive_text(original, **kwargs).split("\n")
        if len(full) == len(old):
            views.append("\n".join(full[i] for i in removed))
        else:
            views.append(redact_sensitive_text(removed_text, **kwargs))
    return removed_text, views


def display_placeholder_error(original: str, new_content: str, new_text: str,
                              path: str) -> Optional[str]:
    """Error string when an edit of ``original`` into ``new_content`` writes a display mask over
    the real value it replaced, else None. ``new_text`` is the model's replacement text (the cheap
    pre-filter). A token is refused when the edit's inserted lines carry more copies of it than
    the lines it replaced, and those replaced lines hold it only in a redacted view — i.e. the
    model copied a masked line from this file. Literal sentinels/``***`` already in the file and
    masks added away from the replaced secret line pass. With redaction off, views equal the
    raw text and nothing is refused."""
    candidates = _placeholder_candidates(new_text)
    if not candidates:
        return None
    removed, inserted = _changed_lines(original, new_content)
    if not removed or not inserted:
        return None
    removed_text, views = _removed_views(original, removed, path)
    for token in sorted(candidates, key=lambda t: (-len(t), t)):
        raw = removed_text.count(token)
        if inserted.count(token) > raw and any(v.count(token) > raw for v in views):
            return (
                f"Refused to patch {path}: the new text writes {token!r}, which is how the read "
                f"view masks a secret on a line this edit replaces. The file holds the real "
                f"value; patch will not write the mask over it. No changes were written. To keep "
                f"the real value, leave its line out of the edit's old and new text (anchor the "
                f"edit around it). To change the value itself, edit it with a terminal script that "
                f"does not print or copy the masked text.")
    return None


# ---------------------------------------------------------------------------
# Pagination clamps
# ---------------------------------------------------------------------------

DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 2000
DEFAULT_SEARCH_OFFSET = 0
DEFAULT_SEARCH_LIMIT = 50


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort integer coercion for tool pagination inputs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(offset: Any = DEFAULT_READ_OFFSET,
                              limit: Any = DEFAULT_READ_LIMIT) -> tuple[int, int]:
    """Clamp read_file pagination so invalid values can never reach a sed range
    like ``0,-1p`` (schemas declare bounds, but not every caller enforces them).
    The ``limit`` ceiling is ``tool_output.max_lines`` from config.yaml."""
    from tools.tool_output_limits import get_max_lines
    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = max(1, min(_coerce_int(limit, DEFAULT_READ_LIMIT), get_max_lines()))
    return normalized_offset, normalized_limit


def normalize_search_pagination(offset: Any = DEFAULT_SEARCH_OFFSET,
                                limit: Any = DEFAULT_SEARCH_LIMIT) -> tuple[int, int]:
    """Return safe search pagination bounds for shell head/tail pipelines."""
    return max(0, _coerce_int(offset, DEFAULT_SEARCH_OFFSET)), max(1, _coerce_int(limit, DEFAULT_SEARCH_LIMIT))
