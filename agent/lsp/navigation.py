"""LSP code navigation behind the ``lsp`` tool: definition, references, hover, symbols.

Rides the diagnostics service (:class:`agent.lsp.manager.LSPService`): the same gate
(``gate_reason``: enabled flag, git workspace, ``exclude_roots``, workspace trust, broken set), the same
clients and the same loop.  A query first waits out any unfinished post-write check of the queried file
(``agent.lsp.edit_check`` syncs its own snapshots), then syncs the file's disk content only when the
server holds something else, so navigation never moves a document under a write's verdict.  Results are
``path:line:col`` plus the source line, capped at :data:`MAX_RESULTS`.  Never marks a server broken: a
slow answer (a cold workspace index) is not a dead server.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Optional

from agent.lsp.client import file_uri, uri_to_path
from agent.lsp.protocol import LSPProtocolError, LSPRequestError
from agent.lsp.servers import language_id_for
from agent.lsp.workspace import find_git_worktree, is_inside_workspace

MAX_RESULTS = 50
REQUEST_TIMEOUT = 60.0  # references on a cold project: the server indexes the workspace first
SPAWN_ALLOWANCE = 60.0  # outer ceiling on top of the request: cold spawn + initialize
EDIT_CHECK_WAIT = 10.0
# A freshly attached root: the server lists its files first and, until then, answers references and
# workspace symbols from the open files alone (pyright: 3 references at 0.2s, all 51 at 3s, on a
# 10k-file repo).  Workspace-wide queries wait this long after the attach, then for open progress.
WORKSPACE_SETTLE = 5.0
_WORKSPACE_WIDE = frozenset({"textDocument/references", "workspace/symbol"})
# Outline: a function's children are its parameters and locals; only containers are expanded.
_CONTAINER_KINDS = frozenset({2, 3, 4, 5, 10, 11, 19, 23})
_SNIPPET_CHARS = 160
_HOVER_CHARS = 1500
_SNIPPET_MAX_BYTES = 4 * 1024 * 1024
_NEARBY_LINES = 3  # a symbol not on the given line is looked for this many lines around it

_POSITION_METHODS = {
    "definition": "textDocument/definition",
    "references": "textDocument/references",
    "hover": "textDocument/hover",
}
ACTIONS = (*_POSITION_METHODS, "symbols")

_SYMBOL_KINDS = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class", 6: "method", 7: "property",
    8: "field", 9: "constructor", 10: "enum", 11: "interface", 12: "function", 13: "variable",
    14: "constant", 15: "string", 16: "number", 17: "boolean", 18: "array", 19: "object", 20: "key",
    21: "null", 22: "enum member", 23: "struct", 24: "event", 25: "operator", 26: "type parameter",
}


class NavigationError(Exception):
    """A query that could not run; the message is written for the model."""


def run(svc: Any, action: str, path: str, *, symbol: Optional[str] = None, line: Optional[int] = None,
        column: Optional[int] = None, query: Optional[str] = None) -> dict[str, Any]:
    """Run one navigation query for ``path`` (absolute) and return the tool payload; raises
    :class:`NavigationError` with a model-facing reason on any failure."""
    if action not in ACTIONS:
        raise NavigationError(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise NavigationError(f"{path} is not a file")
    if (reason := svc.gate_reason(abs_path)) is not None:
        raise NavigationError(f"LSP unavailable for {path}: {reason}")
    method, params, at_line = _request(action, abs_path, symbol=symbol, line=line, column=column, query=query)
    server_id = svc._server_for(abs_path).server_id  # gate_reason guarantees a server
    try:
        raw, folders = svc._loop.run(_query_async(svc, abs_path, method, params),
                                     timeout=REQUEST_TIMEOUT + SPAWN_ALLOWANCE)
    except NavigationError:
        raise
    except TimeoutError:
        raise NavigationError(f"{server_id} did not answer {action} within {REQUEST_TIMEOUT:.0f}s "
                              "(likely still indexing the workspace); retry shortly") from None
    except LSPRequestError as e:
        raise NavigationError(f"{server_id} rejected {method}: {e}") from None
    except (LSPProtocolError, RuntimeError, OSError) as e:  # dead server, stopped loop, unreadable file
        raise NavigationError(f"{server_id} {action} failed: {e}") from None
    root = find_git_worktree(abs_path) or os.path.dirname(abs_path)
    payload: dict[str, Any] = {"action": action, "server": server_id, "root": root}
    if at_line is not None:
        payload["at"] = f"{_display(abs_path, root)}:{at_line}"
    if action == "hover":
        text = _hover_text(raw)
        payload["hover"] = text or "(no hover information)"
        return payload
    if action == "symbols" and not query:
        rows = [(abs_path, ln, ch, label) for ln, ch, label in _outline(raw)]
    elif action == "symbols":
        rows = _best_matches(_workspace_symbols(raw), query or "")
    else:
        rows = [(p, ln, ch, None) for p, ln, ch in _locations(raw)]
        if action == "references":
            rows.sort(key=lambda r: (r[0], r[1], r[2]))
    rows = _own_root_only(rows, root, folders)
    payload["total"] = len(rows)
    payload["results"] = _format(rows[:MAX_RESULTS], root, outline=action == "symbols" and not query)
    if len(rows) > MAX_RESULTS:
        payload["note"] = f"{len(rows) - MAX_RESULTS} more not shown; narrow the query"
    elif not rows:
        payload["note"] = "no results"
    return payload


# ---- request building ----

def _request(action: str, abs_path: str, *, symbol: Optional[str], line: Optional[int], column: Optional[int],
             query: Optional[str]) -> tuple[str, dict[str, Any], Optional[int]]:
    """``(method, params, 1-based line queried)`` for one action."""
    doc = {"uri": file_uri(abs_path)}
    if action == "symbols":
        if query:
            return "workspace/symbol", {"query": query}, None
        return "textDocument/documentSymbol", {"textDocument": doc}, None
    position, at_line = _position(_read_text(abs_path), symbol=symbol, line=line, column=column)
    params: dict[str, Any] = {"textDocument": doc, "position": position}
    if action == "references":
        params["context"] = {"includeDeclaration": True}
    return _POSITION_METHODS[action], params, at_line


def _symbol_re(symbol: str) -> re.Pattern:
    return re.compile(rf"(?<![\w$]){re.escape(symbol)}(?![\w$])")


def _position(text: str, *, symbol: Optional[str], line: Optional[int], column: Optional[int]) -> tuple[dict[str, int], int]:
    """LSP position (0-based line, UTF-16 character) for a 1-based ``line`` and either a 1-based
    ``column`` or the ``symbol`` occurring on (or within a few lines of) that line; without ``line``
    the symbol's first occurrence in the file."""
    lines = text.split("\n")
    symbol = (symbol or "").strip() or None
    pattern = _symbol_re(symbol) if symbol else None
    if line is None:
        if pattern is None:
            raise NavigationError("give symbol (the identifier), line (1-based), or both")
        line = next((i + 1 for i, src in enumerate(lines) if pattern.search(src)), None)
        if line is None:
            raise NavigationError(f"{symbol!r} does not occur in the file")
    if not 1 <= line <= len(lines):
        raise NavigationError(f"line {line} is out of range: the file has {len(lines)} lines")
    if column is not None:
        idx = min(max(column - 1, 0), len(lines[line - 1]))
    elif pattern is not None:
        line, idx = _nearest_occurrence(lines, line, pattern, symbol or "")
    else:
        src = lines[line - 1]
        idx = len(src) - len(src.lstrip())
    prefix = lines[line - 1][:idx]
    return {"line": line - 1, "character": len(prefix.encode("utf-16-le")) // 2}, line


def _nearest_occurrence(lines: list[str], line: int, pattern: re.Pattern, symbol: str) -> tuple[int, int]:
    """``(1-based line, index)`` of ``symbol`` on ``line``, else on the nearest line within
    :data:`_NEARBY_LINES` (models are often off by one)."""
    for delta in range(_NEARBY_LINES + 1):
        for cand in ((line,) if delta == 0 else (line - delta, line + delta)):
            if 1 <= cand <= len(lines) and (m := pattern.search(lines[cand - 1])):
                return cand, m.start()
    raise NavigationError(f"{symbol!r} does not occur on line {line} (or within {_NEARBY_LINES} lines of it)")


def _read_text(abs_path: str) -> str:
    try:
        return Path(abs_path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise NavigationError(f"cannot read {abs_path}: {e}") from None


# ---- loop side ----

async def _query_async(svc: Any, abs_path: str, method: str, params: dict[str, Any]) -> tuple[Any, list[str]]:
    """Send one request through the file's client: ``(result, the client's workspace folders)``."""
    client = await svc._get_or_spawn(abs_path)
    if client is None:
        raise NavigationError("the language server could not start (see `hermes lsp status`)")
    tail = svc._edit_checks.tail.get(abs_path)
    if tail is not None and not tail.task.done():
        await asyncio.wait({tail.task}, timeout=EDIT_CHECK_WAIT)  # never cancels the check
    disk = _read_text(abs_path)
    if client.document_text(abs_path) != disk:
        await client.open_file(abs_path, language_id=language_id_for(abs_path, svc._server_for(abs_path)))
    await client.wait_settled(REQUEST_TIMEOUT)  # rust-analyzer: no answers from a half-loaded crate graph
    if method in _WORKSPACE_WIDE:
        if (settle := WORKSPACE_SETTLE - client.folder_age(abs_path)) > 0:
            await asyncio.sleep(settle)
        await client.wait_progress_idle(REQUEST_TIMEOUT)
    try:
        return await client.request(method, params, timeout=REQUEST_TIMEOUT), list(client.workspace_folders)
    finally:
        svc._touch(client)


# ---- result shaping ----

def _start(rng: Any) -> tuple[int, int]:
    start = rng.get("start") if isinstance(rng, dict) else None
    if not isinstance(start, dict):
        return 0, 0
    return int(start.get("line") or 0), int(start.get("character") or 0)


def _locations(raw: Any) -> list[tuple[str, int, int]]:
    """``Location | Location[] | LocationLink[] | null`` → ``(path, line0, utf16 char)``, deduped in order."""
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    out: list[tuple[str, int, int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri") or item.get("targetUri")
        rng = item.get("range") or item.get("targetSelectionRange") or item.get("targetRange")
        if isinstance(uri, str):
            loc = (uri_to_path(uri), *_start(rng))
            if loc not in out:
                out.append(loc)
    return out


def _workspace_symbols(raw: Any) -> list[tuple[str, int, int, str]]:
    """``SymbolInformation[] | WorkspaceSymbol[]`` → ``(path, line0, char, label)``."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        loc = item.get("location") if isinstance(item, dict) else None
        if not isinstance(loc, dict) or not isinstance(loc.get("uri"), str):
            continue
        out.append((uri_to_path(loc["uri"]), *_start(loc.get("range")), _label(item)))
    return out


def _best_matches(rows: list, query: str) -> list:
    """Order workspace symbols exact → prefix → substring, and drop the servers' fuzzy subsequence
    matches (pyright answers "LSPService" with test_unreadable_subtree_...) when any real match exists."""
    q = query.casefold()

    def rank(row: tuple) -> int:
        name = row[3].split(" (in ", 1)[0].rsplit(" ", 1)[-1]
        folded = name.casefold()
        return next(i for i, hit in enumerate((name == query, folded == q, folded.startswith(q), q in folded, True)) if hit)
    ranked = sorted(rows, key=rank)
    return [r for r in ranked if rank(r) < 4] or ranked


def _outline(raw: Any, depth: int = 0) -> list[tuple[int, int, str]]:
    """``DocumentSymbol[]`` (hierarchical, children indented) or ``SymbolInformation[]`` → ``(line0, char, label)``."""
    out: list[tuple[int, int, str]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        rng = item.get("selectionRange") or item.get("range") or (item.get("location") or {}).get("range")
        out.append((*_start(rng), "  " * depth + _label(item)))
        if item.get("kind") in _CONTAINER_KINDS:
            out.extend(_outline(item.get("children"), depth + 1))
    return out


def _label(item: dict[str, Any]) -> str:
    kind = item.get("kind")
    label = f"{_SYMBOL_KINDS.get(kind, 'symbol') if isinstance(kind, int) else 'symbol'} {item.get('name', '?')}"
    return f"{label} (in {item['containerName']})" if item.get("containerName") else label


def _own_root_only(rows: list, root: str, folders: list[str]) -> list:
    """Drop rows from ANOTHER git worktree the server serves (a shared multi-root pyright attached to
    sibling worktrees of one repo): a query from one worktree must not answer with a sibling's copy of
    the same code.  Rows outside any served worktree (stdlib, site-packages, typeshed) stay."""
    def foreign(path: str) -> bool:
        wt = find_git_worktree(path) if os.path.isabs(path) else None
        return wt is not None and wt in served
    served = {find_git_worktree(f) for f in folders} - {root, None}
    return [r for r in rows if not foreign(r[0])] if served else rows


def _display(path: str, root: str) -> str:
    return os.path.relpath(path, root) if os.path.isabs(path) and is_inside_workspace(path, root) else path


def _format(rows: list, root: str, *, outline: bool) -> list[str]:
    lines_cache: dict[str, Optional[list[str]]] = {}
    out = []
    for path, line0, char, label in rows:
        src = _source_line(path, line0, lines_cache)
        col = _utf16_to_index(src or "", char) + 1
        if outline:
            out.append(f"{line0 + 1}: {label}")
            continue
        loc = f"{_display(path, root)}:{line0 + 1}:{col}"
        detail = label if label is not None else (src or "").strip()[:_SNIPPET_CHARS]
        out.append(f"{loc}  {detail}" if detail else loc)
    return out


def _source_line(path: str, line0: int, cache: dict[str, Optional[list[str]]]) -> Optional[str]:
    """One source line for the snippet; None for unreadable, oversized or read-denied files."""
    if path not in cache:
        cache[path] = None
        try:
            from agent.file_safety import get_read_block_error
            if os.path.isabs(path) and os.path.getsize(path) <= _SNIPPET_MAX_BYTES and get_read_block_error(path) is None:
                cache[path] = Path(path).read_text(encoding="utf-8-sig", errors="replace").split("\n")
        except OSError:
            pass
    lines = cache[path]
    return lines[line0] if lines is not None and 0 <= line0 < len(lines) else None


def _utf16_to_index(text: str, units: int) -> int:
    """Python index of the UTF-16 offset ``units`` in ``text`` (clamped to the line)."""
    seen = 0
    for i, ch in enumerate(text):
        if seen >= units:
            return i
        seen += 2 if ord(ch) > 0xFFFF else 1
    return len(text)


def _hover_text(raw: Any) -> str:
    """``Hover.contents`` (MarkupContent, MarkedString or a list of them) → plain text, capped."""
    contents = raw.get("contents") if isinstance(raw, dict) else None
    parts = contents if isinstance(contents, list) else [contents]
    texts = []
    for part in parts:
        value = part.get("value") if isinstance(part, dict) else part
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    text = "\n\n".join(texts)
    return text if len(text) <= _HOVER_CHARS else text[:_HOVER_CHARS - 1] + "…"


__all__ = ["ACTIONS", "MAX_RESULTS", "NavigationError", "run"]
