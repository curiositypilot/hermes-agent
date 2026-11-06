"""``lsp`` tool: code navigation (definition, references, hover, symbols) through the language
servers ``agent.lsp`` already runs for post-write diagnostics.

Same gate as the diagnostics (``lsp.enabled``, git workspace, ``lsp.exclude_roots``, workspace trust:
an untrusted checkout only gets the servers that never load project code), same clients.  Local
backends only: the servers run on the host and cannot see a container's or remote host's files.
Fails open: every failure is a one-line error the model can act on (fall back to text search).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from tools.registry import registry, tool_error

LSP_SCHEMA = {
    "name": "lsp",
    "description": (
        "Navigate code through the project's language server (pyright, tsserver, rust-analyzer, clangd, ...; "
        "local git checkouts). Answers by meaning, not text: no hits from comments, strings or same-named "
        "symbols. definition / references / hover: give path and symbol (the identifier), plus the 1-based "
        "line it is on (omit line to use its first occurrence in the file). symbols: with query, find a name "
        "across the workspace (path = any source file of that language in the project); without query, an "
        "outline of path. Returns path:line:col plus the source line, at most 50 results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["definition", "references", "hover", "symbols"]},
            "path": {"type": "string", "description": "Source file to query (absolute, or relative to the working directory)."},
            "symbol": {"type": "string", "description": "The identifier to look up (definition / references / hover)."},
            "line": {"type": "integer", "description": "1-based line where symbol occurs."},
            "column": {"type": "integer", "description": "1-based column; only when symbol is ambiguous on its line."},
            "query": {"type": "string", "description": "symbols: name or prefix to search the workspace for."},
        },
        "required": ["action", "path"],
    },
}


def _check_lsp_reqs() -> bool:
    """File tools available and LSP not switched off for this profile (``lsp.enabled``)."""
    from hermes_cli.config import load_config_readonly
    from tools import check_file_requirements
    lsp_cfg = load_config_readonly().get("lsp")
    return check_file_requirements() and not (isinstance(lsp_cfg, dict) and lsp_cfg.get("enabled") is False)


def _int_or_none(value: Any, name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def lsp_tool(action: str, path: str, *, symbol: Optional[str] = None, line: Any = None, column: Any = None,
             query: Optional[str] = None, task_id: str = "default") -> str:
    from agent.file_safety import get_read_block_error
    from agent.lsp import get_service
    from agent.lsp.navigation import NavigationError, run
    from tools.file_tools import _get_file_ops
    from tools.file_tools_paths import _resolve_path_for_task

    if not isinstance(path, str) or not path.strip():
        return tool_error("lsp: 'path' is required (a source file in the project)")
    try:
        line_no, column_no = _int_or_none(line, "line"), _int_or_none(column, "column")
    except ValueError as e:
        return tool_error(f"lsp: {e}")
    if not _get_file_ops(task_id)._lsp_local_only():
        return tool_error("lsp: language servers run on the host and cannot see this terminal backend's files; "
                          "search the text instead")
    resolved = str(_resolve_path_for_task(path, task_id))
    if (blocked := get_read_block_error(resolved)) is not None:
        return tool_error(blocked)
    svc = get_service()
    if svc is None:
        return tool_error("lsp: LSP is disabled for this profile (lsp.enabled) or its config failed to load; "
                          "search the text instead")
    try:
        result = run(svc, action, resolved, symbol=symbol if isinstance(symbol, str) else None,
                     line=line_no, column=column_no, query=query if isinstance(query, str) else None)
    except NavigationError as e:
        return tool_error(f"lsp: {e}. Search the text instead if this persists.")
    return json.dumps(result, ensure_ascii=False)


def _handle_lsp(args: dict, **kw) -> str:
    return lsp_tool(
        action=args.get("action", ""), path=args.get("path", ""), symbol=args.get("symbol"),
        line=args.get("line"), column=args.get("column"), query=args.get("query"),
        task_id=kw.get("task_id") or "default",
    )


registry.register(name="lsp", toolset="file", schema=LSP_SCHEMA, handler=_handle_lsp, check_fn=_check_lsp_reqs,
                  emoji="🧭", max_result_size_chars=30_000)
