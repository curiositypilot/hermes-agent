"""Token usage of a Kanban run, read from the worker's session row at run close.

A dispatcher-spawned worker stamps ``worker_session_id`` into the metadata of its
terminal board call (``tools/kanban_tools.py::_stamp_worker_session_metadata``).
The worker's own ``state.db`` (under the run's profile home) already accumulates
per-session token counters, so the run close copies them into
``task_runs.metadata.usage`` instead of adding new plumbing.

Compression rotates the session id mid-run (``HERMES_SESSION_ID`` becomes the
child id), so the usage sums the session and its ``kanban``-sourced ancestors.

Everything here is best-effort and read-only: a missing profile, DB or row
yields ``None`` and the run closes exactly as before.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# Upper bound on the compression chain walked per run; real chains are 1-3 long.
_MAX_LINEAGE = 32

_USAGE_COLUMNS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def profile_state_db(profile: Optional[str]) -> Optional[Path]:
    """``state.db`` of ``profile``'s home, or None when it cannot be resolved."""
    try:
        from hermes_cli.profiles import get_profile_dir

        path = get_profile_dir(profile or "default") / "state.db"
    except Exception:
        return None
    return path if path.is_file() else None


def read_session_usage(db_path: Path, session_id: str) -> Optional[dict]:
    """Usage dict for ``session_id`` plus its kanban ancestors in ``db_path``.

    Returns ``{model, provider, input_tokens, output_tokens, cached_tokens,
    cache_write_tokens, reasoning_tokens, api_calls, session_ids}`` or None when
    the row does not exist. ``cached_tokens`` is the cache-read count.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    try:
        totals = dict.fromkeys(_USAGE_COLUMNS, 0)
        api_calls = 0
        model = provider = None
        seen: list[str] = []
        sid: Optional[str] = session_id
        while sid and sid not in seen and len(seen) < _MAX_LINEAGE:
            row = conn.execute(
                "SELECT id, source, parent_session_id, model, billing_provider, api_call_count, "
                + ", ".join(_USAGE_COLUMNS) + " FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            if row is None or (seen and row["source"] != "kanban"):
                break
            seen.append(sid)
            for col in _USAGE_COLUMNS:
                totals[col] += int(row[col] or 0)
            api_calls += int(row["api_call_count"] or 0)
            # The newest session in the chain names the route the run ended on.
            model = model or row["model"]
            provider = provider or row["billing_provider"]
            sid = row["parent_session_id"]
    except sqlite3.Error as exc:
        _log.debug("kanban run usage: read of %s failed (%s)", db_path, exc)
        return None
    finally:
        conn.close()
    if not seen:
        return None
    return {
        "model": model,
        "provider": provider,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "cached_tokens": totals["cache_read_tokens"],
        "cache_write_tokens": totals["cache_write_tokens"],
        "reasoning_tokens": totals["reasoning_tokens"],
        "api_calls": api_calls,
        "session_ids": seen,
    }


def usage_for_run(profile: Optional[str], metadata: Optional[dict]) -> Optional[dict]:
    """Usage for a run whose metadata names ``worker_session_id``; None otherwise."""
    if not isinstance(metadata, dict):
        return None
    session_id = metadata.get("worker_session_id")
    if not session_id or not isinstance(session_id, str):
        return None
    db_path = profile_state_db(profile)
    if db_path is None:
        return None
    return read_session_usage(db_path, session_id)


def with_run_usage(profile: Optional[str], metadata: Optional[dict]) -> Optional[dict]:
    """``metadata`` with ``usage`` added when the worker session is readable.

    Never raises: run close must not fail on accounting.
    """
    try:
        if not isinstance(metadata, dict) or "usage" in metadata:
            return metadata
        usage = usage_for_run(profile, metadata)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban run usage: lookup failed (%s)", exc)
        return metadata
    return {**metadata, "usage": usage} if usage else metadata
