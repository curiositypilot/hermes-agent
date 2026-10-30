"""When a PR URL in a card's comments stops guarding its respawn (``active_pr``).

``check_respawn_guard`` refuses to re-spawn a ready card whose recent comment
carries a PR URL, so a crashed or reclaimed worker never opens a duplicate PR.
A handoff to another profile, or a clean dependency park of the worker itself,
is the intended resume on that PR and lifts the guard.

Split out of ``hermes_cli.kanban_db_dispatch``; ``kanban_db`` is late-imported.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def _clean_park_resume(
    conn: sqlite3.Connection, task_id: str, comment_at: int,
    latest_run: Optional[sqlite3.Row],
) -> bool:
    """Whether the newest PR comment was followed by a clean park and its resume.

    True when the latest ended run is ``blocked``, ended no earlier than the
    comment (same second counts: a worker comments, then blocks), and a
    ``promoted``/``unblocked`` event followed that run. A crash, reclaim,
    timeout or spawn failure is never ``blocked``, so it stays guarded; an
    unreadable ``ended_at`` stays guarded too (fail closed). The handoff half
    (:func:`_is_handoff_event`) is checked by the dispatcher first.
    """
    if latest_run is None or latest_run["outcome"] != "blocked":
        return False
    from hermes_cli import kanban_db as _kb

    parked_at = _kb._to_epoch(latest_run["ended_at"])
    if parked_at is None or parked_at < comment_at:
        return False
    return conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND created_at >= ? "
        "AND kind IN ('promoted', 'unblocked') LIMIT 1",
        (task_id, parked_at),
    ).fetchone() is not None


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed."""
    if kind != "assigned":
        return True
    from hermes_cli import kanban_db as _kb

    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to
