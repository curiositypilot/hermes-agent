"""Hold streaks: a held card's hold event is written once per streak, not once per tick.

The dispatcher re-checks a held ready/review card every tick. Writing its hold event each
time buried the card's real events under one row per minute (~1440 a day, intake t_6190ed34).
A streak lasts while the newest event outside :data:`_HOLD_STREAK_NOISE` is the same hold.

Split out of ``hermes_cli.kanban_db_dispatch``; ``kanban_db`` is late-imported (cycle).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_routing import RouteDecision

# Kinds that do not end a hold streak. An unknown future kind is absent on
# purpose: an allowlist fails safe (an extra row, never a missing hold).
_HOLD_STREAK_NOISE = frozenset({
    "commented", "heartbeat", "linked", "unlinked", "attached", "edited",
    "terminal_worker_reaped",
})


def _last_streak_event(conn: sqlite3.Connection, task_id: str) -> tuple[Optional[str], dict]:
    """``(kind, payload)`` of the newest event on the card outside ``_HOLD_STREAK_NOISE``."""
    from hermes_cli import kanban_db as _kb

    noise = tuple(sorted(_HOLD_STREAK_NOISE))
    last = conn.execute(
        "SELECT kind, payload FROM task_events "
        f"WHERE task_id = ? AND kind NOT IN ({','.join('?' * len(noise))}) ORDER BY id DESC LIMIT 1",
        (task_id, *noise),
    ).fetchone()
    return (None, {}) if last is None else (last["kind"], _kb._json_dict(last["payload"]))


def _record_routing_hold(
    conn: sqlite3.Connection, task_id: str, route: Optional[RouteDecision],
    reason: str = "tier_exhausted",
) -> None:
    """``routing_held`` once per streak: same reason, note and tier; noise events ignored."""
    from hermes_cli import kanban_db as _kb

    try:
        payload: dict[str, Any] = {"reason": reason}
        if route is not None:
            payload.update({k: v for k, v in route.event_payload().items()
                            if k in ("lane", "requested_tier", "skipped", "note")})
            if route.retry_at:
                payload["retry_at"] = int(route.retry_at)
        kind, prev = _last_streak_event(conn, task_id)
        if kind == "routing_held" and all(
            prev.get(k) == payload.get(k) for k in ("reason", "note", "requested_tier")
        ):
            return
        with _kb.write_txn(conn):
            _kb._append_event(conn, task_id, "routing_held", payload)
    except sqlite3.Error:
        _kb._log.debug("kanban routing: could not record hold for %s", task_id, exc_info=True)
