"""Dated ``scheduled`` cards: when a parked card wakes, and what it wakes into.

A card in ``scheduled`` carries an optional wake time (``tasks.scheduled_until``,
epoch seconds) and a wake mode (``tasks.scheduled_then``):

- ``start``: the card goes to ``ready`` (``todo`` while a parent is still open)
  and the dispatcher runs it like any other ready card.
- ``ask``: the card goes to a sticky ``blocked`` with a ``blocked`` event
  (``{"reason": ..., "woke_from": "scheduled"}``), so a human sees it. Ask-mode
  waits while a parent is open: there is nothing to decide yet.

Cards parked before the columns existed keep working: with no column value the
wake date is read from ``until YYYY-MM-DD`` in the latest ``scheduled`` event
reason, and a card with no date at all wakes in ask mode after
``kanban.schedule_recheck_days`` (default 7).

The dispatcher calls :func:`wake_due_scheduled` once per tick before promotion
and claiming, so a start-mode card can spawn on the tick it wakes.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

VALID_SCHEDULE_THEN = ("start", "ask")
DEFAULT_SCHEDULE_THEN = "ask"
DEFAULT_RECHECK_DAYS = 7
# Same pattern the out-of-tree orchestrator watch script used, so legacy reasons
# ("until 2026-10-04: check X") keep their date.
UNTIL_RE = re.compile(r"\buntil (\d{4}-\d{2}-\d{2})\b")
_UNTIL_FORMATS = ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d")
_REASON_CAP = 500


def parse_until(value) -> Optional[int]:
    """``YYYY-MM-DD`` (local midnight) or ``YYYY-MM-DDTHH:MM`` (local time) ->
    epoch seconds. Ints pass through; empty/None -> None; anything else raises."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    raw = str(value).strip()
    for fmt in _UNTIL_FORMATS:
        try:
            return int(time.mktime(datetime.strptime(raw, fmt).timetuple()))
        except ValueError:
            continue
    raise ValueError(f"until must be YYYY-MM-DD or YYYY-MM-DDTHH:MM, got {value!r}")


def normalize_then(value) -> Optional[str]:
    """``start`` | ``ask`` (case-insensitive); empty/None -> None; else raises."""
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in VALID_SCHEDULE_THEN:
        return raw
    raise ValueError(f"then must be one of {', '.join(VALID_SCHEDULE_THEN)}, got {value!r}")


def until_from_reason(reason: Optional[str]) -> Optional[int]:
    """Epoch of local midnight for ``until YYYY-MM-DD`` in a free-text reason."""
    m = UNTIL_RE.search(reason or "")
    if not m:
        return None
    try:
        return parse_until(m.group(1))
    except ValueError:  # "until 2026-13-45": free text, not a date
        return None


def resolve_schedule(
    reason: Optional[str], until=None, then=None,
) -> tuple[Optional[int], Optional[str]]:
    """Validated ``(scheduled_until, scheduled_then)`` for a schedule request.

    No explicit ``until`` -> the reason's ``until YYYY-MM-DD`` (if any) becomes
    the stored date. A dated card defaults to ``ask`` (the legacy behaviour).
    ``start`` needs a date: an undated start would silently run a week later.
    """
    until_ts = parse_until(until)
    mode = normalize_then(then)
    if until_ts is None:
        until_ts = until_from_reason(reason)
    if until_ts is None:
        if mode == "start":
            raise ValueError("then=start needs a date (until YYYY-MM-DD[THH:MM])")
        return None, None
    return until_ts, mode or DEFAULT_SCHEDULE_THEN


def format_until(ts: Optional[int]) -> str:
    """``YYYY-MM-DD`` at local midnight, else ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return ""
    lt = time.localtime(ts)
    if lt.tm_hour == 0 and lt.tm_min == 0:
        return time.strftime("%Y-%m-%d", lt)
    return time.strftime("%Y-%m-%d %H:%M", lt)


def configured_recheck_days() -> int:
    """``kanban.schedule_recheck_days`` (default 7; ``0`` turns the undated
    recheck off; unreadable config fails open to the default)."""
    try:
        from hermes_cli.config import load_config
        raw = ((load_config() or {}).get("kanban") or {}).get("schedule_recheck_days")
        return max(0, int(raw)) if raw is not None else DEFAULT_RECHECK_DAYS
    except Exception:
        return DEFAULT_RECHECK_DAYS


@dataclass(frozen=True)
class Schedule:
    """Effective wake plan of one ``scheduled`` card."""

    task_id: str
    wake_at: Optional[int]  # epoch seconds; None = never (undated, recheck off)
    then: str               # start | ask
    source: str             # column | reason | undated
    reason: str             # latest ``scheduled`` event reason ("" when none)

    @property
    def label(self) -> str:
        if self.wake_at is None:
            return "undated"
        if self.source == "undated":
            return "undated, recheck"
        return f"due {format_until(self.wake_at)}"


def _latest_scheduled_event(conn: sqlite3.Connection, task_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT payload, created_at FROM task_events WHERE task_id = ? AND kind = 'scheduled' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()


def _event_reason(ev: Optional[sqlite3.Row]) -> str:
    if ev is None:
        return ""
    try:
        payload = json.loads(ev["payload"] or "{}") or {}
    except (TypeError, ValueError):
        return ""
    return str(payload.get("reason") or "") if isinstance(payload, dict) else ""


def schedule_for(
    conn: sqlite3.Connection, task_id: str, *,
    scheduled_until: Optional[int] = None, scheduled_then: Optional[str] = None,
    recheck_days: Optional[int] = None,
) -> Schedule:
    """Wake plan for a scheduled card: column first, then the legacy reason date,
    then ``recheck_days`` after the latest ``scheduled`` event (ask)."""
    ev = _latest_scheduled_event(conn, task_id)
    reason = _event_reason(ev)
    if scheduled_until:
        return Schedule(task_id, int(scheduled_until), scheduled_then or DEFAULT_SCHEDULE_THEN, "column", reason)
    legacy = until_from_reason(reason)
    if legacy is not None:
        return Schedule(task_id, legacy, DEFAULT_SCHEDULE_THEN, "reason", reason)
    days = recheck_days if recheck_days is not None else configured_recheck_days()
    if days <= 0:
        return Schedule(task_id, None, DEFAULT_SCHEDULE_THEN, "undated", reason)
    since = int(ev["created_at"]) if ev is not None else 0
    return Schedule(task_id, since + days * 86400, DEFAULT_SCHEDULE_THEN, "undated", reason)


def schedule_label(conn: sqlite3.Connection, task) -> str:
    """Display string for a scheduled card, e.g. ``⏱ → 2026-10-03 start``."""
    s = schedule_for(
        conn, task.id, scheduled_until=getattr(task, "scheduled_until", None),
        scheduled_then=getattr(task, "scheduled_then", None),
    )
    if s.wake_at is None:
        return "⏱ undated (no recheck)"
    if s.source == "undated":
        return f"⏱ → {format_until(s.wake_at)} {s.then} (undated recheck)"
    return f"⏱ → {format_until(s.wake_at)} {s.then}"


def due_scheduled(
    conn: sqlite3.Connection, now: Optional[float] = None, *, recheck_days: Optional[int] = None,
) -> list[Schedule]:
    """Scheduled cards whose wake time has passed (parents not yet considered)."""
    now = time.time() if now is None else now
    days = recheck_days if recheck_days is not None else configured_recheck_days()
    due = []
    for row in conn.execute(
        "SELECT id, scheduled_until, scheduled_then FROM tasks WHERE status = 'scheduled' ORDER BY created_at"
    ).fetchall():
        s = schedule_for(
            conn, row["id"], scheduled_until=row["scheduled_until"],
            scheduled_then=row["scheduled_then"], recheck_days=days,
        )
        if s.wake_at is not None and s.wake_at <= now:
            due.append(s)
    return due


def _wake_one(conn: sqlite3.Connection, s: Schedule) -> Optional[str]:
    """Apply one wake inside the caller's write txn; returns the new status or
    None when the card moved on (no longer scheduled) or must keep waiting."""
    from hermes_cli import kanban_db as kb

    parents_done = kb._parents_satisfied(conn, s.task_id)
    clear = "scheduled_until = NULL, scheduled_then = NULL"
    if s.then == "ask":
        if not parents_done:
            return None
        if conn.execute(
            f"UPDATE tasks SET status = 'blocked', {clear} WHERE id = ? AND status = 'scheduled'",
            (s.task_id,),
        ).rowcount != 1:
            return None
        reason = f"woke from scheduled ({s.label}): {s.reason}".rstrip(": ")
        kb._append_event(conn, s.task_id, "blocked", {
            "reason": reason[:_REASON_CAP], "woke_from": "scheduled",
        })
        return "blocked"
    landing = "ready" if parents_done else "todo"
    kb._reclaim_dangling_run(
        conn, s.task_id, statuses=("scheduled",), now=int(time.time()),
        note="invariant recovery on scheduled wake",
    )
    if conn.execute(
        f"UPDATE tasks SET status = ?, current_run_id = NULL, consecutive_failures = 0, "
        f"last_failure_error = NULL, {clear} WHERE id = ? AND status = 'scheduled'",
        (landing, s.task_id),
    ).rowcount != 1:
        return None
    # ``unblocked`` (not a new kind) so every reader that pairs blocked/unblocked
    # events (sticky-block detection, resume status) treats the wake as a release.
    kb._append_event(conn, s.task_id, "unblocked", {
        "status": landing, "woke_from": "scheduled", "then": "start", "schedule": s.label,
    })
    return landing


def wake_due_scheduled(
    conn: sqlite3.Connection, *, now: Optional[float] = None,
    recheck_days: Optional[int] = None, dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Wake every due scheduled card. Returns ``[(task_id, new_status)]``.

    ``dry_run`` writes nothing and reports the status each card WOULD take
    (ask-mode cards with an open parent are left out either way). Opens its own
    IMMEDIATE txn — call OUTSIDE any write txn.
    """
    from hermes_cli import kanban_db as kb

    due = due_scheduled(conn, now, recheck_days=recheck_days)
    if not due:
        return []
    if dry_run:
        out = []
        for s in due:
            parents_done = kb._parents_satisfied(conn, s.task_id)
            if s.then == "ask":
                if parents_done:
                    out.append((s.task_id, "blocked"))
            else:
                out.append((s.task_id, "ready" if parents_done else "todo"))
        return out
    woke: list[tuple[str, str]] = []
    with kb.write_txn(conn):
        for s in due:
            status = _wake_one(conn, s)
            if status:
                woke.append((s.task_id, status))
    return woke
