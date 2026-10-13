"""Retry gate: a re-run after a failed attempt must be an iteration, not a repetition.

``claim_task`` (the single ``ready -> running`` choke point) calls :func:`gate_claim` inside its
write txn. Every claim records an input fingerprint (title, body, model, assignee, skills, runtime
limit, parents) on its ``claimed`` event. When the newest attempt that actually ran the card ended
as a failure (:data:`FAILED_OUTCOMES`), the next claim needs evidence that the input changed since
then: a fingerprint field differs, a comment was posted after the failed run ended, or a reviewer's
``changes_requested`` reason exists. With evidence, a ``retry_iteration`` event records what
changed and the note (why); without, the card is parked ``blocked`` (sticky, ``needs_input``) with
reason ``retry without change`` and only an unblock plus a real change releases it.

Infra retries are not iterations and pass untouched: crash, timeout, rate-limit, spawn failure,
``gave_up``, reclaim and stale runs are skipped when looking for the newest attempt, as are
``transient`` and ``dependency`` blocks; ``tier_exhausted`` holds never claim, so never reach here.
A run with no ``claimed`` event (operator block on a ready card, pre-gate history) is not an attempt.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Optional

RETRY_WITHOUT_CHANGE = "retry without change"

# Outcomes of an attempt that ran the card and did not land; the next run must change something.
FAILED_OUTCOMES = frozenset({"blocked", "changes_requested"})
# Restarts caused by the host, provider or supervisor, not by the card's input. A timeout is the
# supervisor's kill, like a crash: the failure breaker (``kanban.failure_limit``) owns its budget.
INFRA_OUTCOMES = frozenset({
    "crashed", "timed_out", "rate_limited", "spawn_failed", "gave_up", "reclaimed", "stale",
})
INFRA_BLOCK_KINDS = frozenset({"transient", "dependency"})
_BLOCK_EVENT_KINDS = ("blocked", "dependency_wait", "block_loop_detected")
_NOTE_LIMIT = 300


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def input_fingerprint(conn: sqlite3.Connection, task_id: str) -> dict[str, str]:
    """Per-field digests of everything a worker is handed besides comments."""
    row = conn.execute(
        "SELECT title, body, assignee, model_override, provider_override, complexity, skills, "
        "max_runtime_seconds FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    parents = conn.execute(
        "SELECT p.id, p.completed_at FROM task_links l JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? ORDER BY p.id", (task_id,),
    ).fetchall()
    return {
        "title": _digest(row["title"]),
        "body": _digest(row["body"]),
        "model": _digest([row["model_override"], row["provider_override"], row["complexity"]]),
        "assignee": _digest(row["assignee"]),
        "skills": _digest(row["skills"]),
        "max_runtime": _digest(row["max_runtime_seconds"]),
        "parents": _digest([[p["id"], p["completed_at"]] for p in parents]),
    }


def _block_kind(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[str]:
    from hermes_cli.kanban_db import _json_dict

    row = conn.execute(
        f"SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
        f"AND kind IN ({', '.join('?' * len(_BLOCK_EVENT_KINDS))}) ORDER BY id DESC LIMIT 1",
        (task_id, run_id, *_BLOCK_EVENT_KINDS),
    ).fetchone()
    return _json_dict(row["payload"]).get("kind") if row else None


def _last_attempt(conn: sqlite3.Connection, task_id: str) -> Optional[sqlite3.Row]:
    """Newest ended run that was not an infra restart."""
    for run in conn.execute(
        "SELECT id, outcome, summary FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC", (task_id,),
    ):
        if run["outcome"] in INFRA_OUTCOMES:
            continue
        if run["outcome"] == "blocked" and _block_kind(conn, task_id, run["id"]) in INFRA_BLOCK_KINDS:
            continue
        return run
    return None


def _claimed_fingerprint(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[dict]:
    from hermes_cli.kanban_db import _json_dict

    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'claimed' "
        "ORDER BY id DESC LIMIT 1", (task_id, run_id),
    ).fetchone()
    fingerprint = _json_dict(row["payload"]).get("input_fingerprint") if row else None
    return fingerprint if isinstance(fingerprint, dict) else None


def _comment_note(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[str]:
    """Newest comment posted after the failed run's last event (a worker's own
    comments precede it), as the 'why' note; None when nobody commented."""
    last_event = conn.execute(
        "SELECT MAX(id) FROM task_events WHERE task_id = ? AND run_id = ?", (task_id, run_id),
    ).fetchone()[0]
    if last_event is None:
        return None
    commented = conn.execute(
        "SELECT created_at FROM task_events WHERE task_id = ? AND kind = 'commented' AND id > ? "
        "ORDER BY id ASC LIMIT 1", (task_id, last_event),
    ).fetchone()
    if commented is None:
        return None
    row = conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ? ORDER BY id DESC LIMIT 1",
        (task_id, commented["created_at"]),
    ).fetchone()
    return (row["body"] if row else "").strip()[:_NOTE_LIMIT] or "(comment)"


def gate_claim(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """Run inside the claim txn. Returns the extra payload for the ``claimed``
    event, or None after parking the card ``blocked`` for a retry without change."""
    from hermes_cli.kanban_db import _append_event

    current = input_fingerprint(conn, task_id)
    extra = {"input_fingerprint": current}
    attempt = _last_attempt(conn, task_id)
    if attempt is None or attempt["outcome"] not in FAILED_OUTCOMES:
        return extra
    before = _claimed_fingerprint(conn, task_id, attempt["id"])
    if before is None and attempt["outcome"] != "changes_requested":
        return extra  # never claimed (operator block on a ready card) or claimed before this gate
    before = before or {}
    changed = sorted(k for k, v in current.items() if k in before and before[k] != v)
    note = _comment_note(conn, task_id, attempt["id"])
    if note is not None:
        changed.append("comments")
    if attempt["outcome"] == "changes_requested" and attempt["summary"]:
        changed.append("review_reason")
        note = note or attempt["summary"].strip()[:_NOTE_LIMIT]
    info = {"after_run": attempt["id"], "after_outcome": attempt["outcome"]}
    if changed:
        _append_event(conn, task_id, "retry_iteration", {**info, "changed": changed, "note": note})
        return extra
    conn.execute(
        "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' WHERE id = ? AND status = 'ready'",
        (task_id,),
    )
    _append_event(conn, task_id, "blocked", {
        **info, "reason": RETRY_WITHOUT_CHANGE, "kind": "needs_input", "gate": "retry",
        "remedy": "change the body, model, assignee or parents, or comment what changed and why, then unblock",
    })
    return None
