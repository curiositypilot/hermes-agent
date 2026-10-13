"""Retry gate (F5, "iterations, not repetitions"): a re-run after a failed attempt must
carry a change; infra restarts pass untouched.

Contracts:
* a crash / timeout / rate-limit restart (and a ``transient`` block) re-claims with nothing changed;
* a failed attempt (the worker blocked) re-queued with nothing changed is parked ``blocked``
  with reason ``retry without change`` instead of spawning, and stays parked across a
  bare unblock;
* a comment posted after the failed run, or a body / model / parent change, lets the
  retry through and records what changed and the note on a ``retry_iteration`` event;
* a reviewer's ``changes_requested`` reason is itself the note.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_retry_gate import RETRY_WITHOUT_CHANGE


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as c:
        yield c


def _ready_task(conn, body="do the thing"):
    tid = kb.create_task(conn, title="t", body=body, assignee="default")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    return tid


def _claim(conn, tid):
    return kb.claim_task(conn, tid, claimer="default:worker")


def _last_payload(conn, tid, kind):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (tid, kind),
    ).fetchone()
    return json.loads(row["payload"]) if row and row["payload"] else None


def _fail_with_block(conn, tid):
    """Run once and stop with a human-facing block, then get unblocked."""
    assert _claim(conn, tid) is not None
    assert kb.block_task(conn, tid, reason="tests fail", kind="needs_input")
    assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "ready"


def _end_run_as(conn, tid, outcome):
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome=outcome, error=outcome)
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL WHERE id=?", (tid,),
        )


@pytest.mark.parametrize("outcome", ["crashed", "timed_out", "rate_limited", "spawn_failed", "reclaimed"])
def test_infra_retry_passes_without_change(conn, outcome):
    tid = _ready_task(conn)
    assert _claim(conn, tid) is not None
    _end_run_as(conn, tid, outcome)

    assert _claim(conn, tid) is not None
    assert kb.get_task(conn, tid).status == "running"
    assert _last_payload(conn, tid, "retry_iteration") is None


def test_transient_block_retry_passes_without_change(conn):
    tid = _ready_task(conn)
    assert _claim(conn, tid) is not None
    assert kb.block_task(conn, tid, reason="registry 503", kind="transient")
    assert kb.unblock_task(conn, tid)

    assert _claim(conn, tid) is not None


def test_infra_restart_after_an_iteration_still_passes(conn):
    """A crash on a legitimate iteration is a restart of that iteration, not a repetition."""
    tid = _ready_task(conn)
    _fail_with_block(conn, tid)
    kb.add_comment(conn, tid, "mb", "use the staging DB this time")
    assert _claim(conn, tid) is not None
    _end_run_as(conn, tid, "crashed")

    assert _claim(conn, tid) is not None


def test_empty_retry_after_block_is_blocked(conn):
    tid = _ready_task(conn)
    _fail_with_block(conn, tid)

    assert _claim(conn, tid) is None
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.current_run_id is None
    payload = _last_payload(conn, tid, "blocked")
    assert payload["reason"] == RETRY_WITHOUT_CHANGE
    assert payload["after_outcome"] == "blocked"

    # Sticky: the dispatcher's promotion pass leaves it parked, and a bare unblock
    # walks straight back into the gate.
    kb.recompute_ready(conn)
    assert kb.get_task(conn, tid).status == "blocked"
    assert kb.unblock_task(conn, tid)
    assert _claim(conn, tid) is None
    assert kb.get_task(conn, tid).status == "blocked"


def test_empty_retry_after_untyped_block_is_blocked(conn):
    tid = _ready_task(conn)
    assert _claim(conn, tid) is not None
    assert kb.block_task(conn, tid, reason="gave up on the flaky fixture")
    assert kb.unblock_task(conn, tid)

    assert _claim(conn, tid) is None
    assert _last_payload(conn, tid, "blocked")["reason"] == RETRY_WITHOUT_CHANGE


def test_worker_comment_on_its_own_failed_run_is_not_a_note(conn):
    tid = _ready_task(conn)
    assert _claim(conn, tid) is not None
    kb.add_comment(conn, tid, "default", "trying approach A")
    assert kb.block_task(conn, tid, reason="approach A failed", kind="needs_input")
    assert kb.unblock_task(conn, tid)

    assert _claim(conn, tid) is None


def test_retry_with_comment_note_is_allowed_and_recorded(conn):
    tid = _ready_task(conn)
    _fail_with_block(conn, tid)
    kb.add_comment(conn, tid, "mb", "Changed: creds rotated. Why: the old token expired.")

    assert _claim(conn, tid) is not None
    payload = _last_payload(conn, tid, "retry_iteration")
    assert payload["changed"] == ["comments"]
    assert payload["note"] == "Changed: creds rotated. Why: the old token expired."
    assert payload["after_outcome"] == "blocked"


def test_retry_after_gate_block_with_note_is_allowed(conn):
    """The documented way out of a gate block: comment what changed, then unblock."""
    tid = _ready_task(conn)
    _fail_with_block(conn, tid)
    assert _claim(conn, tid) is None
    kb.add_comment(conn, tid, "mb", "split the migration out; this card only adds the column")
    assert kb.unblock_task(conn, tid)

    assert _claim(conn, tid) is not None
    assert _last_payload(conn, tid, "retry_iteration")["changed"] == ["comments"]


@pytest.mark.parametrize("field, sql, params", [
    ("body", "UPDATE tasks SET body = ? WHERE id = ?", ("narrower spec",)),
    ("model", "UPDATE tasks SET model_override = ? WHERE id = ?", ("other-model",)),
])
def test_retry_with_input_change_is_allowed(conn, field, sql, params):
    tid = _ready_task(conn)
    _fail_with_block(conn, tid)
    with kb.write_txn(conn):
        conn.execute(sql, (*params, tid))

    assert _claim(conn, tid) is not None
    assert _last_payload(conn, tid, "retry_iteration")["changed"] == [field]


def test_retry_with_new_parent_is_allowed(conn):
    tid = _ready_task(conn)
    _fail_with_block(conn, tid)
    parent = kb.create_task(conn, title="prerequisite", assignee="default")
    kb.link_tasks(conn, parent, tid)
    kb.complete_task(conn, parent, summary="done")
    kb.recompute_ready(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=? AND status='todo'", (tid,))

    assert _claim(conn, tid) is not None
    assert _last_payload(conn, tid, "retry_iteration")["changed"] == ["parents"]


def test_changes_requested_reason_is_the_note(conn):
    tid = _ready_task(conn)
    assert _claim(conn, tid) is not None
    assert kb.request_review(conn, tid, summary="implemented")
    assert kb.claim_review_task(conn, tid, claimer="reviewer:worker") is not None
    ok, _ = kb.request_changes(conn, tid, reason="add the missing migration test")
    assert ok

    assert _claim(conn, tid) is not None
    payload = _last_payload(conn, tid, "retry_iteration")
    assert "review_reason" in payload["changed"]
    assert payload["note"] == "add the missing migration test"


def test_operator_block_on_ready_card_is_not_an_attempt(conn):
    tid = _ready_task(conn)
    assert kb.block_task(conn, tid, reason="hold until Monday")
    assert kb.unblock_task(conn, tid)

    assert _claim(conn, tid) is not None
