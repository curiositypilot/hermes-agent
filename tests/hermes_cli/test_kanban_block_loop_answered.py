"""The unblock-loop breaker must count only BLIND re-blocks.

Regression for t_c635f6e9 (2026-10-04): a card blocked twice with
``needs_input``, each time legitimately waiting on MB, and each question was
answered before the next block. The breaker still counted 2 recurrences, routed
the card (by then in review, implementation finished) to ``triage``, and the
auto-decomposer split the finished card into 5 duplicate children.

Contracts:
* a block answered while parked (a comment posted before the next claim, or a
  parent completing) resets the recurrence count;
* a card whose implementation finished never routes to ``triage`` and is never
  offered to the auto-decomposer;
* a real loop (same kind re-blocked, nothing but an unblock in between, worker
  chatter on its own run included) still trips the breaker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _status(conn, tid):
    return conn.execute(
        "SELECT status, block_recurrences FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _kinds(conn, tid):
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]


def _last_payload(conn, tid, kind):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (tid, kind),
    ).fetchone()
    return json.loads(row["payload"]) if row and row["payload"] else {}


def _ready_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="default")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    return tid


def _claim(conn, tid):
    task = kb.claim_task(conn, tid, claimer="default:worker")
    assert task is not None
    return task.current_run_id


def test_t_c635f6e9_sequence_never_reaches_triage(kanban_home: Path) -> None:
    """Replay of t_c635f6e9's lifecycle from kanban.db (run ids 494/509/521).

    01:05 claimed 494 · 01:08 worker comment · 01:08 blocked needs_input
    ("need a lit scene") · 01:36/08:17/09:52 comments while parked (sibling
    handoff, orchestrator, MB decision) · 09:52 unblocked · 09:52 claimed 509 ·
    worker progress comments · 11:31 review_requested · 11:32 claimed 521
    (source_status=review) · 11:35/11:36 reviewer comments · 11:36 blocked
    needs_input ("confirm the Foxglove view") -- live: block_loop_detected.
    """
    with kbc.connect_closing() as conn:
        tid = _ready_task(conn, "S2 harvest acceptance")
        run = _claim(conn, tid)
        kb.add_comment(conn, tid, "default", "Verified movis.service; /detections empty in the dark")
        assert kb.block_task(conn, tid, kind="needs_input", expected_run_id=run,
                             reason="need a lit scene with a person or COCO object")
        kb.add_comment(conn, tid, "default", "From t_7511be56: offline harness decodes CDR")
        kb.add_comment(conn, tid, "default", "orchestrator: surfaced to MB as #3")
        kb.add_comment(conn, tid, "default", "MB decision: OK to record 1 h in daylight")
        assert kb.unblock_task(conn, tid)

        run = _claim(conn, tid)
        kb.add_comment(conn, tid, "default", "Live acceptance gate passed: chair 0.843")
        kb.add_comment(conn, tid, "default", "Final measurements: 3.36 GB/h")
        assert kb.request_review(conn, tid, summary="1 h capture verified", expected_run_id=run)

        review = kb.claim_review_task(conn, tid)
        assert review is not None
        kb.add_comment(conn, tid, "default", "orchestrator: handoff claims check out")
        kb.add_comment(conn, tid, "default", "Independent review round 1: Foxglove check remains")
        assert kb.block_task(conn, tid, kind="needs_input", expected_run_id=review.current_run_id,
                             reason="confirm the Foxglove view")

        row = _status(conn, tid)
        assert row["status"] == "blocked"
        assert "block_loop_detected" not in _kinds(conn, tid)
        payload = _last_payload(conn, tid, "blocked")
        assert payload["loop_reset"] == "comment"
        assert payload["recurrences"] == 1
        assert tid not in kanban_decompose.list_triage_ids(exclude_finished=True)


def test_answered_block_resets_count_outside_review(kanban_home: Path) -> None:
    """A comment while parked answers the block even on a plain implementation card."""
    with kbc.connect_closing() as conn:
        tid = _ready_task(conn)
        kb.block_task(conn, tid, kind="needs_input", expected_run_id=_claim(conn, tid), reason="q1")
        kb.add_comment(conn, tid, "user", "answer to q1")
        kb.unblock_task(conn, tid)
        kb.block_task(conn, tid, kind="needs_input", expected_run_id=_claim(conn, tid), reason="q2")
        row = _status(conn, tid)
        assert (row["status"], row["block_recurrences"]) == ("blocked", 1)


def test_parent_completion_answers_the_block(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _ready_task(conn)
        kb.block_task(conn, tid, kind="needs_input", expected_run_id=_claim(conn, tid), reason="q")
        parent = _ready_task(conn, "parent")
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
        assert kb.complete_task(conn, parent, result="done", force=True)
        kb.unblock_task(conn, tid)
        kb.block_task(conn, tid, kind="needs_input", expected_run_id=_claim(conn, tid), reason="q")
        assert _status(conn, tid)["status"] == "blocked"
        assert _last_payload(conn, tid, "blocked")["loop_reset"] == "parent_completed"


def test_review_card_loop_stays_blocked_not_triage(kanban_home: Path) -> None:
    """A real loop on a finished card (no answer in between) is still recorded,
    but the card stays blocked for a human instead of triage/auto-decompose."""
    with kbc.connect_closing() as conn:
        tid = _ready_task(conn)
        assert kb.request_review(conn, tid, summary="done", expected_run_id=_claim(conn, tid))
        for _ in range(2):
            review = kb.claim_review_task(conn, tid)
            assert review is not None
            kb.block_task(conn, tid, kind="needs_input",
                          expected_run_id=review.current_run_id, reason="sign off please")
            kb.unblock_task(conn, tid)
            assert _status(conn, tid)["status"] == "review"
        review = kb.claim_review_task(conn, tid)
        kb.block_task(conn, tid, kind="needs_input",
                      expected_run_id=review.current_run_id, reason="sign off please")
        row = _status(conn, tid)
        assert row["status"] == "blocked"
        assert row["block_recurrences"] >= kb.BLOCK_RECURRENCE_LIMIT
        assert "block_loop_detected" not in _kinds(conn, tid)
        assert _last_payload(conn, tid, "blocked")["loop_suppressed"] == "implementation_complete"


def test_auto_decompose_skips_finished_triage_cards(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        finished = _ready_task(conn, "finished")
        assert kb.request_review(conn, finished, summary="done", expected_run_id=_claim(conn, finished))
        fresh = kb.create_task(conn, title="fresh", assignee="default")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id IN (?, ?)", (finished, fresh))
    ids = kanban_decompose.list_triage_ids(exclude_finished=True)
    assert fresh in ids and finished not in ids
    assert {fresh, finished} <= set(kanban_decompose.list_triage_ids())


def test_real_loop_still_trips_despite_worker_chatter(kanban_home: Path) -> None:
    """Same kind, only an unblock in between; worker comments on its own runs
    (before the block / after the claim) are not answers."""
    with kbc.connect_closing() as conn:
        tid = _ready_task(conn)
        run = _claim(conn, tid)
        kb.add_comment(conn, tid, "default", "still stuck")
        kb.block_task(conn, tid, kind="capability", expected_run_id=run, reason="no creds")
        kb.unblock_task(conn, tid)
        run = _claim(conn, tid)
        kb.add_comment(conn, tid, "default", "still no creds")
        kb.block_task(conn, tid, kind="capability", expected_run_id=run, reason="no creds")
        assert _status(conn, tid)["status"] == "triage"
        payload = _last_payload(conn, tid, "block_loop_detected")
        assert payload["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT
        assert "loop_reset" not in payload
