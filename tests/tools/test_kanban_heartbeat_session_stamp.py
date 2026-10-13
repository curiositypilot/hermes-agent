"""The worker's auto-heartbeat stamps its session id on its own open run, so a run the
dispatcher closes (crash, quota, timeout, stale) still carries usage (t_d7bccb06)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def board(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _running_task(kb, title: str) -> tuple[str, int]:
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, assignee="default")
        assert kb.claim_task(conn, tid) is not None
        run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        return tid, int(run_id)


def _run_metadata(run_id: int):
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        raw = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0]
    return json.loads(raw) if raw else None


def _beat(monkeypatch, tid: str, run_id: int, session_id: str) -> bool:
    from tools import kanban_tools

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", session_id)
    monkeypatch.setattr(kanban_tools, "_auto_heartbeat_last_attempt", 0.0)
    return kanban_tools.heartbeat_current_worker_from_env()


def test_auto_heartbeat_stamps_own_run_with_session_id(board, monkeypatch):
    tid, run_id = _running_task(board, "worker")
    assert _beat(monkeypatch, tid, run_id, "s_worker") is True
    assert _run_metadata(run_id) == {"worker_session_id": "s_worker"}


def test_heartbeat_tool_stamps_own_run_and_never_another_tasks(board, monkeypatch):
    from tools import kanban_tools

    other_tid, other_run = _running_task(board, "sibling")
    tid, run_id = _running_task(board, "worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", "s_worker")
    # A worker scoped to `tid` naming the sibling is refused; the sibling's run stays unstamped.
    assert json.loads(kanban_tools._handle_heartbeat({"task_id": other_tid})).get("ok") is not True
    assert _run_metadata(other_run) is None
    assert json.loads(kanban_tools._handle_heartbeat({"note": "alive"}))["ok"] is True
    assert _run_metadata(run_id) == {"worker_session_id": "s_worker"}
