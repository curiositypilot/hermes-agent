"""Run close copies the worker session's token usage into ``task_runs.metadata.usage``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _session(home: Path, sid: str, *, parent: str | None = None, model: str, provider: str,
             inp: int, out: int, cached: int) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(sid, "kanban", model=model, parent_session_id=parent)
        db.update_token_counts(sid, input_tokens=inp, output_tokens=out, cache_read_tokens=cached,
                               model=model, billing_provider=provider, api_call_count=1)
    finally:
        db.close()


def _run_metadata(conn, tid: str) -> dict:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    return json.loads(row["metadata"])


def test_complete_records_worker_session_usage_across_compression(home):
    # Compression rotates the id: the run's usage is the whole kanban chain.
    _session(home, "s_root", model="m-1", provider="prov", inp=100, out=10, cached=1000)
    _session(home, "s_child", parent="s_root", model="m-1", provider="prov", inp=50, out=5, cached=500)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="default")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        assert kb.complete_task(conn, tid, summary="done", metadata={"worker_session_id": "s_child"})
        usage = _run_metadata(conn, tid)["usage"]
    assert usage["model"] == "m-1" and usage["provider"] == "prov"
    assert (usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"]) == (150, 15, 1500)
    assert usage["session_ids"] == ["s_child", "s_root"]


def test_missing_session_closes_run_without_usage(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="default")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        assert kb.complete_task(conn, tid, summary="done", metadata={"worker_session_id": "nope"})
        meta = _run_metadata(conn, tid)
    assert meta["worker_session_id"] == "nope"
    assert "usage" not in meta


def _open_run(conn, tid: str) -> int:
    return int(conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0])


def _raw_run_metadata(conn, run_id: int):
    return conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0]


def test_dispatcher_close_records_usage_from_heartbeat_stamp(home):
    # A crashed/rate_limited/timed_out run is closed by the dispatcher, which never
    # learns the session id: the worker's heartbeat stamp carries it (t_d7bccb06).
    from hermes_cli import kanban_db_dispatch as kbd

    _session(home, "s_root", model="m-1", provider="prov", inp=100, out=10, cached=1000)
    _session(home, "s_child", parent="s_root", model="m-1", provider="prov", inp=50, out=5, cached=500)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="default")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        run = _open_run(conn, tid)
        assert kbd.heartbeat_worker(conn, tid, worker_session_id="s_child", expected_run_id=run)
        with kb.write_txn(conn):
            assert kb._end_run(conn, tid, outcome="crashed", error="x") == run
        meta = _run_metadata(conn, tid)
    assert meta["worker_session_id"] == "s_child"
    usage = meta["usage"]
    assert usage["model"] == "m-1" and usage["provider"] == "prov"
    assert (usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"]) == (150, 15, 1500)
    assert usage["session_ids"] == ["s_child", "s_root"]


def test_heartbeat_stamp_preserves_open_run_metadata(home):
    from hermes_cli import kanban_db_dispatch as kbd

    gate = {"kind": "test_gate", "ok": True}
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="default")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        run = _open_run(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", (json.dumps({"test_gate": gate}), run))
        assert kbd.heartbeat_worker(conn, tid, worker_session_id="s_1", expected_run_id=run)
        # A compression rotation is picked up by the next beat.
        assert kbd.heartbeat_worker(conn, tid, worker_session_id="s_2", expected_run_id=run)
        stored = json.loads(_raw_run_metadata(conn, run))
    assert stored == {"test_gate": gate, "worker_session_id": "s_2"}


def test_heartbeat_stamp_absent_leaves_run_metadata_null(home):
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="default")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        run = _open_run(conn, tid)
        assert kbd.heartbeat_worker(conn, tid, expected_run_id=run)
        assert _raw_run_metadata(conn, run) is None


def test_passed_worker_session_id_wins(home):
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="default")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        run = _open_run(conn, tid)
        assert kbd.heartbeat_worker(conn, tid, worker_session_id="s_old", expected_run_id=run)
        assert kb.complete_task(conn, tid, summary="done", metadata={"worker_session_id": "s_new"})
        meta = _run_metadata(conn, tid)
    assert meta["worker_session_id"] == "s_new"
