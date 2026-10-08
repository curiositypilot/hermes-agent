"""Invariant: a worker that outlives its own terminal transition is still
reachable and gets reaped by the dispatcher (issue #111791).

``complete_task`` wipes ``tasks.worker_pid`` and the ``running``-only reclaim
sweeps never look at a ``done`` card again, so a worker that called
``kanban_complete`` and then hung (holding deleted ``state.db`` sidecar fds)
was invisible to every command. The closed ``task_runs`` row now keeps the pid
and its spawn fingerprint, and ``reap_terminal_workers`` ends such a worker on
the next tick — never a recycled PID, never a legacy row without a fingerprint.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _completed_card_with_worker(conn, proc, *, ended_ago: int = 600) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    # Default: the run closed long enough ago that the reaper's grace window has passed.
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def test_worker_alive_after_completion_is_reaped_on_dispatch_tick(conn):
    proc = _sleeper()
    try:
        tid, run_id = _completed_card_with_worker(conn, proc)
        # The evidence the running-only sweeps lost lives on the closed run row.
        run = conn.execute("SELECT worker_pid, worker_started_at, ended_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["ended_at"] is not None and run["worker_pid"] == proc.pid and run["worker_started_at"] is not None

        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

        assert result.reaped_terminal_workers == [tid]
        assert proc.wait(timeout=10) is not None
        run = conn.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] is None and run["worker_started_at"] is None
        kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
        assert "terminal_worker_reaped" in kinds
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_recycled_pid_and_legacy_row_are_never_signalled(conn):
    stranger = _sleeper()
    legacy = _sleeper()
    try:
        tid, run_id = _completed_card_with_worker(conn, stranger)
        # PID reuse: the recorded fingerprint belongs to a process that no longer exists.
        conn.execute("UPDATE task_runs SET worker_started_at = worker_started_at - 1000000 WHERE id=?", (run_id,))
        _, legacy_run = _completed_card_with_worker(conn, legacy)
        conn.execute("UPDATE task_runs SET worker_started_at = NULL WHERE id=?", (legacy_run,))

        assert kbd.reap_terminal_workers(conn) == []

        assert stranger.poll() is None and legacy.poll() is None
        # Stale evidence is dropped so the row is not rescanned; the legacy row is left alone.
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()["worker_pid"] is None
        assert conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (legacy_run,)).fetchone()["worker_pid"] == legacy.pid
    finally:
        for p in (stranger, legacy):
            p.kill()
            p.wait()


def test_fresh_terminal_run_is_left_alone_until_grace_passes(conn):
    """A worker is still alive for a moment after its own kanban_complete returns
    (final turn, session persistence): a just-closed run is not signalled."""
    proc = _sleeper()
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        os.kill(pid, sig)

    try:
        tid, run_id = _completed_card_with_worker(conn, proc, ended_ago=0)

        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == []

        assert signals == [] and proc.poll() is None
        run = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] == proc.pid  # evidence kept for a later tick
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
            (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id),
        )
        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == [tid]
        assert [pid for pid, _ in signals] == [proc.pid]
    finally:
        proc.kill()
        proc.wait()


def test_one_failing_row_does_not_abort_the_sweep(conn):
    """A signal failure on one run is logged and skipped; the other rows are still reaped."""
    broken, healthy = _sleeper(), _sleeper()
    try:
        _completed_card_with_worker(conn, broken)
        tid, _ = _completed_card_with_worker(conn, healthy)

        def signal_fn(pid, sig):
            if pid == broken.pid:
                raise RuntimeError("boom")
            healthy.kill()

        assert kbd.reap_terminal_workers(conn, signal_fn=signal_fn) == [tid]
    finally:
        for p in (broken, healthy):
            p.kill()
            p.wait()


# --- ended-run scopes (t_b1fa8477) ------------------------------------------------------------
# A clean worker exit still left its background shells inside the run's ``--collect`` scope, which
# kept the scope (and the shells) alive for days. The dispatcher stops the scope of an ended run.


def _scope(tid, run_id):
    return f"hermes-worker-kanban-{tid}-run-{run_id}.scope"


@pytest.fixture
def systemctl(monkeypatch):
    """Stub ``systemctl --user list-units`` / ``stop``; ``units`` is what the user manager lists."""
    from hermes_cli import kanban_db_run_scopes as rs
    import tools.process_registry as pr

    state = SimpleNamespace(units=[], listings=0, stopped=[], binary="/usr/bin/systemctl")

    def fake_run(argv, **_kw):
        state.listings += 1
        out = "".join(f"{u} loaded active running hermes worker\n" for u in state.units)
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(rs, "_last_scan", {})
    monkeypatch.setattr(rs.shutil, "which", lambda name: state.binary if name == "systemctl" else None)
    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    monkeypatch.setattr(pr, "_stop_systemd_unit", lambda unit: state.stopped.append(unit) or True)
    return state


def _ended_run(conn, *, ended_ago: int) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def test_reap_ended_run_scopes_stops_only_scopes_of_runs_ended_past_grace(conn, systemctl):
    from hermes_cli.kanban_db_run_scopes import reap_ended_run_scopes

    old_tid, old_run = _ended_run(conn, ended_ago=600)
    fresh_tid, fresh_run = _ended_run(conn, ended_ago=0)
    live_tid = kb.create_task(conn, title="running", assignee="coder")
    kb.claim_task(conn, live_tid, claimer=kb._claimer_id())
    live_run = kb._current_run_id(conn, live_tid)
    systemctl.units = [
        _scope(old_tid, old_run),
        _scope(fresh_tid, fresh_run),          # ended < grace ago: worker may still be finalising
        _scope(live_tid, live_run),            # ended_at IS NULL: a live worker (the dispatcher's own tick)
        _scope("t_deadbeef", old_run),         # unknown pair: another board's run
        _scope(live_tid, old_run),             # run id of this board, wrong task
        f"hermes-worker-kanban-{old_tid}-run-missing.scope",
    ]

    assert reap_ended_run_scopes(conn) == [_scope(old_tid, old_run)]

    assert systemctl.stopped == [_scope(old_tid, old_run)]
    events = conn.execute(
        "SELECT payload, run_id FROM task_events WHERE task_id=? AND kind='run_scope_reaped'", (old_tid,),
    ).fetchall()
    assert len(events) == 1 and events[0]["run_id"] == old_run
    assert _scope(old_tid, old_run) in events[0]["payload"]


def test_reap_ended_run_scopes_is_throttled_and_needs_systemctl(conn, systemctl):
    from hermes_cli import kanban_db_run_scopes as rs

    tid, run_id = _ended_run(conn, ended_ago=600)
    systemctl.units = [_scope(tid, run_id)]
    assert rs.reap_ended_run_scopes(conn) == [_scope(tid, run_id)]
    assert rs.reap_ended_run_scopes(conn) == []  # within the interval: no second listing
    assert systemctl.listings == 1

    rs._last_scan.clear()
    systemctl.binary = None
    assert rs.reap_ended_run_scopes(conn) == []
    assert systemctl.listings == 1


def test_reap_ended_run_scopes_throttles_each_board_on_its_own(tmp_path, systemctl):
    """The gateway ticks every board in one process: board a's scan must not starve board b."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_run_scopes as rs

    a = kbc.connect(db_path=tmp_path / "a.db")
    b = kbc.connect(db_path=tmp_path / "b.db")
    unit_a, unit_b = _scope(*_ended_run(a, ended_ago=600)), _scope(*_ended_run(b, ended_ago=600))
    a.commit()
    b.commit()
    systemctl.units = [unit_a, unit_b]

    assert rs.reap_ended_run_scopes(a) == [unit_a]
    assert rs.reap_ended_run_scopes(b) == [unit_b]
    assert systemctl.listings == 2
    assert rs.reap_ended_run_scopes(a) == [] and rs.reap_ended_run_scopes(b) == []
    assert systemctl.listings == 2  # same board within the interval: no second listing


def test_dispatch_tick_reports_reap_ended_run_scopes(conn, systemctl):
    tid, run_id = _ended_run(conn, ended_ago=600)
    systemctl.units = [_scope(tid, run_id)]

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, max_spawn=0)

    assert result.reaped_run_scopes == [_scope(tid, run_id)]


def test_dry_run_dispatch_tick_never_stops_run_scopes(conn, systemctl):
    tid, run_id = _ended_run(conn, ended_ago=600)
    systemctl.units = [_scope(tid, run_id)]

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

    assert result.reaped_run_scopes == [] and systemctl.stopped == [] and systemctl.listings == 0
