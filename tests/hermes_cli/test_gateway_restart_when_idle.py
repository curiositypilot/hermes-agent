"""``hermes gateway restart --when-idle`` (t_50ffb4fd): hold new Kanban work, wait, then restart.

The incident: a retry loop of plain restarts waited 86 minutes because the dispatcher kept starting
runs and a 5-minute sample rarely landed on an idle instant. The live-work sequence below replays its
shape (2 units, then 1, then none). Every service-manager seam is a recorder.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_restart_when_idle as when_idle
from hermes_cli import kanban_restart_hold as hold

GATEWAY_PID = 4242
T0_WALL = 1_793_000_000.0  # epoch of the fake clock's 0 (task_runs.started_at is epoch seconds)


def _work(n: int, pid: int | None = GATEWAY_PID) -> dict:
    units = [{"kind": "managed", "pid": GATEWAY_PID}] * n
    answer = {"active_agents": n, "active_work": units}
    return answer if pid is None else answer | {"pid": pid}


def _args(**kw) -> SimpleNamespace:
    base = dict(all=False, system=False, force=False, interrupt=False, when_idle=True, idle_timeout=3600)
    return SimpleNamespace(**(base | kw))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A gateway answering ``active-work`` from ``env.answers`` (the last answer repeats)."""
    from gateway import control_socket

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban"))
    state = SimpleNamespace(answers=[None], queries=[], service=[], sleeps=[], clock=[0.0],
                            live=None, timeline=[], events=[])

    def _query(home, verb, params=None, **kw):
        if verb != "active-work":
            return None
        state.queries.append(hold.active_hold(GATEWAY_PID))
        state.timeline.append((state.clock[0], hold.active(GATEWAY_PID)))
        if state.live is not None:
            return state.live()
        return state.answers[min(len(state.queries) - 1, len(state.answers) - 1)]

    def _sleep(seconds):
        state.sleeps.append(len(state.queries))
        state.clock[0] += seconds
        for event in [e for e in state.events if e[0] <= state.clock[0]]:
            state.events.remove(event)
            event[1]()

    monkeypatch.setattr(control_socket, "query_gateway_control", _query)
    monkeypatch.setattr(when_idle, "_sleep", _sleep)
    monkeypatch.setattr(when_idle, "_now", lambda: state.clock[0])
    monkeypatch.setattr(when_idle, "_wall", lambda: T0_WALL + state.clock[0])
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "_host_multiplexer_for_all_verb", lambda: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda force=False: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda probe: "systemd")
    monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", lambda: None)
    monkeypatch.setattr(gw, "_service_call", lambda *a: state.service.append(
        {"call": a, "queries": len(state.queries), "held": hold.active(GATEWAY_PID), "clock": state.clock[0]}))
    return state


def test_hold_engages_before_polling_and_restart_runs_only_once_idle(env, capsys):
    env.answers = [_work(2), _work(2), _work(1), _work(0)]

    gw._cmd_restart(_args())

    assert env.queries[0] is None, "the pid is resolved before anything is held"
    assert env.queries[1] is not None and env.queries[1]["gateway_pid"] == GATEWAY_PID, \
        "the hold names the gateway pid before the first poll"
    assert env.sleeps == [2, 3], "polled while 2 then 1 unit(s) were live"
    assert len(env.service) == 1
    restart = env.service[0]
    assert restart["call"][1] == "restart"
    assert restart["queries"] >= 4, "the restart ran only after a zero answer (and the guard's re-check)"
    assert restart["held"] is True, "new work stays held through the restart itself"
    assert not hold.hold_path().exists(), "the hold is released afterwards"
    out = capsys.readouterr().out
    assert f"new Kanban work held in gateway pid {GATEWAY_PID}" in out
    assert "managed run" in out


def test_timeout_exits_1_without_restarting_and_releases_the_hold(env, capsys):
    env.answers = [_work(1)]

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(idle_timeout=30))

    assert exc.value.code == 1
    assert env.service == []
    assert env.queries[-1] is not None, "dispatch was held while waiting"
    assert not hold.hold_path().exists()
    out = capsys.readouterr()
    assert "still busy after 30s" in out.out + out.err
    assert "1 live session(s)" in out.out


@pytest.mark.parametrize("extra", [{"interrupt": True}, {"all": True}], ids=["--yes", "--all"])
def test_when_idle_cannot_be_combined_with_yes_or_all(env, extra):
    env.answers = [_work(1)]

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(**extra))

    assert exc.value.code == 2
    assert env.queries == [] and env.service == []
    assert not hold.hold_path().exists()


@pytest.mark.parametrize("answer", [None, _work(0, pid=None)], ids=["no-gateway", "answer-without-pid"])
def test_no_gateway_pid_runs_the_normal_path_without_a_hold(env, answer):
    env.answers = [answer]

    gw._cmd_restart(_args())

    assert len(env.service) == 1 and env.service[0]["held"] is False
    assert all(q is None for q in env.queries)
    assert not hold.hold_path().exists()


def test_refused_inside_a_kanban_worker_whose_own_run_would_never_finish(env, monkeypatch):
    """The worker's own managed run is reported under the gateway pid, so it cannot be excluded from
    the count: inside a worker the wait could only time out with all dispatch frozen."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc12345")
    env.answers = [_work(1)]

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args())

    assert exc.value.code == 2
    assert env.queries == [] and env.service == []
    assert not hold.hold_path().exists()


def test_refused_for_a_profile_the_host_serves(env, tmp_path, monkeypatch):
    """A served profile restarts inside the host, whose single dispatcher serves every profile."""
    from gateway import control_socket
    from hermes_cli import gateway_profile_lifecycle as gpl

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(gw, "_current_profile_name", lambda: "coder")
    monkeypatch.setattr(gpl, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(gw, "_served_by_another_host_gateway", lambda: SimpleNamespace(home=tmp_path / "root"))
    requests: list = []
    monkeypatch.setattr(control_socket, "request_unserve_profile", lambda home, name: requests.append(name))
    env.answers = [_work(1)]

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args())

    assert exc.value.code == 2
    assert requests == [] and env.queries == [] and env.service == []
    assert not hold.hold_path().exists()


def test_plain_restart_refusal_without_a_terminal_points_at_when_idle(env, capsys, monkeypatch):
    env.answers = [_work(1)]
    # The stop path scans host processes; on a host running a real gateway that scan reads the real
    # HERMES_HOME. The fake gateway is the only one.
    monkeypatch.setattr(gw, "find_gateway_pids", lambda *a, **k: [GATEWAY_PID])

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(when_idle=False))
    restart_out = capsys.readouterr().out
    with pytest.raises(SystemExit):
        gw._cmd_stop(SimpleNamespace(all=False, system=False, interrupt=False))
    stop_out = capsys.readouterr().out

    assert exc.value.code == 1 and env.service == []
    assert "hermes gateway restart --when-idle" in restart_out
    assert "--when-idle" not in stop_out


# --- Bounded hold (t_3d310b19) --------------------------------------------------------------------
# The incident: one Kanban run claimed before the hold (t_9ead9bfd, ~2 h) kept every new spawn held for
# the whole 3300 s idle timeout, then the call exited 1 with no restart. ``board`` is a real temp
# ``kanban.db``; ``env.live`` answers ``active-work`` with one managed unit per running row of this
# gateway (unless a test overrides it), and ``env.events`` stand in for the dispatcher and workers.


class _Board:
    def __init__(self, path):
        self.path = path

    def _exec(self, sql: str, params: tuple):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            return conn.execute(sql, params).lastrowid

    def run(self, task_id: str, started: float, *, pid: int = GATEWAY_PID, status: str = "running",
            ended: float | None = None) -> int:
        """Insert a run claimed by gateway ``pid`` at fake-clock time ``started``; returns its id."""
        from hermes_cli.kanban_db import _host_prefix

        return self._exec(
            "INSERT INTO task_runs (task_id, status, claim_lock, started_at, ended_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, status, f"{_host_prefix()}{pid}", int(T0_WALL + started),
             None if ended is None else int(T0_WALL + ended)))

    def end(self, run_id: int, at: float) -> None:
        self._exec("UPDATE task_runs SET status = 'done', ended_at = ? WHERE id = ?", (int(T0_WALL + at), run_id))

    def end_task(self, task_id: str, at: float) -> None:
        self._exec("UPDATE task_runs SET status = 'done', ended_at = ? WHERE task_id = ?", (int(T0_WALL + at), task_id))

    def running(self, task_id: str | None = None) -> int:
        from hermes_cli.kanban_db import _host_prefix

        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("SELECT task_id FROM task_runs WHERE status = 'running' AND claim_lock = ?",
                                (f"{_host_prefix()}{GATEWAY_PID}",)).fetchall()
        return len([r for r in rows if task_id is None or r[0] == task_id])

    def claim_if_dispatch_allowed(self, task_id: str, at: float) -> None:
        """The dispatcher's view: a ready card is claimed only while this gateway is not held."""
        if not hold.active(GATEWAY_PID):
            self.run(task_id, at)


@pytest.fixture
def board(env):
    from hermes_cli.kanban_db_connect import init_db

    b = _Board(init_db(board="default"))
    env.live = lambda: _work(b.running())
    return b


def _held_after_start(env) -> list[tuple[float, bool]]:
    """(clock, held) per ``active-work`` query, minus the pid lookup that precedes the hold."""
    return env.timeline[1:]


def test_hold_released_after_window_while_pre_hold_run_lives(env, board, capsys):
    run = board.run("t_long", -3600)
    env.events.append((900, lambda: board.end(run, 900)))
    # One unit stays live 30 s past the row's end, so the re-engagement below is driven by the run
    # ending, not by the idle answer.
    env.live = lambda: _work(1 if env.clock[0] < 930 else 0)

    gw._cmd_restart(_args())

    timeline = _held_after_start(env)
    unheld = [t for t, held in timeline if not held]
    assert all(held for t, held in timeline if t < 600), "held for the whole window"
    assert unheld and min(unheld) == 600 and max(unheld) <= 900, "released at the window, until the run ended"
    assert min(t for t, held in timeline if held and t > 600) == 900, "re-engaged once ended_at was set"
    assert len(env.service) == 1 and env.service[0]["held"] is True, "restart ran once, held"
    assert not hold.hold_path().exists()
    out = capsys.readouterr().out
    assert f"releasing the hold after 600s; waiting unheld for t_long run {run} (running 70m00s)" in out


def test_runs_claimed_while_unheld_are_not_blockers(env, board):
    run = board.run("t_long", -3600)
    env.events += [(700, lambda: board.claim_if_dispatch_allowed("t_new", 700)),
                   (900, lambda: board.end(run, 900))]
    env.events.append((1000, lambda: board.end_task("t_new", 1000)))

    gw._cmd_restart(_args())

    timeline = _held_after_start(env)
    assert min(t for t, held in timeline if held and t > 600) == 900, \
        "re-engaged when the pre-hold run ended, though t_new (claimed unheld) still ran"
    assert len(env.service) == 1 and env.service[0]["held"] is True
    assert board.running("t_new") == 0 and env.service[0]["clock"] >= 1000, "restart only after t_new ended"


def test_keeps_holding_when_only_chat_or_cron_live(env, board):
    env.live = lambda: {"pid": GATEWAY_PID, "active_agents": 1,
                        "active_work": [{"kind": "chat", "session": "s1"}]}

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(idle_timeout=1200))

    assert exc.value.code == 1 and env.service == []
    assert all(held for _, held in _held_after_start(env)), "no Kanban blocker: held to the timeout"
    assert not hold.hold_path().exists()


@pytest.mark.parametrize("row", [dict(pid=GATEWAY_PID + 1), dict(status="done", ended=-10)],
                         ids=["other-gateway-lock", "ended-run"])
def test_other_gateway_lock_is_not_a_blocker(env, board, row):
    board.run("t_other", -3600, **row)
    env.live = lambda: _work(1)

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(idle_timeout=1200))

    assert exc.value.code == 1 and env.service == []
    assert all(held for _, held in _held_after_start(env))
    assert when_idle.gateway_kanban_runs(GATEWAY_PID) == []


def test_unreadable_db_keeps_holding(env, tmp_path):
    from hermes_cli.kanban_db import kanban_db_path

    path = kanban_db_path(board="default")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a sqlite database" * 64)
    env.live = lambda: _work(1)

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(idle_timeout=1200))

    assert exc.value.code == 1 and env.service == []
    assert all(held for _, held in _held_after_start(env)), "an unreadable board fails toward holding"
    assert when_idle.gateway_kanban_runs(GATEWAY_PID) is None
    assert not hold.hold_path().exists()


def test_hold_window_zero_keeps_old_behaviour(env, board):
    board.run("t_long", -3600)

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(idle_timeout=1200, hold_window=0))

    assert exc.value.code == 1 and env.service == []
    assert all(held for _, held in _held_after_start(env))
    assert not hold.hold_path().exists()


def test_negative_hold_window_exits_2(env, board):
    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(hold_window=-1))

    assert exc.value.code == 2
    assert env.queries == [] and env.service == []
    assert not hold.hold_path().exists()


def test_idle_timeout_caps_unheld_wait(env, board, capsys):
    board.run("t_long", -3600)

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(idle_timeout=1200))

    assert exc.value.code == 1 and env.service == []
    timeline = _held_after_start(env)
    assert timeline[-1] == (1200, False), "the cap fired inside the unheld wait"
    assert not hold.hold_path().exists()
    out = capsys.readouterr()
    assert "still busy after 1200s" in out.out + out.err


def test_progress_lines_name_kanban_runs(env, board, capsys):
    run = board.run("t_long", -3600)
    env.events.append((130, lambda: board.end(run, 130)))

    gw._cmd_restart(_args())

    out = capsys.readouterr().out
    assert f"kanban run t_long run {run}, running 60m00s" in out
    assert "managed run pid" not in out


@pytest.mark.parametrize("hold_window", [600, 0], ids=["bounded-hold", "full-hold"])
def test_unheld_claim_outliving_the_cap_is_the_accepted_miss(env, board, hold_window):
    """t_d44c5617#1: a card ready before the window ends is claimed in the unheld gap and must finish
    too; if it outlives --idle-timeout the call exits 1, where the full hold (--hold-window 0) never
    claims it and restarts when the pre-hold run ends. Accepted trade-off, documented in --help."""
    run = board.run("t_long", -3600)
    env.events += [(610, lambda: board.claim_if_dispatch_allowed("t_gap", 610)),
                   (900, lambda: board.end(run, 900))]

    if hold_window:
        with pytest.raises(SystemExit) as exc:
            gw._cmd_restart(_args(idle_timeout=1800, hold_window=hold_window))
        assert exc.value.code == 1 and env.service == []
        assert board.running("t_gap") == 1, "claimed while unheld and still running at the cap"
    else:
        gw._cmd_restart(_args(idle_timeout=1800, hold_window=hold_window))
        assert len(env.service) == 1 and env.service[0]["held"] is True
        assert board.running("t_gap") == 0, "never claimed under the full hold"
    assert not hold.hold_path().exists()


def test_unheld_wait_restarts_when_live_count_drops_to_zero(env, board):
    """t_d44c5617#2: a crashed worker leaves the gateway's live set at once, while its task_runs row
    stays 'running' until the dispatcher's next reclaim tick; the zero answer still restarts."""
    board.run("t_crashed", -3600)
    env.live = lambda: _work(1 if env.clock[0] < 700 else 0)

    gw._cmd_restart(_args(idle_timeout=1800))

    assert len(env.service) == 1 and env.service[0]["held"] is True, "re-held, then restarted"
    assert board.running("t_crashed") == 1, "the row was never ended"
    assert min(t for t, held in _held_after_start(env) if held and t > 600) == 700
    assert not hold.hold_path().exists()
