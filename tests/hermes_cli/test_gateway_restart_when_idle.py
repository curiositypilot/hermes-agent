"""``hermes gateway restart --when-idle`` (t_50ffb4fd): hold new Kanban work, wait, then restart.

The incident: a retry loop of plain restarts waited 86 minutes because the dispatcher kept starting
runs and a 5-minute sample rarely landed on an idle instant. The live-work sequence below replays its
shape (2 units, then 1, then none). Every service-manager seam is a recorder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_restart_when_idle as when_idle
from hermes_cli import kanban_restart_hold as hold

GATEWAY_PID = 4242


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
    state = SimpleNamespace(answers=[None], queries=[], service=[], sleeps=[], clock=[0.0])

    def _query(home, verb, params=None, **kw):
        if verb != "active-work":
            return None
        state.queries.append(hold.active_hold(GATEWAY_PID))
        return state.answers[min(len(state.queries) - 1, len(state.answers) - 1)]

    def _sleep(seconds):
        state.sleeps.append(len(state.queries))
        state.clock[0] += seconds

    monkeypatch.setattr(control_socket, "query_gateway_control", _query)
    monkeypatch.setattr(when_idle, "_sleep", _sleep)
    monkeypatch.setattr(when_idle, "_now", lambda: state.clock[0])
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "_host_multiplexer_for_all_verb", lambda: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda force=False: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda probe: "systemd")
    monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", lambda: None)
    monkeypatch.setattr(gw, "_service_call", lambda *a: state.service.append(
        {"call": a, "queries": len(state.queries), "held": hold.active(GATEWAY_PID)}))
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


def test_plain_restart_refusal_without_a_terminal_points_at_when_idle(env, capsys):
    env.answers = [_work(1)]

    with pytest.raises(SystemExit) as exc:
        gw._cmd_restart(_args(when_idle=False))
    restart_out = capsys.readouterr().out
    with pytest.raises(SystemExit):
        gw._cmd_stop(SimpleNamespace(all=False, system=False, interrupt=False))
    stop_out = capsys.readouterr().out

    assert exc.value.code == 1 and env.service == []
    assert "hermes gateway restart --when-idle" in restart_out
    assert "--when-idle" not in stop_out
