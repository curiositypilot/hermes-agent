"""Lifecycle kill sweeps must skip terminal(background=true, persist_on_release=true) jobs (#41225).

Background processes are killed from three agent-lifecycle paths — the release()
``kill_all`` sweep, a gateway turn timeout's ``kill_started_since``, and
``agent_close``'s owned-process loop. An explicitly persisted job must survive
all three, while an operator-driven stop still reaches it.
"""

import pytest
from unittest.mock import MagicMock, patch

from tools.process_registry import ProcessRegistry, ProcessSession, _CHECKPOINT_FIELDS


def _make_session(sid="proc_test123", task_id="t1", persist=False) -> ProcessSession:
    s = ProcessSession(id=sid, command="sleep 60", task_id=task_id, started_at=0.0)
    s.persist_on_release = persist
    return s


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _fake_kill_collector(registry):
    """Replace kill_process so tests observe targeting without signalling."""
    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        registry._running[session_id].exited = True
        return {"status": "killed"}

    registry.kill_process = fake_kill
    return calls


def test_kill_all_release_sweep_skips_persisted_sessions(registry):
    """The default-source kill_all (agent release()) must not kill a
    persist_on_release job owned by the task (#41225)."""
    volatile = _make_session(sid="proc_volatile", task_id="session-a")
    persisted = _make_session(sid="proc_persisted", task_id="session-a", persist=True)
    registry._running[volatile.id] = volatile
    registry._running[persisted.id] = persisted
    calls = _fake_kill_collector(registry)

    assert registry.kill_all("session-a") == 1
    assert [c[0] for c in calls] == ["proc_volatile"]
    assert persisted.exited is False


def test_gateway_turn_timeout_reap_skips_persisted_sessions(registry):
    """A timed-out turn's kill_started_since (source=gateway_turn_timeout) is a
    lifecycle sweep: a persisted job started mid-turn must survive it (#41225)."""
    persisted = _make_session(sid="proc_persisted", task_id="session-a", persist=True)
    volatile = _make_session(sid="proc_new", task_id="session-a")
    registry._running[persisted.id] = persisted
    registry._running[volatile.id] = volatile
    calls = _fake_kill_collector(registry)

    assert registry.kill_started_since("session-a", frozenset(), source="gateway_turn_timeout") == 1
    assert [c[0] for c in calls] == ["proc_new"]
    assert persisted.exited is False


def test_agent_close_owned_loop_skips_persisted_sessions(registry, monkeypatch):
    """_close_task_resources' owned-process loop (source=agent_close) walks
    list_sessions() directly, bypassing kill_all's filter: a persisted session
    must be skipped there too (#41225)."""
    from types import SimpleNamespace

    from agent.client_lifecycle import ClientLifecycleMixin
    import tools.process_registry as registry_module

    persisted = _make_session(sid="proc_persisted", task_id="turn-1", persist=True)
    volatile = _make_session(sid="proc_volatile", task_id="turn-1")
    registry._running[persisted.id] = persisted
    registry._running[volatile.id] = volatile
    calls = _fake_kill_collector(registry)

    agent = SimpleNamespace(_process_owner_task_ids=("turn-1",))
    monkeypatch.setattr(registry_module, "process_registry", registry)
    # _close_task_resources also runs cleanup_vm/cleanup_browser/release_computer_use;
    # stub them so the loop under test is the only thing that runs.
    import run_agent as ra
    monkeypatch.setattr(ra, "cleanup_vm", lambda *a, **k: None)
    monkeypatch.setattr(ra, "cleanup_browser", lambda *a, **k: None)
    import tools.computer_use.tool as cu
    monkeypatch.setattr(cu, "release_computer_use_session", lambda *a, **k: None)

    ClientLifecycleMixin._close_task_resources(agent, "turn-1")

    assert [c[0] for c in calls] == ["proc_volatile"]
    assert all(c[1]["source"] == "agent_close" for c in calls)
    assert persisted.exited is False


def test_explicit_operator_stop_still_reaches_persisted_sessions(registry):
    """persist_on_release is an agent-lifecycle opt-out, never a protection
    against being stopped on purpose: a caller-driven source still kills."""
    persisted = _make_session(sid="proc_persisted", task_id="session-a", persist=True)
    registry._running[persisted.id] = persisted
    calls = _fake_kill_collector(registry)

    assert registry.kill_all("session-a", source="process.kill") == 1
    assert [c[0] for c in calls] == ["proc_persisted"]


def test_list_sessions_flags_persist_on_release(registry):
    """The agent_close loop reads list_sessions(); the flag must be visible
    there for it (and for the agent) to act on."""
    persisted = _make_session(sid="proc_persisted", task_id="session-a", persist=True)
    volatile = _make_session(sid="proc_volatile", task_id="session-a")
    registry._running[persisted.id] = persisted
    registry._running[volatile.id] = volatile

    entries = {e["session_id"]: e for e in registry.list_sessions()}
    assert entries["proc_persisted"].get("persist_on_release") is True
    assert "persist_on_release" not in entries["proc_volatile"]


def test_spawn_local_stamps_persist_on_release(registry, monkeypatch, tmp_path):
    """spawn_local(persist_on_release=True) stamps the flag onto the minted
    ProcessSession so every kill filter can see it (#41225)."""
    import os

    from tools import terminal_tool_sudo

    # Stay off the real hermes home: the spawn-path env sanitizer resolves the
    # real console-script install (_resolve_hermes_bin_dir), which the test
    # suite's HomeIOGuard forbids. None == "no managed install found".
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.environments import local as local_env
    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr(registry, "_track_started", lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool_sudo, "_rewrite_compound_background", lambda c: c)
    monkeypatch.setattr(ProcessRegistry, "_scope_argv", lambda *a, **k: None)
    fake_popen = MagicMock()
    fake_popen.pid = 4242
    monkeypatch.setattr("subprocess.Popen", MagicMock(return_value=fake_popen))

    session = registry.spawn_local(
        "python -c 'import time; time.sleep(60)'", task_id="t1", persist_on_release=True)
    assert session.persist_on_release is True

    session = registry.spawn_local("echo hi", task_id="t1")
    assert session.persist_on_release is False


def test_checkpoint_carries_persist_on_release():
    """A crash-recovery checkpoint must carry the flag, or a persisted job
    resurrects as killable by the next session's release sweep (#41225)."""
    assert "persist_on_release" in _CHECKPOINT_FIELDS


# The dispatcher reaps a finished Kanban run's scope, persisted jobs included
# (intake t_1af94ae5): a persisted spawn inside a run scope must say so.

def _spawn_result(monkeypatch, *, persist, in_scope, env_type="local"):
    import json

    import tools.terminal_tool as tt
    import tools.terminal_tool_background as bg

    class _Session:
        id = "proc_bg"
        pid = 4242

    monkeypatch.setattr(bg, "_spawn", lambda *a, **k: _Session())
    monkeypatch.setattr(bg, "_in_kanban_run_scope", lambda: in_scope)
    monkeypatch.setattr(tt, "_resolve_command_cwd", lambda **k: "/tmp")
    return json.loads(bg.spawn_background_process(
        command="sleep 60", env=None, env_type=env_type, effective_task_id="t1", task_id="t1",
        session_key="", workdir=None, cwd="/tmp", effective_pty=False, notify_on_complete=False,
        watch_patterns=None, approval_note=None, pty_disabled_reason=None,
        persist_on_release=persist))


def test_persist_note_in_kanban_run_scope(monkeypatch):
    result = _spawn_result(monkeypatch, persist=True, in_scope=True)
    assert result["persist_on_release"] is True
    assert "systemd-run" in result["persist_note"]


@pytest.mark.parametrize("persist,in_scope,env_type", [
    (True, False, "local"),   # persisted, but not a Kanban worker
    (False, True, "local"),   # Kanban worker, nothing persisted
    (True, True, "docker"),   # process lives in the container, not the run scope
])
def test_no_persist_note_outside_the_case(monkeypatch, persist, in_scope, env_type):
    result = _spawn_result(monkeypatch, persist=persist, in_scope=in_scope, env_type=env_type)
    assert "error" not in result or result["error"] is None
    assert "persist_note" not in result


@pytest.mark.parametrize("line,expected", [
    ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
     "hermes-worker-kanban-t_8ca578ee-run-6604.scope\n", True),
    ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/probe-x.service\n", False),
    ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
     "hermes-worker-kanban-t_8ca578ee-run-missing.scope\n", False),
    ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service\n", False),
])
def test_in_kanban_run_scope_matches_cgroup(tmp_path, line, expected):
    from tools.terminal_tool_background import _in_kanban_run_scope

    cgroup = tmp_path / "cgroup"
    cgroup.write_text(line)
    assert _in_kanban_run_scope(str(cgroup)) is expected


def test_in_kanban_run_scope_missing_file_is_false(tmp_path):
    from tools.terminal_tool_background import _in_kanban_run_scope

    assert _in_kanban_run_scope(str(tmp_path / "absent")) is False
