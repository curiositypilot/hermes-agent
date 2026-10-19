"""``cron.failure_route``: failure notices become cards, not chat messages (t_c7535015).

The router command gets the notice as JSON on stdin. Exit 0 = it owns the error (no chat send);
anything else = the notice is delivered to chat exactly as before (never dropped). Success-path
delivery never consults the router.
"""

import json
import sys

import pytest

import cron.scheduler as s
from cron import failure_route as fr
from tests.cron import test_cron_failure_deliver as _fd

# Fixture re-export by assignment: an imported fixture shadowed by the test argument is F811.
run_env = _fd.run_env
_failing_run_job = _fd._failing_run_job
_succeeding_run_job = _fd._succeeding_run_job


def _router(tmp_path, *, rc=0, out='{"action": "filed_intake", "card": "t_abc"}', sleep=0):
    log = tmp_path / "router-calls.jsonl"
    script = tmp_path / "router.py"
    script.write_text(
        "import sys, time, json\n"
        f"time.sleep({sleep})\n"
        f"open({str(log)!r}, 'a').write(sys.stdin.read() + '\\n')\n"
        f"print({out!r})\n"
        f"sys.exit({rc})\n")
    return [sys.executable, str(script)], log


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []


def _configure(monkeypatch, cmd, timeout=30):
    monkeypatch.setattr(s, "load_config", lambda: {"cron": {"failure_route": {"command": cmd, "timeout": timeout}}})


def test_failure_goes_to_router_not_chat(run_env, monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    monkeypatch.setattr(s, "run_job", _failing_run_job("Script exited with code 1 stderr: boom"))

    s.run_one_job({"id": "j1", "name": "scout", "deliver": "slack:D0MAIN"})

    assert run_env["send"] == []
    calls = _calls(log)
    assert len(calls) == 1
    assert calls[0]["source"] == "cron" and calls[0]["job_id"] == "j1" and calls[0]["job_name"] == "scout"
    assert "boom" in calls[0]["text"]
    outcome = run_env["finished"][-1][1].get("delivery_outcome")
    assert outcome == "routed"


def test_router_failure_falls_back_to_chat(run_env, monkeypatch, tmp_path):
    cmd, log = _router(tmp_path, rc=3, out="")
    _configure(monkeypatch, cmd)
    monkeypatch.setattr(s, "run_job", _failing_run_job())

    s.run_one_job({"id": "j2", "name": "scout", "deliver": "slack:D0MAIN"})

    assert len(_calls(log)) == 1
    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]


def test_router_timeout_falls_back_to_chat(run_env, monkeypatch, tmp_path):
    cmd, _log = _router(tmp_path, sleep=10)
    _configure(monkeypatch, cmd, timeout=5)
    monkeypatch.setattr(fr, "DEFAULT_TIMEOUT", 5)
    monkeypatch.setattr(s, "run_job", _failing_run_job())
    import subprocess as sp
    real = sp.run
    monkeypatch.setattr(fr.subprocess, "run", lambda *a, **kw: real(*a, **{**kw, "timeout": 0.5}))

    s.run_one_job({"id": "j3", "name": "scout", "deliver": "slack:D0MAIN"})

    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]


def test_router_crash_falls_back_to_chat(run_env, monkeypatch, tmp_path):
    """An unexpected error inside the router call (a bug, not a known failure) still delivers."""
    cmd, _log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    monkeypatch.setattr(s, "run_job", _failing_run_job())

    def _bug(_cfg):
        raise KeyError("router bug")

    monkeypatch.setattr(fr, "_route_config", _bug)

    s.run_one_job({"id": "j7", "name": "scout", "deliver": "slack:D0MAIN"})

    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]


def test_success_never_consults_router(run_env, monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    monkeypatch.setattr(s, "run_job", _succeeding_run_job())

    s.run_one_job({"id": "j4", "name": "scout", "deliver": "slack:D0MAIN"})

    assert _calls(log) == []
    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]


def test_local_failure_deliver_jobs_are_routed_too(run_env, monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    monkeypatch.setattr(s, "run_job", _failing_run_job())

    s.run_one_job({"id": "j5", "name": "scout", "deliver": "slack:D0MAIN", "failure_deliver": "local"})

    assert run_env["send"] == [] and len(_calls(log)) == 1


def test_unconfigured_is_byte_identical(run_env, monkeypatch):
    monkeypatch.setattr(s, "run_job", _failing_run_job())
    s.run_one_job({"id": "j6", "name": "scout", "deliver": "slack:D0MAIN"})
    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]


@pytest.mark.parametrize("cfg,expected", [
    ({}, []),
    ({"cron": {"failure_route": "python r.py route"}}, ["python", "r.py", "route"]),
    ({"cron": {"failure_route": {"command": ["a", "b"], "enabled": False}}}, []),
    ({"cron": {"failure_route": {"command": ["a", "b"]}}}, ["a", "b"]),
    # Malformed values read as "no route", never raise into the notifier tick or shutdown.
    ({"cron": "oops"}, []),
    ({"cron": {"failure_route": {"command": 42}}}, []),
    ({"cron": {"failure_route": "python 'unclosed"}}, []),
])
def test_route_config_shapes(cfg, expected):
    assert fr._route_config(cfg)[0] == expected


def test_route_configured_unreadable_config_is_unset(monkeypatch):
    from hermes_cli.config import InvalidUserConfigError

    def _bad():
        raise InvalidUserConfigError("config.yaml: bad yaml")

    monkeypatch.setattr(s, "load_config", _bad)
    assert fr.route_configured() is False
    assert fr.route_event({"source": "kanban", "text": "boom"}) is None


def test_route_event_missing_router_binary_returns_none(tmp_path):
    cfg = {"cron": {"failure_route": {"command": [str(tmp_path / "no-such-router")]}}}
    assert fr.route_event({"source": "kanban", "job_id": "t_1", "text": "boom"}, cfg) is None


@pytest.mark.parametrize("cfg,expected", [
    ({}, False),
    ({"cron": {"failure_route": {"command": "", "timeout": 180}}}, False),
    ({"cron": {"failure_route": {"command": ["a", "b"], "enabled": False}}}, False),
    ({"cron": {"failure_route": {"command": ["a", "b"]}}}, True),
    ({"cron": {"failure_route": "python r.py route"}}, True),
])
def test_route_configured(cfg, expected):
    assert fr.route_configured(cfg) is expected


def test_route_configured_reads_scheduler_config_by_default(monkeypatch, tmp_path):
    cmd, _log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    assert fr.route_configured() is True


def test_route_event_passes_source_through(tmp_path):
    cmd, log = _router(tmp_path)
    cfg = {"cron": {"failure_route": {"command": cmd, "timeout": 30}}}
    payload = {"source": "kanban", "job_id": "t_1", "job_name": "card", "text": "t_1 gave up"}

    result = fr.route_event(payload, cfg)

    assert result == {"action": "filed_intake", "card": "t_abc"}
    assert _calls(log) == [payload]


def test_route_event_unconfigured_or_empty_text_returns_none(tmp_path):
    cmd, log = _router(tmp_path)
    cfg = {"cron": {"failure_route": {"command": cmd}}}
    assert fr.route_event({"source": "kanban", "text": "boom"}, {}) is None
    assert fr.route_event({"source": "kanban", "text": "  "}, cfg) is None
    assert _calls(log) == []


def test_route_failure_payload_unchanged(tmp_path):
    cmd, log = _router(tmp_path)
    cfg = {"cron": {"failure_route": {"command": cmd}}}

    assert fr.route_failure({"id": "j7", "name": "scout"}, "boom", cfg)

    (call,) = _calls(log)
    assert set(call) == {"source", "job_id", "job_name", "text", "output_file"}
    assert call["source"] == "cron" and call["job_id"] == "j7" and call["job_name"] == "scout"
    assert call["text"] == "boom"
