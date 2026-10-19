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
from tests.cron import test_cron_failure_deliver as _deliver
from tests.cron.test_cron_failure_deliver import _failing_run_job, _succeeding_run_job

# Shared fixture: pytest finds fixtures by module attribute. An assignment, not an import, so the
# test parameters that request it do not shadow an unused import (F811).
run_env = _deliver.run_env


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
])
def test_route_config_shapes(cfg, expected):
    assert fr._route_config(cfg)[0] == expected
