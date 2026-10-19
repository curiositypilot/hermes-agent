"""``cron.failure_route``: failure notices become cards, not chat messages (t_c7535015).

The router command gets the notice as JSON on stdin. Exit 0 = it owns the error (no chat send);
anything else = the notice is delivered to chat exactly as before (never dropped). Success-path
delivery never consults the router, except for a successful run's side notices (the verifier
footer and the provider-fallback line, t_2cd8f62d), which go to the router before delivery.
"""

import contextlib
import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

import cron.scheduler as s
from cron import failure_route as fr
from hermes_cli.auth import AuthError
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


# --- side notices of a SUCCESSFUL run (t_2cd8f62d): verifier footer, provider fallback ---------

_REAL_RUN_JOB = s.run_job
_BRIEF = ("Still running; nothing new needs you.\n"
          "Still open from before: three Naval reading decisions, Movis laser wiring, +6 more.")
# Incident replay (state.db message 278671): the refused write the footer reported.
_META = "/tmp/scratch/t_d6134531_meta.json"
_FAILED = {_META: {"tool": "write_file", "error_preview": (
    f"Refusing to overwrite `{_META}`: `{_META}` exists but this task has n…")}}


def _cron_agent(response, failed):
    """A cron agent whose turn ends in the REAL finalizer footer step."""
    from agent.turn_finalizer import _append_file_mutation_footer
    from run_agent import AIAgent

    agent = MagicMock()
    agent._turn_failed_file_mutations = dict(failed)
    agent._file_mutation_verifier_enabled.return_value = True
    agent._file_mutations_still_failed = AIAgent._file_mutations_still_failed
    agent._format_file_mutation_failure_footer = AIAgent._format_file_mutation_failure_footer
    agent.run_conversation.side_effect = lambda *_a, **_kw: {
        "final_response": _append_file_mutation_footer(agent, response, logging.getLogger("t"))}
    return agent


@contextlib.contextmanager
def _job_runtime(tmp_path, agent, *, primary_fails=False):
    """Real ``run_job`` with AIAgent and provider resolution mocked (as in the fallback test)."""
    home = tmp_path / "job-home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n"
        "fallback_providers:\n  - provider: anthropic\n    model: claude-sonnet-5\n    api_key: fb-key\n")

    def _resolve(**kwargs):
        if primary_fails and kwargs.get("requested") != "anthropic":
            raise AuthError("expired")
        return {"api_key": "k", "base_url": "https://example.invalid/v1",
                "provider": kwargs.get("requested") or "openai-codex", "api_mode": "chat_completions"}

    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler._get_hermes_home", return_value=home), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent", return_value=agent):
        yield


def _job(job_id="jv"):
    return {"id": job_id, "name": "kanban-orchestrator", "prompt": "brief", "model": None,
            "provider": None, "provider_snapshot": None, "base_url": None}


def _run(tmp_path, response, failed=None, **kw):
    with _job_runtime(tmp_path, _cron_agent(response, failed or {}), **kw):
        return _REAL_RUN_JOB(_job())


def test_verifier_footer_routed_not_delivered(run_env, monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)

    def _run_job(job, **kw):
        with _job_runtime(tmp_path, _cron_agent(_BRIEF, _FAILED)):
            return _REAL_RUN_JOB(job, **kw)
    monkeypatch.setattr(s, "run_job", _run_job)

    s.run_one_job({**_job("j7"), "deliver": "slack:D0MAIN"})

    delivered = [c["message"] for c in run_env["send"]]
    assert len(delivered) == 1 and "Still running; nothing new needs you." in delivered[0]
    assert "File-mutation verifier" not in delivered[0] and "t_d6134531" not in delivered[0]
    calls = _calls(log)
    assert len(calls) == 1 and "File-mutation verifier" in calls[0]["text"] and _META in calls[0]["text"]
    assert calls[0]["signature"] == "cron-verifier:j7"


def test_incident_278671_replay_delivers_the_brief_exactly(monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    success, _out, final, error = _run(tmp_path, _BRIEF, _FAILED)
    assert (success, error) == (True, None)
    assert final == _BRIEF
    assert len(_calls(log)) == 1


def test_verifier_footer_kept_when_router_fails(monkeypatch, tmp_path):
    from run_agent import AIAgent
    cmd, log = _router(tmp_path, rc=1, out="")
    _configure(monkeypatch, cmd)
    _s, _o, final, _e = _run(tmp_path, _BRIEF, _FAILED)
    assert len(_calls(log)) == 1
    assert final == _BRIEF + "\n\n" + AIAgent._format_file_mutation_failure_footer(_FAILED)


def test_verifier_footer_unconfigured_byte_identical(monkeypatch, tmp_path):
    from run_agent import AIAgent
    monkeypatch.setattr(s, "load_config", lambda: {})
    _s, _o, final, _e = _run(tmp_path, _BRIEF, _FAILED)
    assert final == _BRIEF + "\n\n" + AIAgent._format_file_mutation_failure_footer(_FAILED)


def test_success_without_side_notice_never_consults_router(monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    _s, _o, final, _e = _run(tmp_path, _BRIEF)
    assert final == _BRIEF and _calls(log) == []


def test_failed_run_carries_the_footer_in_its_error(monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    agent = _cron_agent("partial answer", _FAILED)
    inner = agent.run_conversation.side_effect
    agent.run_conversation.side_effect = lambda *a, **kw: {
        **inner(*a, **kw), "failed": True, "error": "provider exploded"}
    with _job_runtime(tmp_path, agent):
        success, _o, final, error = _REAL_RUN_JOB(_job())
    assert (success, final) == (False, "")
    assert error.startswith("RuntimeError: provider exploded\n\n") and "File-mutation verifier" in error
    assert _calls(log) == []  # the failure notice (routed at delivery) carries it


def test_provider_fallback_notice_routed(monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    success, _o, final, _e = _run(tmp_path, "Morning brief: all green.", primary_fails=True)
    assert success and final == "Morning brief: all green."
    calls = _calls(log)
    assert len(calls) == 1 and "anthropic/claude-sonnet-5" in calls[0]["text"]
    assert calls[0]["signature"] == "cron-provider-fallback:jv"

    fail_dir = tmp_path / "failing-router"
    fail_dir.mkdir()
    cmd, _log = _router(fail_dir, rc=1, out="")
    _configure(monkeypatch, cmd)
    _s, _o, final, _e = _run(tmp_path, "Morning brief: all green.", primary_fails=True)
    first, _, rest = final.partition("\n\n")
    assert "anthropic/claude-sonnet-5" in first and rest == "Morning brief: all green."


def test_silent_response_footer(monkeypatch, tmp_path):
    cmd, log = _router(tmp_path)
    _configure(monkeypatch, cmd)
    _s, _o, final, _e = _run(tmp_path, "[SILENT]", _FAILED)
    assert final == "[SILENT]" and s._is_cron_silence_response(final)
    calls = _calls(log)
    assert len(calls) == 1 and calls[0]["signature"] == "cron-verifier:jv"
