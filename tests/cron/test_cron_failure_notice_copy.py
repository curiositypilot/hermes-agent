"""User-facing cron failure notices: plain words, the real output path, and the exact `hermes cron`
command to act on. Contract tests, not snapshots (root AGENTS.md).

The classifier is `agent.error_classifier.classify_api_error`; these tests pin what the copy table
does with its verdict, not the verdict itself.
"""

import re

from cron import scheduler
from cron.scheduler import _compose_run_delivery, _summarize_cron_failure_for_delivery

JOB = {"name": "Morning brief", "id": "ab12cd34"}
_HTTP_LEAD = re.compile(r"failed: (HTTP|Error code:|provider )")


def _no_chain(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])






def test_auth_failure_names_the_pinned_provider_and_the_failing_profile(monkeypatch, tmp_path):
    """A profile's credentials are its own (93889b770da): the notice must send the operator to
    THIS profile's sign-in for the job's pinned provider, never a bare placeholder (#114012)."""
    _no_chain(monkeypatch)
    profile_home = tmp_path / ".hermes" / "profiles" / "ops"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    msg = _summarize_cron_failure_for_delivery(
        {**JOB, "provider": "openai-codex"}, "Error code: 401 - Unauthorized")
    assert "`hermes -p ops auth add openai-codex --type oauth`" in msg, msg
    assert "<provider>" not in msg
    unpinned = _summarize_cron_failure_for_delivery(JOB, "Error code: 401 - Unauthorized")
    assert "`hermes -p ops auth add <provider>`" in unpinned, unpinned


def test_rate_and_usage_limit_phrases_still_yield_a_provider_notice(monkeypatch):
    """The old cron regex ladder matched these substrings; the shared classifier must too, or a
    Nous Portal limit turns into a raw generic notice."""
    _no_chain(monkeypatch)
    for text in (
        "Nous Portal rate limit active until 15:00",
        "RuntimeError: usage limit reached for this key",
        "You have hit your weekly usage limit",
        "insufficient quota",
    ):
        msg = _summarize_cron_failure_for_delivery(JOB, text)
        assert "limit" in msg.lower(), msg
        assert not _HTTP_LEAD.search(msg), msg
        assert "`hermes cron run ab12cd34`" in msg or "`hermes cron edit ab12cd34" in msg, msg


def test_cron_cause_gloss_is_the_shared_table():
    """Cron, subagent and chat notices read one reason->cause table (agent/turn_failure_copy.py)."""
    from agent.turn_failure_copy import FAILURE_CAUSE_GLOSS
    from cron.scheduler_failure_copy import provider_failure_notice

    for reason in FAILURE_CAUSE_GLOSS:
        notice = provider_failure_notice("Morning brief", "ab12cd34", reason, backup_provider_phrase="x.")
        assert notice is not None and "`hermes cron" in notice, reason
    assert provider_failure_notice("Morning brief", "ab12cd34", "unknown", backup_provider_phrase="x.") is None


_XAI_SPENDING_LIMIT = (
    'RuntimeError: HTTP 403: {"code":"personal-team-blocked:spending-limit","error":"You have run '
    'out of credits or need a Grok subscription. Purchase credits or subscribe to continue."}'
)


def test_xai_spending_limit_403_is_billing(monkeypatch):
    """xAI's out-of-credits 403 carries a provider-scoped code in its JSON body: with the job's
    provider and the parsed body the notice says credits, never "sign in again" (re-login cannot
    restore credits)."""
    from cron.scheduler_failure_copy import classify_cron_failure_reason

    _no_chain(monkeypatch)
    assert classify_cron_failure_reason(_XAI_SPENDING_LIMIT, provider="xai-oauth") == "billing"
    msg = _summarize_cron_failure_for_delivery(
        {**JOB, "provider": "xai-oauth", "model": "grok-4.7"}, _XAI_SPENDING_LIMIT)
    assert "credit limit" in msg, msg
    assert "/login" not in msg, msg


def test_provider_scoped_code_and_plain_auth_stay_auth():
    """Must-not-match: the xAI code is billing only for xai-oauth; a plain 401 stays auth; a
    malformed JSON tail neither raises nor changes the verdict."""
    from cron.scheduler_failure_copy import classify_cron_failure_reason

    assert classify_cron_failure_reason(_XAI_SPENDING_LIMIT, provider="openrouter") == "auth"
    assert classify_cron_failure_reason("HTTP 401: invalid api key", provider="xai-oauth") == "auth"
    assert classify_cron_failure_reason('HTTP 403: {"code":', provider="xai-oauth") == "auth"


def test_unpinned_job_failed_on_fallback_provider_is_billing(monkeypatch):
    """An unpinned job that fell back onto xai-oauth has no pin: the notice classifies against the
    provider the run failed on (``_failed_provider``, stamped by run_job), so an out-of-credits
    403 says credits, not "sign in again" (kanban-orchestrator, 2026-11-21)."""
    _no_chain(monkeypatch)
    unpinned = {**JOB, "provider": None}
    msg = _summarize_cron_failure_for_delivery(
        {**unpinned, "_failed_provider": "xai-oauth"}, _XAI_SPENDING_LIMIT)
    assert "credit limit" in msg, msg
    assert "/login" not in msg, msg
    # Must-not-match: another provider, no stamp, or a plain 401 on xai-oauth all stay auth.
    for job, text in (({**unpinned, "_failed_provider": "anthropic"}, _XAI_SPENDING_LIMIT),
                      (unpinned, _XAI_SPENDING_LIMIT),
                      ({**unpinned, "_failed_provider": "xai-oauth"}, "HTTP 401: invalid api key")):
        msg = _summarize_cron_failure_for_delivery(job, text)
        assert "/login" in msg and "credit limit" not in msg, (job, text, msg)


def _spy_bookkeeping(monkeypatch):
    """Record the run-side flags on the delivering scheduler's job dict before and after the
    bookkeeping tail (``finish_completed_run``)."""
    from cron import scheduler_bookkeeping

    real, seen = scheduler_bookkeeping.finish_completed_run, {}
    flags = ("_failed_provider", "_quota_hold_route")

    def spy(d, *a, **kw):
        seen["before"] = {k: d.job[k] for k in flags if k in d.job}
        try:
            return real(d, *a, **kw)
        finally:
            seen["after"] = {k: d.job[k] for k in flags if k in d.job}

    monkeypatch.setattr(scheduler_bookkeeping, "finish_completed_run", spy)
    return seen


def test_failed_provider_and_hold_route_cross_the_owner_bridge(tmp_path, monkeypatch, cron_owner):
    """A real scheduler tick through the gateway owner: the agent ran on xai-oauth and failed
    (out of credits, raised as a rate-limited AuthError with a wait hint). The delivered notice
    says credits; ``_failed_provider`` and ``_quota_hold_route`` reach the firing scheduler, and
    the bookkeeping tail consumes both, so neither is saved."""
    from unittest.mock import patch

    from cron.jobs import create_job, get_job
    from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, AuthError

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n", encoding="utf-8")
    job = create_job("kanban orchestrator", "every 15m", deliver="slack:alerts")
    seen, deliveries = _spy_bookkeeping(monkeypatch), []
    window = AuthError("usage window closed; retry after 900s", provider="xai-oauth",
                       code=CODEX_RATE_LIMITED_CODE)

    def fail_on_xai(*_a, **_kw):
        raise RuntimeError(_XAI_SPENDING_LIMIT) from window

    runtime = {"api_key": "k", "base_url": "https://api.x.ai/v1", "provider": "xai-oauth",
               "api_mode": "chat_completions"}
    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), \
         patch("cron.delivery_queue.enqueue",
               side_effect=lambda execution_id, jb, content, **kw: (
                   deliveries.append(content), {"status": "pending"})[1]), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.provider = "xai-oauth"
        agent_cls.return_value.run_conversation.side_effect = fail_on_xai
        scheduler.run_one_job(dict(get_job(job["id"]) or {}))

    assert seen["before"].get("_failed_provider") == "xai-oauth", seen
    assert seen["before"].get("_quota_hold_route"), seen
    assert seen["after"] == {}, seen
    assert len(deliveries) == 1 and "credit limit" in deliveries[0], deliveries
    assert "/login" not in deliveries[0], deliveries
    assert "_failed_provider" not in (get_job(job["id"]) or {})


def test_blocked_config_notice_says_it_did_not_run_and_will_self_heal():
    text, blocked, *_ = _compose_run_delivery(
        JOB, success=False, error="[blocked_config] provider credential missing: no key",
        final_response="", output_file=None)
    assert blocked is True
    assert "provider credential missing: no key" in text
