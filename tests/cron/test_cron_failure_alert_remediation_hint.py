"""The empty-chain failure alert must tell the operator how to fix it.

Field report: a user whose cron died with "No fallback
chain configured." still cannot self-serve — the alert names the problem but
not the remedy. The empty-chain branch of _fallback_chain_phrase() must name
the exact commands: `hermes fallback add` for the chain, and the
cron.model / cron.model_provider fleet-default keys as the operator-level
alternative. The exhausted branch stays terse — the chain is intact and the
problem is provider-side, so no config command applies.
"""

import cron.scheduler as scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery


def test_empty_chain_alert_names_the_remediation_commands(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "semi-analyst-radar", "id": "aaa111"}
    msg = _summarize_cron_failure_for_delivery(job, "Request timed out.")
    assert "No backup provider is configured" in msg
    assert "hermes fallback add" in msg
    assert "cron.model_provider" in msg


def test_exhausted_chain_alert_does_not_carry_the_config_hint(monkeypatch):
    monkeypatch.setattr(
        scheduler, "load_config",
        lambda: {"fallback_providers": [{"provider": "openrouter", "model": "x"}]},
    )
    monkeypatch.setattr(
        scheduler, "get_fallback_chain",
        lambda cfg: [{"provider": "openrouter", "model": "x"}],
    )
    job = {"name": "semi-analyst-radar", "id": "aaa111"}
    msg = _summarize_cron_failure_for_delivery(job, "Request timed out.")
    assert "No backup provider succeeded either." in msg
    assert "hermes fallback add" not in msg


def test_rate_limit_empty_chain_also_carries_the_hint(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "kz-coverage", "id": "bbb222"}
    msg = _summarize_cron_failure_for_delivery(job, "HTTP 429: rate limit exceeded")
    assert "No backup provider is configured" in msg
    assert "hermes fallback add" in msg


CODEX_ONLY = [{"provider": "openai-codex", "model": "a"}, {"provider": "openai-codex", "model": "b"}]


def _patch_chain(monkeypatch, cfg, chain):
    monkeypatch.setattr(scheduler, "load_config", lambda: cfg)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda _cfg: chain)


def test_same_provider_only_chain_names_the_dead_provider_and_the_fix(monkeypatch):
    _patch_chain(monkeypatch, {}, CODEX_ONLY)
    job = {"name": "radar", "id": "ccc333", "provider": "openai-codex"}
    msg = _summarize_cron_failure_for_delivery(job, "HTTP 429: rate limit exceeded")
    assert "openai-codex" in msg
    assert "hermes fallback add" in msg
    assert "hermes cron edit ccc333 --provider" in msg
    assert "succeeded either" not in msg


def test_mixed_chain_keeps_the_backups_failed_text(monkeypatch):
    _patch_chain(monkeypatch, {}, [CODEX_ONLY[0], {"provider": "anthropic", "model": "c"}])
    job = {"name": "radar", "id": "ccc333", "provider": "openai-codex"}
    msg = _summarize_cron_failure_for_delivery(job, "HTTP 429: rate limit exceeded")
    assert "No backup provider succeeded either." in msg


def test_unpinned_job_uses_global_model_provider(monkeypatch):
    _patch_chain(monkeypatch, {"model": {"provider": "openai-codex"}}, CODEX_ONLY)
    msg = _summarize_cron_failure_for_delivery({"name": "radar", "id": "ddd444"},
                                               "HTTP 429: rate limit exceeded")
    assert "Every backup provider is on `openai-codex`" in msg
    assert "succeeded either" not in msg


def test_cron_model_provider_outranks_global_model_provider(monkeypatch):
    cfg = {"cron": {"model_provider": "anthropic"}, "model": {"provider": "openai-codex"}}
    _patch_chain(monkeypatch, cfg, CODEX_ONLY)
    msg = _summarize_cron_failure_for_delivery({"name": "radar", "id": "ddd444"},
                                               "HTTP 429: rate limit exceeded")
    assert "No backup provider succeeded either." in msg


def test_unknown_primary_keeps_the_backups_failed_text(monkeypatch):
    _patch_chain(monkeypatch, {}, CODEX_ONLY)
    msg = _summarize_cron_failure_for_delivery({"name": "radar", "id": "eee555"},
                                               "HTTP 429: rate limit exceeded")
    assert "No backup provider succeeded either." in msg
    assert scheduler._fallback_chain_phrase() == "No backup provider succeeded either."
