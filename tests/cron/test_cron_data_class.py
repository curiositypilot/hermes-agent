"""A cron job's ``data_class`` filters its provider routes exactly as a Kanban card's does.

Cron runs never get the dispatcher's ``HERMES_DATA_CLASS``, so without a per-job class a job that
reads confidential data resolved to the default class and kept the global fallback chain
(xai-oauth, antigravity): a primary outage sent the confidential prompt to another vendor.
"""
from __future__ import annotations

import concurrent.futures
import contextvars
from types import SimpleNamespace

import pytest

from agent.provider_policy import ProviderDenied, current_data_class

_CONFIG = """
fallback_providers:
  - provider: xai-oauth
    model: grok-4.7
  - provider: antigravity
    model: claude-opus-4-6-thinking
  - provider: openai-codex
    model: gpt-5.4
kanban:
  data_policies:
    default: internal
    classes:
      internal:
        providers: any
        fallback: true
        auxiliary: any
      confidential:
        providers: [anthropic, openai-codex]
        fallback: false
        auxiliary: same_provider_or_fail
      vendor-pair:
        providers: [anthropic, openai-codex]
        fallback: true
        auxiliary: any
"""


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_TASK", "HERMES_DATA_CLASS", "HERMES_KANBAN_PINNED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("agent.agent_runtime_helpers.sync_credential_pool_entry_id", lambda a: None)
    return home


def _agent_chain_in_run_scope(job):
    """The fallback chain a cron agent gets: the scheduler's config chain, filtered by
    ``_init_fallback_chain`` on the agent thread the scheduler hands a copied context to."""
    from agent.agent_init import _init_fallback_chain
    from cron.scheduler import _CronRunScope
    from hermes_cli.config import load_config_readonly
    from hermes_cli.fallback_config import drop_chat_only_entries, get_fallback_chain

    def build():
        agent = SimpleNamespace(provider="anthropic", platform="cron", quiet_mode=True)
        chain = drop_chat_only_entries(get_fallback_chain(load_config_readonly()), background=True)
        _init_fallback_chain(agent, chain)
        return agent

    scope = _CronRunScope(job, job["id"], "exec-1")
    try:
        scope.enter()
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            agent = pool.submit(ctx.run, build).result()
    finally:
        scope.exit()
    return agent


def _providers(agent):
    return [entry["provider"] for entry in agent._fallback_chain]


def test_confidential_job_drops_cross_vendor_fallbacks():
    from cron.jobs import create_job

    job = create_job(prompt="brief", schedule="every 1h", data_class="Confidential")
    assert job["data_class"] == "confidential"

    agent = _agent_chain_in_run_scope(job)
    assert agent._data_class == "confidential"
    assert agent._fallback_chain == [] and agent._fallback_model is None
    assert current_data_class() == "internal"  # scope reset after the run


def test_job_without_class_keeps_the_global_chain():
    from cron.jobs import create_job

    job = create_job(prompt="brief", schedule="every 1h")
    assert "data_class" not in job
    agent = _agent_chain_in_run_scope(job)
    assert agent._data_class == "internal"
    assert _providers(agent) == ["xai-oauth", "antigravity", "openai-codex"]


def test_fallback_class_keeps_only_allowlisted_providers():
    job = {"id": "j1", "data_class": "vendor-pair"}
    assert _providers(_agent_chain_in_run_scope(job)) == ["openai-codex"]


def test_confidential_job_refuses_an_off_list_primary(monkeypatch):
    from agent.agent_init import _init_fallback_chain
    from agent.provider_policy import bind_data_class, reset_data_class

    token = bind_data_class("confidential")
    try:
        agent = SimpleNamespace(provider="xai-oauth", platform="cron", quiet_mode=True)
        with pytest.raises(ProviderDenied, match="xai-oauth"):
            _init_fallback_chain(agent, [])
    finally:
        reset_data_class(token)


def test_pre_agent_resolve_fallback_skips_denied_providers(monkeypatch):
    """The scheduler's own resolve-time fallback walk (primary auth failure) honours the class."""
    from hermes_cli.auth import AuthError
    from hermes_cli.config import load_config_readonly
    from cron import scheduler

    tried = []

    def fake_resolve(requested=None, **kw):
        tried.append(requested)
        if requested == "anthropic":
            raise AuthError("expired")
        return {"provider": requested, "api_key": "k", "base_url": "https://x"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)
    # Unpinned jobs (no provider/model/base_url of their own) reach the primary through
    # cron.default_provider; a pinned job never borrows the global chain (#100437), so it is
    # the unpinned walk that the data class must filter.
    jc = scheduler._CronJobConfig(cfg=load_config_readonly(), model="claude-opus-5-5",
                                  model_cfg={}, cron_default_provider="anthropic")
    job = {"id": "j1", "data_class": "confidential"}
    scope = scheduler._CronRunScope(job, "j1", "exec-1")
    try:
        scope.enter()
        with pytest.raises(RuntimeError):
            scheduler._resolve_job_runtime(job, "j1", jc)
    finally:
        scope.exit()
    assert tried == ["anthropic"]

    tried.clear()
    runtime, model = scheduler._resolve_job_runtime({"id": "j2"}, "j2", jc)
    assert runtime["provider"] == "xai-oauth" and model == "grok-4.7"


def test_store_rejects_unknown_class_and_edit_clears():
    from cron.jobs import create_job, update_job

    with pytest.raises(ValueError, match="data_class"):
        create_job(prompt="brief", schedule="every 1h", data_class="top-secret")
    job = create_job(prompt="brief", schedule="every 1h")
    assert update_job(job["id"], {"data_class": "confidential"})["data_class"] == "confidential"
    with pytest.raises(ValueError, match="data_class"):
        update_job(job["id"], {"data_class": "nope"})
    assert update_job(job["id"], {"data_class": ""}).get("data_class") is None


def test_bound_class_never_loosens_a_restricted_worker(monkeypatch):
    from agent.provider_policy import bind_data_class, reset_data_class

    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")
    token = bind_data_class("internal")
    try:
        assert current_data_class() == "confidential"
    finally:
        reset_data_class(token)
    token = bind_data_class("vendor-pair")
    try:
        with pytest.raises(ProviderDenied, match="conflicts"):
            current_data_class()
    finally:
        reset_data_class(token)


def test_cli_create_and_edit_round_trip(capsys):
    from hermes_cli.cron import cron_edit
    from cron.jobs import create_job, get_job

    job = create_job(prompt="brief", schedule="every 1h")
    args = SimpleNamespace(job_id=job["id"], data_class="confidential")
    assert cron_edit(args) == 0
    assert get_job(job["id"])["data_class"] == "confidential"
    assert "Data class: confidential" in capsys.readouterr().out
