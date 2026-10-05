"""Kanban data classes constrain primary and fallback provider selection."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.provider_policy import ProviderDenied, assert_provider_allowed, resolve_data_class


_POLICY_CONFIG = """
kanban:
  data_policies:
    default: internal
    tenant_defaults:
      work: confidential
    classes:
      internal:
        providers: any
        fallback: true
        auxiliary: any
      confidential:
        providers: [anthropic, openai-codex]
        fallback: false
        auxiliary: same_provider_or_fail
"""


@pytest.fixture
def policy_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(_POLICY_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DATA_CLASS", raising=False)
    return home


def test_resolve_data_class_explicit_tenant_default_and_global_default(policy_home):
    assert resolve_data_class({"data_class": "confidential", "tenant": "other"}) == "confidential"
    assert resolve_data_class({"data_class": None, "tenant": "work"}) == "confidential"
    assert resolve_data_class({"data_class": None, "tenant": "other"}) == "internal"


def test_unknown_data_class_fails_closed(policy_home):
    with pytest.raises(ProviderDenied, match="unknown data class"):
        resolve_data_class({"data_class": "secret-ish", "tenant": "work"})


def test_provider_allowlist_and_any_policy(policy_home):
    assert_provider_allowed("anthropic", "confidential")
    assert_provider_allowed("openrouter", "internal")
    with pytest.raises(ProviderDenied, match="antigravity"):
        assert_provider_allowed("antigravity", "confidential")


def test_confidential_task_never_activates_antigravity_fallback(policy_home, monkeypatch):
    from agent.agent_init import _init_fallback_chain
    from agent.chat_completion_helpers import try_activate_fallback

    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")
    agent = SimpleNamespace(
        provider="anthropic", model="claude-sonnet", api_key="test-key",
        _credential_pool=None, quiet_mode=True, _fallback_activated=False,
    )
    _init_fallback_chain(agent, [{"provider": "antigravity", "model": "gemini-3.8-flash"}])

    assert agent._fallback_chain == []
    assert try_activate_fallback(agent) is False
    assert agent.provider == "anthropic"


def test_confidential_policy_disallows_otherwise_allowlisted_fallback(policy_home, monkeypatch):
    from agent.agent_init import _init_fallback_chain

    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")
    agent = SimpleNamespace(
        provider="anthropic", model="claude-sonnet", api_key="test-key",
        _credential_pool=None, quiet_mode=True, _fallback_activated=False,
    )
    _init_fallback_chain(agent, [{"provider": "openai-codex", "model": "gpt-5.4"}])

    assert agent._fallback_chain == []


def test_try_activate_fallback_rechecks_data_class_at_runtime(policy_home, monkeypatch):
    from agent.chat_completion_helpers import try_activate_fallback

    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")
    monkeypatch.setattr("agent.chat_completion_helpers._fallback_chain_exhausted", lambda *_args: False)
    agent = SimpleNamespace(
        provider="anthropic", model="claude-sonnet", _credential_pool=None,
        _fallback_chain=[{"provider": "openai-codex", "model": "gpt-5.4"}],
        _fallback_index=0, _fallback_activated=False, _unavailable_fallback_keys=None,
    )

    assert try_activate_fallback(agent) is False
    assert agent.provider == "anthropic"


def test_primary_provider_denial_raises(policy_home, monkeypatch):
    from agent.agent_init import _init_fallback_chain

    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")
    agent = SimpleNamespace(
        provider="antigravity", model="gemini", api_key="test-key",
        _credential_pool=None, quiet_mode=True, _fallback_activated=False,
    )
    with pytest.raises(ProviderDenied, match="antigravity"):
        _init_fallback_chain(agent, [])


def test_provider_refusal_is_recorded_on_kanban_task(policy_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_KANBAN_DB", str(policy_home / "kanban.db"))
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="confidential card", data_class="confidential")

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    with pytest.raises(ProviderDenied):
        assert_provider_allowed("antigravity", "confidential")

    with kbc.connect_closing() as conn:
        denials = [event for event in kb.list_events(conn, task_id) if event.kind == "provider_denied"]
    assert denials
    assert denials[-1].payload["provider"] == "antigravity"
    assert denials[-1].payload["data_class"] == "confidential"

