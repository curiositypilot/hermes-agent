"""``chat_only`` fallback entries reach interactive chat and never unattended work."""

import pytest

from hermes_cli.fallback_config import drop_chat_only_entries, is_background_context

SUB = {"provider": "antigravity", "model": "gemini-x"}
PAYG = {"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "chat_only": True}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)


@pytest.mark.parametrize("platform", ["cli", "desktop", "telegram", None])
def test_chat_surfaces_keep_chat_only_entries(platform):
    assert drop_chat_only_entries([SUB, PAYG], platform=platform) == [SUB, PAYG]


@pytest.mark.parametrize("platform", ["cron", "subagent"])
def test_unattended_platforms_drop_chat_only_entries(platform):
    assert drop_chat_only_entries([SUB, PAYG], platform=platform) == [SUB]


def test_kanban_worker_drops_chat_only_even_on_cli_platform(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert is_background_context("cli")
    assert drop_chat_only_entries([SUB, PAYG], platform="cli") == [SUB]


def test_delegated_child_process_drops_chat_only(monkeypatch):
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    assert drop_chat_only_entries([SUB, PAYG], platform="cli") == [SUB]


def test_explicit_background_override_wins():
    assert drop_chat_only_entries([SUB, PAYG], platform="cli", background=True) == [SUB]
    assert drop_chat_only_entries([SUB, PAYG], platform="cron", background=False) == [SUB, PAYG]


def test_only_literal_true_marks_chat_only():
    loose = {"provider": "openrouter", "model": "m", "chat_only": "yes"}
    assert drop_chat_only_entries([loose], platform="cron") == [loose]


def test_agent_init_chain_honours_chat_only(monkeypatch):
    """Real ``_init_fallback_chain``: a Kanban worker agent never carries the chat-only entry."""
    from types import SimpleNamespace
    import agent.agent_init as ai

    monkeypatch.setattr("agent.agent_runtime_helpers.sync_credential_pool_entry_id", lambda a: None)

    def build(platform):
        agent = SimpleNamespace(provider="anthropic", platform=platform, quiet_mode=True)
        ai._init_fallback_chain(agent, [SUB, PAYG])
        return [e["provider"] for e in agent._fallback_chain]

    assert build("desktop") == ["antigravity", "openrouter"]
    monkeypatch.setenv("HERMES_DATA_CLASS", "internal")  # dispatcher-resolved class for the worker
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert build("cli") == ["antigravity"]
