"""A session bound to a credential-pool entry keeps that entry across the per-request refresh.

Replay of t_8d6c7d5a (2026-10-29 21:41:28): a dead ``ANTHROPIC_API_KEY`` sat in the worker's env next
to a healthy OAuth pool row. Every 400 rotated the session to the OAuth entry (``_swap_credential``),
and the refresh that runs before every ``messages.create`` re-resolved through
``resolve_anthropic_token`` — env key first — and swapped the dead key straight back. The turn burned
~20 rotations in 11 s and fell back off Anthropic.

Real ``load_pool`` against a temp ``HERMES_HOME``; only the HTTP client is stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

# Built from parts so no credential-shaped literal sits in the source (secret scanners match ``sk-…``).
_ANT = "sk-" + "ant-"
DEAD_ENV_KEY = _ANT + "api03-dead-env"
LIVE_OAUTH = _ANT + "oat01-live-oauth"
FAR_FUTURE_MS = 4102444800000  # 2100-01-01


def _oauth_row(entry_id: str, token: str, *, source: str = "manual", priority: int = 0) -> dict:
    return {
        "id": entry_id, "label": entry_id, "auth_type": "oauth", "priority": priority, "source": source,
        "access_token": token, "refresh_token": f"rt-{entry_id}", "expires_at_ms": FAR_FUTURE_MS,
    }


def _env_row(secret: str) -> dict:
    return {
        "id": "env-row", "label": "ANTHROPIC_API_KEY", "auth_type": "api_key", "priority": 1,
        "source": "env:ANTHROPIC_API_KEY", "access_token": secret,
    }


def _write_pool(home: Path, rows: list, **extra) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": rows}, **extra}))


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_credentials.read_hermes_oauth_credentials", lambda: None)
    return home


@pytest.fixture
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    a.provider, a.api_mode, a.model = "anthropic", "anthropic_messages", "claude-opus-5-5"
    a.base_url = a._anthropic_base_url = "https://api.anthropic.com"
    a._anthropic_api_key = a.api_key = DEAD_ENV_KEY
    a._anthropic_client = MagicMock()
    a._credential_pool = None
    a._credential_pool_entry_id = None
    return a


def _bind(agent, entry_id: str) -> None:
    """What ``_recover_with_credential_pool`` does after a 400: rotate to *entry_id* and swap it in."""
    from agent.credential_pool import load_pool

    agent._credential_pool = load_pool("anthropic")
    entry = next(e for e in agent._credential_pool.entries() if e.id == entry_id)
    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        assert agent._swap_credential(entry) is True


def _refresh(agent) -> bool:
    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        return agent._try_refresh_anthropic_client_credentials()


def test_refresh_keeps_bound_oauth_entry_over_dead_env_key(home, agent, monkeypatch):
    """21:41:28: env key alive in os.environ, pool [oauth p0, env row p1], session rotated to oauth."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEAD_ENV_KEY)
    _write_pool(home, [_oauth_row("oauth-1", LIVE_OAUTH), _env_row(DEAD_ENV_KEY)])
    _bind(agent, "oauth-1")
    assert agent._anthropic_api_key == LIVE_OAUTH

    _refresh(agent)

    assert agent._anthropic_api_key == LIVE_OAUTH
    assert agent.api_key == LIVE_OAUTH


def test_refresh_keeps_bound_oauth_entry_when_env_row_has_no_secret(home, agent, monkeypatch):
    """23:56: the env row lost its secret, so the failing key matched no pool entry at all."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEAD_ENV_KEY)
    _write_pool(home, [_oauth_row("oauth-1", LIVE_OAUTH), _env_row("")])
    _bind(agent, "oauth-1")

    _refresh(agent)

    assert agent._anthropic_api_key == LIVE_OAUTH


def test_unbound_session_keeps_explicit_env_key_precedence(home, agent, monkeypatch):
    """No bound entry: ``resolve_anthropic_token`` still decides, and an explicit env key wins."""
    explicit = _ANT + "api03-explicit-user-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", explicit)
    _write_pool(home, [_oauth_row("oauth-1", LIVE_OAUTH)])
    agent._anthropic_api_key = agent.api_key = _ANT + "api03-previous-key"

    assert _refresh(agent) is True

    assert agent._anthropic_api_key == explicit


def test_bound_entry_token_refreshed_on_disk_is_adopted(home, agent):
    """Another process refreshed the bound OAuth row: the session follows the entry's new token."""
    _write_pool(home, [_oauth_row("oauth-1", LIVE_OAUTH)])
    _bind(agent, "oauth-1")
    rotated = _ANT + "oat01-rotated-by-peer"
    _write_pool(home, [_oauth_row("oauth-1", rotated)])

    assert _refresh(agent) is True

    assert agent._anthropic_api_key == rotated
    assert agent._is_anthropic_oauth is True


def test_bound_claude_code_entry_follows_the_credentials_file(home, agent, monkeypatch):
    """A borrowed Claude Code login rotates in ~/.claude; ``load_pool`` re-seeds it, so it is not stuck."""
    old, new = _ANT + "oat01-claude-old", _ANT + "oat01-claude-new"
    creds = {"accessToken": old, "refreshToken": "rt-cc", "expiresAt": FAR_FUTURE_MS}
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: dict(creds))
    _write_pool(home, [_oauth_row("cc-1", old, source="claude_code")], active_provider="anthropic")
    _bind(agent, "cc-1")
    assert agent._anthropic_api_key == old

    creds.update(accessToken=new, refreshToken="rt-cc-2")
    assert _refresh(agent) is True

    assert agent._anthropic_api_key == new
