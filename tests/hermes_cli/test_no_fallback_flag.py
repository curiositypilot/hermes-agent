"""``--no-fallback`` keeps a pinned ``--provider/-m`` run on the named model (intake t_2a93e9a2).

Replays the 2026-10-29 model-eval incident: ``hermes chat -Q --provider antigravity -m
gemini-3.8-flash`` walked the global ``fallback_providers`` chain on a 429 and answered on
claude-sonnet-5-5 while ``sessions.model`` still said gemini. The flag sets
``HERMES_NO_FALLBACK=1``; every chain site (agent init, per-turn re-arm, resolution-time walker,
CLI init/sync, oneshot) reads it through ``fallback_config.fallback_disabled``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.agent_init as ai
from hermes_cli.fallback_config import fallback_disabled

# ``fallback_providers`` of the live config on 2026-10-29, copied so the replay does not depend on it.
INCIDENT_CHAIN = [
    {"provider": "anthropic", "model": "claude-sonnet-5-5"},
    {"provider": "antigravity", "model": "claude-opus-5-5-high"},
    {"provider": "xai-oauth", "model": "grok-4.7"},
    {"provider": "antigravity", "model": "gemini-3.8-flash-tiered"},
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_NO_FALLBACK", "HERMES_IGNORE_USER_CONFIG", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_PINNED", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.setenv(var, "")  # registers the var so a direct os.environ write is undone too
        monkeypatch.delenv(var)
    monkeypatch.setenv("HERMES_DATA_CLASS", "internal")
    monkeypatch.setattr("agent.agent_runtime_helpers.sync_credential_pool_entry_id", lambda a: None)


# --- flag -> env --------------------------------------------------------------------------

def test_parser_accepts_flag_on_oneshot_and_chat():
    from hermes_cli._parser import build_top_level_parser

    parser, _, chat = build_top_level_parser()
    assert parser.parse_args(["-z", "x", "--no-fallback"]).no_fallback is True
    assert parser.parse_args(["chat", "--no-fallback", "-Q", "-q", "x"]).no_fallback is True
    assert chat.parse_args(["--no-fallback"]).no_fallback is True
    assert parser.parse_args(["-z", "x"]).no_fallback is False
    assert parser.parse_args(["chat", "-Q", "-q", "x"]).no_fallback is False


def test_main_helper_sets_env_only_when_flag_given():
    from hermes_cli.main import _apply_user_config_bypass

    _apply_user_config_bypass(SimpleNamespace(no_fallback=False))
    _apply_user_config_bypass(SimpleNamespace())  # partial namespace (chat parser alone)
    assert "HERMES_NO_FALLBACK" not in os.environ
    _apply_user_config_bypass(SimpleNamespace(no_fallback=True))
    assert os.environ["HERMES_NO_FALLBACK"] == "1"
    assert "HERMES_IGNORE_USER_CONFIG" not in os.environ  # the sibling bypass is independent


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("true", False), ("1", True),
])
def test_fallback_disabled_reads_exactly_one(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("HERMES_NO_FALLBACK", value)
    assert fallback_disabled() is expected


# --- agent init + per-turn re-arm ---------------------------------------------------------

def _chain(fallback=INCIDENT_CHAIN):
    agent = SimpleNamespace(provider="antigravity", platform="cli", quiet_mode=True)
    ai._init_fallback_chain(agent, fallback)
    return agent


def test_init_drops_chain_without_kanban_pin(monkeypatch, caplog):
    kept = _chain()
    assert len(kept._fallback_chain) == 4 and kept._fallback_disabled is False

    monkeypatch.setenv("HERMES_NO_FALLBACK", "1")
    with caplog.at_level(logging.INFO, logger="run_agent"):
        agent = _chain()
    assert agent._fallback_chain == [] and agent._fallback_model is None
    assert agent._fallback_disabled is True
    assert agent._kanban_pinned_route is False  # the Kanban 429 fast-fail must not fire
    assert "fallback disabled (--no-fallback): 4 entries dropped" in caplog.text


def test_reapply_keeps_disabled_chain_empty():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(_fallback_disabled=True, _fallback_chain=[], _fallback_model=None,
                            _fallback_index=0, _fallback_activated=False)
    GatewayRunner._apply_fallback_chain_to_agent(agent, list(INCIDENT_CHAIN))
    assert agent._fallback_chain == [] and agent._fallback_model is None

    plain = SimpleNamespace(_fallback_chain=[], _fallback_model=None, _fallback_index=0,
                            _fallback_activated=False)
    GatewayRunner._apply_fallback_chain_to_agent(plain, list(INCIDENT_CHAIN))
    assert len(plain._fallback_chain) == 4  # an agent without the attribute behaves as before


def test_cli_turn_sync_keeps_empty_chain(monkeypatch):
    from cli import HermesCLI

    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: anthropic\n    model: claude-sonnet-5-5\n"
        "  - provider: xai-oauth\n    model: grok-4.7\n"
    )
    monkeypatch.setenv("HERMES_NO_FALLBACK", "1")
    agent = _chain()
    cli = SimpleNamespace(_fallback_model=[])
    HermesCLI._sync_fallback_chain_with_config(cli, agent)
    assert cli._fallback_model == [] and agent._fallback_chain == []

    monkeypatch.delenv("HERMES_NO_FALLBACK")
    HermesCLI._sync_fallback_chain_with_config(cli, agent)
    assert [e["provider"] for e in cli._fallback_model] == ["anthropic", "xai-oauth"]


# --- incident replay: a 429 mid-turn ------------------------------------------------------

def _eval_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="k", base_url="https://antigravity.invalid/v1", provider="antigravity",
                        model="gemini-3.8-flash", quiet_mode=True, skip_context_files=True,
                        skip_memory=True, fallback_model=list(INCIDENT_CHAIN))
    agent.client = MagicMock()
    return agent


def _rate_limited_switch(agent):
    from agent.chat_completion_helpers import try_activate_fallback
    from agent.error_classifier import FailoverReason

    client = MagicMock(base_url="https://api.anthropic.com", api_key="fb-key")
    # The anthropic SDK is a lazy extra; the native client build is not what this test is about.
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "claude-sonnet-5-5")), \
         patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        return try_activate_fallback(agent, FailoverReason.rate_limit)


def test_incident_replay_rate_limit(monkeypatch):
    unflagged = _eval_agent()
    assert _rate_limited_switch(unflagged) is True
    assert unflagged.model == "claude-sonnet-5-5"  # the 2026-10-29 behaviour

    monkeypatch.setenv("HERMES_NO_FALLBACK", "1")
    flagged = _eval_agent()
    assert _rate_limited_switch(flagged) is False
    assert flagged.model == "gemini-3.8-flash" and flagged.provider == "antigravity"


# --- resolution-time walker ---------------------------------------------------------------

def test_resolution_time_reraises_primary(monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback

    calls = []

    def fake_resolve(**kw):
        calls.append(kw.get("requested"))
        if kw.get("requested") == "antigravity":
            raise AuthError("antigravity quota exhausted (429)")
        return {"provider": kw["requested"], "api_key": "k"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)
    monkeypatch.setenv("HERMES_NO_FALLBACK", "1")
    with pytest.raises(AuthError, match="antigravity quota exhausted"):
        resolve_runtime_with_fallback({"fallback_providers": INCIDENT_CHAIN}, requested="antigravity")
    assert calls == ["antigravity"]  # no fallback entry was resolved

    monkeypatch.delenv("HERMES_NO_FALLBACK")
    _, entry = resolve_runtime_with_fallback({"fallback_providers": INCIDENT_CHAIN}, requested="antigravity")
    assert entry["model"] == "claude-sonnet-5-5"
