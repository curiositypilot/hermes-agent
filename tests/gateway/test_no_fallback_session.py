"""``hermes chat --no-fallback`` on a gateway-run session (intake t_bec8a9cb).

Replays 2026-11-02 10:55:16Z: ``hermes chat -Q --provider antigravity -m gemini-3.1-pro --no-fallback
--run-budget 120 -t web,file`` ran its turn in the shared gateway, whose env never had the client's
HERMES_NO_FALLBACK; after three 404s on gemini-3.1-pro the agent switched to anthropic/claude-sonnet-5-5
and the run exited 0. The flag now rides ``session.create`` as ``no_fallback`` and is bound per turn for
``fallback_config.fallback_disabled`` (the one reader), and the launch-policy route resolver honours it
outside a turn too (pre-turn resolutions in ``run_inbound``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.fallback_config import fallback_disabled, get_fallback_chain

# session.create flags of the 10:55:16Z run (``-t web,file`` as the thin client splits it).
INCIDENT_CREATE = {"source": "oneshot", "provider": "antigravity", "model": "gemini-3.1-pro",
                   "toolsets": ["web", "file"], "run_budget": 120}
# The live ``fallback_providers`` at that instant, copied so the replay does not depend on it.
INCIDENT_FALLBACKS = [
    {"provider": "anthropic", "model": "claude-sonnet-5-5"},
    {"provider": "antigravity", "model": "claude-opus-latest"},
    {"provider": "xai-oauth", "model": "grok-4.7"},
    {"provider": "antigravity", "model": "gemini-3.8-flash-tiered"},
    {"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "chat_only": True},
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for var in ("HERMES_NO_FALLBACK", "HERMES_KANBAN_TASK", "HERMES_KANBAN_PINNED", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setenv("HERMES_DATA_CLASS", "internal")
    monkeypatch.setattr("agent.agent_runtime_helpers.sync_credential_pool_entry_id", lambda a: None)


def _policy(tmp_path, **extra):
    from gateway.session_policy import build_policy
    config = {"model": {"provider": "antigravity", "default": "gemini-3.1-pro"},
              "fallback_providers": [dict(entry) for entry in INCIDENT_FALLBACKS]}
    return build_policy({**INCIDENT_CREATE, "cwd": str(tmp_path), **extra}, config)


def _session_agent(policy):
    """The agent a turn builds: inside the session's policy scope, handed the session's chain."""
    from gateway.session_policy import policy_scope
    from run_agent import AIAgent
    with policy_scope(policy), \
            patch("model_tools.get_tool_definitions", return_value=[]), \
            patch("model_tools.check_toolset_requirements", return_value={}), \
            patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="k", base_url="https://antigravity.invalid/v1", provider="antigravity",
                        model=policy.model, quiet_mode=True, skip_context_files=True, skip_memory=True,
                        platform="cli", fallback_model=get_fallback_chain(policy.config()))
    agent.client = MagicMock()
    return agent


def test_no_fallback_session_builds_an_agent_without_a_chain(tmp_path):
    from gateway.run import GatewayRunner
    plain = _session_agent(_policy(tmp_path))
    assert plain._fallback_chain and plain._fallback_chain[0]["model"] == "claude-sonnet-5-5"  # the incident
    assert plain._fallback_disabled is False

    agent = _session_agent(_policy(tmp_path, no_fallback=True))
    assert agent._fallback_chain == [] and agent._fallback_disabled is True
    assert agent._kanban_pinned_route is False  # not a Kanban pin: the 429 requeue must not fire
    # The per-turn re-arm of a cached agent keeps it empty.
    GatewayRunner._apply_fallback_chain_to_agent(agent, [dict(entry) for entry in INCIDENT_FALLBACKS])
    assert agent._fallback_chain == [] and agent._fallback_model is None


def _dead_primary(monkeypatch):
    """resolve_runtime_provider: the pinned route's credentials fail, every fallback resolves."""
    from hermes_cli.auth import AuthError
    calls = []

    def resolve(**kw):
        calls.append((kw.get("requested"), kw.get("target_model")))
        if kw.get("requested") == "antigravity" and kw.get("target_model") == "gemini-3.1-pro":
            raise AuthError("antigravity: gemini-3.1-pro unavailable")
        return {"provider": kw.get("requested"), "api_key": "k", "base_url": "https://fallback.invalid/v1"}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr("hermes_cli.runtime_provider_custom._resolve_named_custom_runtime", lambda **kw: None)
    return calls


def test_session_route_resolution_reraises_outside_the_turn_scope(tmp_path, monkeypatch):
    """The pre-turn resolutions (``run_inbound`` vision / context length) run outside ``policy_scope``:
    the launch-policy resolver itself must refuse to walk the chain for a ``no_fallback`` session."""
    from hermes_cli.auth import AuthError
    from gateway.session_worker_construct import route_runtime
    calls = _dead_primary(monkeypatch)
    plain = _policy(tmp_path)
    frozen, model = plain.effective()
    _, entry = route_runtime(plain, frozen, model, None)
    assert entry["model"] == "claude-sonnet-5-5"  # the incident's switch

    flagged = _policy(tmp_path, no_fallback=True)
    frozen, model = flagged.effective()
    calls.clear()
    assert fallback_disabled() is False  # no scope bound: only the policy says so
    with pytest.raises(AuthError, match="gemini-3.1-pro unavailable"):
        route_runtime(flagged, frozen, model, None)
    assert calls == [("antigravity", "gemini-3.1-pro")]


def test_turn_scope_binds_fallback_disabled_without_leaking(tmp_path, monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback
    from gateway.session_policy import policy_scope
    _dead_primary(monkeypatch)
    flagged, plain = _policy(tmp_path, no_fallback=True), _policy(tmp_path)
    config = flagged.config()
    with policy_scope(flagged):
        assert fallback_disabled() is True
        with pytest.raises(AuthError):
            resolve_runtime_with_fallback(config, requested="antigravity", target_model="gemini-3.1-pro")
    assert fallback_disabled() is False
    assert resolve_runtime_with_fallback(config, requested="antigravity", target_model="gemini-3.1-pro")[1]

    # Two sessions' turns at once on the shared gateway: each sees only its own flag.
    barrier = threading.Barrier(2)

    def turn(policy):
        with policy_scope(policy):
            barrier.wait(timeout=10)
            seen = fallback_disabled()
            barrier.wait(timeout=10)
        return seen, fallback_disabled()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(turn, (flagged, plain)))
    assert results == [(True, False), (False, False)]


def test_managed_worker_child_sees_the_session_flag(tmp_path, monkeypatch):
    """A managed worker restores the frozen policy from its bootstrap frame and builds its agent inside
    ``policy_scope`` (``agent.managed_worker.execute``): the flag reaches agent init and ``worker_route``."""
    from hermes_cli.auth import AuthError
    from gateway.session_policy import policy_scope, restore_policy
    from gateway.session_worker_construct import worker_route
    _dead_primary(monkeypatch)
    for flagged in (True, False):
        frame = {"policy": asdict(_policy(tmp_path, no_fallback=flagged))}
        policy = restore_policy(frame["policy"])
        with policy_scope(policy):
            assert fallback_disabled() is flagged
            if flagged:
                with pytest.raises(AuthError):
                    worker_route(policy, {"api_key": None})
            else:  # the fallback entry's route is the agent's own chain's business: no primary overrides
                assert worker_route(policy, {"api_key": None}) == {}
