"""A failed turn that ended on a fallback carries why it left the primary (``primary_failure_reason``).

Regression for intake t_88508d36: the primary was rate-limited, a later fallback hop returned a 401,
and the Kanban worker exited 78 (parked) because only the last ``failure_reason`` reached
``turn_exit_code``. The first hop's reason now rides on the result, through the owner-run worker
frame too, so the exit code can requeue instead.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason, classify_api_error
from agent.managed_worker import accept_result, result_frame
from agent.turn_recovery import nonretryable_client_error_result
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
from hermes_cli.turn_exit import turn_exit_code
from run_agent import AIAgent

_CHAIN = [
    {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
    {"provider": "zai", "model": "glm-4.7"},
]


class _Http(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _PrimaryPool:
    """Credential pool on the primary that reports no reset window, so a restore goes through."""
    provider = "custom"

    def next_available_at(self, **_kwargs):
        return None

    def has_credentials(self):
        return True

    def has_available(self, **_kwargs):
        return True

    def select(self, **_kwargs):
        return None


def _make_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://my-llm.example.com/v1", provider="custom",
            quiet_mode=True, skip_context_files=True, skip_memory=True, fallback_model=_CHAIN,
        )
    agent.client = MagicMock()
    return agent


def _hop(agent, reason):
    client = MagicMock()
    client.api_key = "fallback-key"
    client.base_url = "https://openrouter.ai/api/v1"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, None)):
        assert agent._try_activate_fallback(reason) is True


def _auth_failure(agent):
    error = _Http(401, "HTTP 401: missing required authentication credential")
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    return nonretryable_client_error_result(
        agent, error, classified, status_code=401, api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=1, approx_tokens=10, provider=agent.provider,
        base_url=agent.base_url, model=agent.model,
    )


def test_fallback_terminal_error_carries_the_primary_reason_from_the_first_hop():
    agent = _make_agent()
    _hop(agent, FailoverReason.rate_limit)
    _hop(agent, FailoverReason.overloaded)  # a later hop must not overwrite the primary's reason
    result = _auth_failure(agent)
    assert result["failure_reason"] == "auth"
    assert result["primary_failure_reason"] == "rate_limit"
    assert turn_exit_code(result, kanban_worker=True) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_primary_restore_clears_the_primary_reason():
    agent = _make_agent()
    _hop(agent, FailoverReason.rate_limit)
    agent._rate_limited_until = 0
    agent._credential_pool = _PrimaryPool()
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True
    assert agent._primary_failure_reason is None
    result = _auth_failure(agent)
    assert "primary_failure_reason" not in result


def test_no_fallback_no_primary_reason():
    result = _auth_failure(_make_agent())
    assert "primary_failure_reason" not in result


def test_owner_run_worker_frame_keeps_the_primary_reason():
    """``gateway/session_kanban.py`` maps the exit code from the accepted frame, not the raw result."""
    raw = {"final_response": "x", "failed": True, "failure_reason": "auth",
           "primary_failure_reason": "rate_limit", "started": time.time()}
    accepted, _usage = accept_result(result_frame(raw))
    assert accepted["primary_failure_reason"] == "rate_limit"
    assert turn_exit_code(accepted, kanban_worker=True) == KANBAN_RATE_LIMIT_EXIT_CODE
