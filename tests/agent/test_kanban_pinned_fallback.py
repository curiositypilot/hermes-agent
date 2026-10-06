"""A pinned Kanban route never falls back to another provider (intake t_a9040028).

The dispatcher marks a pinned route ``HERMES_KANBAN_PINNED=1`` on the worker; the worker's
``_init_fallback_chain`` then attaches no chain, a rate limit ends the run promptly instead of
holding the slot through the reset window, and the exit code requeues the card.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.agent_init as ai
from agent.delegation_context import delegated_child_context
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error

CHAIN = [
    {"provider": "anthropic", "model": "claude-sonnet-5-5"},
    {"provider": "antigravity", "model": "gemini-3.8-flash-tiered"},
]

# ``fallback_providers`` of ~/.hermes/config.yaml.bak-tool-defer-20261006 (15:42 WEST snapshot,
# the only snapshot with anthropic first), pinned here so the replay does not depend on live config.
SNAPSHOT_20261006_CHAIN = [
    {"provider": "anthropic", "model": "claude-sonnet-5-5"},
    {"provider": "antigravity", "model": "claude-opus-4-6-thinking"},
    {"provider": "antigravity", "model": "gemini-3.8-flash-tiered"},
    {"provider": "antigravity-2", "model": "claude-opus-4-6-thinking"},
    {"provider": "antigravity-2", "model": "gemini-3.8-flash-tiered"},
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_PINNED", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_DATA_CLASS", "internal")  # dispatcher-resolved class for the worker
    monkeypatch.setattr("agent.agent_runtime_helpers.sync_credential_pool_entry_id", lambda a: None)


def _chain(provider="antigravity", fallback=CHAIN):
    agent = SimpleNamespace(provider=provider, platform="cli", quiet_mode=True)
    ai._init_fallback_chain(agent, fallback)
    return agent


def test_pinned_worker_has_no_fallback(monkeypatch, caplog):
    assert len(_chain()._fallback_chain) == 2  # no kanban env

    monkeypatch.setenv("HERMES_KANBAN_PINNED", "1")
    assert len(_chain()._fallback_chain) == 2  # pin env without a task: not a worker

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    with caplog.at_level(logging.INFO, logger="run_agent"):
        agent = _chain()
    assert agent._fallback_chain == [] and agent._fallback_model is None
    assert agent._kanban_pinned_route is True
    assert "kanban pinned route: fallback disabled (2 entries dropped)" in caplog.text

    monkeypatch.delenv("HERMES_KANBAN_PINNED")
    assert len(_chain()._fallback_chain) == 2  # tier-routed worker keeps the chain


def test_delegate_child_of_pinned_worker_keeps_its_chain(monkeypatch):
    """In-process delegate children share the worker's os.environ; the gate must not zero
    the chain delegation gave them (inherited or explicitly declared)."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    monkeypatch.setenv("HERMES_KANBAN_PINNED", "1")
    assert _chain()._fallback_chain == []
    with delegated_child_context():
        child = _chain()
    assert len(child._fallback_chain) == 2 and child._kanban_pinned_route is False


def test_replay_run_681_pinned_antigravity(monkeypatch):
    """Run 681 (pinned antigravity:gemini-3.8-flash-tiered) with the 2026-10-06 snapshot chain:
    the old worker walked to anthropic first; a pinned worker now carries nothing."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_18b37fe4")
    old = _chain(fallback=SNAPSHOT_20261006_CHAIN)  # pre-change: no pin marker reached the worker
    assert len(old._fallback_chain) == 5 and old._fallback_chain[0]["provider"] == "anthropic"
    monkeypatch.setenv("HERMES_KANBAN_PINNED", "1")
    assert _chain(fallback=SNAPSHOT_20261006_CHAIN)._fallback_chain == []


# --- rate-limit fast fail ----------------------------------------------------------------

class _RateLimitErr(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("Error code: 429 - rate limit exceeded")
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = SimpleNamespace(headers=headers)
        self.body = {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}}


class _Agent:
    log_prefix = ""
    verbose = False
    provider = "antigravity"
    _fallback_index = 0
    _credential_pool = None
    _auto_recovery_cycles = 0

    def __init__(self, pinned, chain=()):
        self._kanban_pinned_route = pinned
        self._fallback_chain = list(chain)
        self.slept = []

    def _has_pending_fallback(self):
        return self._fallback_index < len(self._fallback_chain)

    def _try_activate_fallback(self, **kwargs):
        return False

    def _try_recover_primary_transport(self, *a, **k):
        return False

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, retry_count=1, error_context=None):
    classified = classify_api_error(err, provider="antigravity")
    assert classified.reason == FailoverReason.rate_limit
    retry = SimpleNamespace(copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=False,
                            has_retried_429=False, restart_with_redirected_messages=False)
    with patch("agent.turn_api_error.interruptible_backoff_sleep", lambda *a, **k: None), \
         patch("agent.turn_api_error.compute_error_backoff", lambda *a, **k: 0.0):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=429,
            error_msg=str(err), error_context=error_context, is_context_length_error=False,
            is_rate_limited=True, _is_zai_coding_overload=False, _provider="antigravity",
            _base="https://example.invalid", _model="gemini", messages=[], api_messages=[],
            api_kwargs={}, active_system_prompt="", conversation_history=None, approx_tokens=10,
            retry_count=retry_count, max_retries=3, compression_attempts=0, api_call_count=1,
        )


def test_pinned_rate_limit_with_long_wait_ends_turn_at_once():
    verdict = _settle(_Agent(pinned=True), _RateLimitErr(retry_after=3600))
    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == "rate_limit"


def test_pinned_rate_limit_reset_at_counts_as_wait():
    verdict = _settle(_Agent(pinned=True), _RateLimitErr(), error_context={"reset_at": time.time() + 7200})
    assert verdict.action == "return"


def test_pinned_rate_limit_short_burst_gets_one_retry_then_ends():
    assert _settle(_Agent(pinned=True), _RateLimitErr(retry_after=5), retry_count=1).action == "fallthrough"
    assert _settle(_Agent(pinned=True), _RateLimitErr(), retry_count=1).action == "fallthrough"
    assert _settle(_Agent(pinned=True), _RateLimitErr(retry_after=5), retry_count=2).action == "return"


def test_unpinned_rate_limit_keeps_normal_retry():
    assert _settle(_Agent(pinned=False), _RateLimitErr(retry_after=3600)).action == "fallthrough"


def test_pinned_rate_limit_with_rotating_pool_keeps_retry(monkeypatch):
    import run_agent
    monkeypatch.setattr(run_agent, "_pool_may_recover_from_rate_limit", lambda pool: True)
    assert _settle(_Agent(pinned=True), _RateLimitErr(retry_after=3600)).action == "fallthrough"


def test_pinned_rate_limit_exits_75(monkeypatch):
    """Guards spec step 4: the failed rate-limit turn maps to the requeue exit code."""
    from hermes_cli.cli_single_query import _single_query_exit_code
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    monkeypatch.setenv("HERMES_KANBAN_PINNED", "1")
    verdict = _settle(_Agent(pinned=True), _RateLimitErr(retry_after=3600))
    assert KANBAN_RATE_LIMIT_EXIT_CODE == 75
    assert _single_query_exit_code(verdict.result) == KANBAN_RATE_LIMIT_EXIT_CODE
    assert _single_query_exit_code({"failed": True, "failure_reason": "rate_limit"}) == 75
