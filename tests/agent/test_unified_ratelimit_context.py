"""``extract_api_error_context`` reads Anthropic's unified (subscription) rate-limit headers
(t_e0c3a616): the representative claim decides whether a 429 is an account wall or a model limit,
and the epoch reset fills ``reset_at`` when nothing earlier did."""
import time
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import extract_api_error_context


def _err(headers):
    err = Exception("HTTP 429: rate limit")
    err.response = MagicMock(headers=headers)
    return err


def _unified(status="rejected", claim="five_hour", reset=None, **extra):
    headers = {
        "anthropic-ratelimit-unified-status": status,
        "anthropic-ratelimit-unified-representative-claim": claim,
        **extra,
    }
    if reset is not None:
        headers["anthropic-ratelimit-unified-reset"] = str(reset)
    return headers


@pytest.mark.parametrize("claim, scope", [("five_hour", "account"), ("seven_day", "account"),
                                          ("seven_day_sonnet", "model"), ("seven_day_opus", "model")])
def test_rejected_claim_names_the_scope(claim, scope):
    reset = int(time.time()) + 3600
    ctx = extract_api_error_context(_err(_unified(claim=claim, reset=reset)))
    assert ctx["rate_limit_scope"] == scope
    assert ctx["rate_limit_claim"] == claim
    assert ctx["reset_at"] == reset


def test_unknown_claim_gives_no_header_verdict():
    ctx = extract_api_error_context(_err(_unified(claim="overage")))
    assert "rate_limit_scope" not in ctx and ctx["rate_limit_claim"] == "overage"


def test_allowed_status_sets_no_scope():
    ctx = extract_api_error_context(_err(_unified(status="allowed", reset=int(time.time()) + 3600)))
    assert "rate_limit_scope" not in ctx and "rate_limit_claim" not in ctx
    assert "reset_at" not in ctx  # an allowed window's reset is not when this 429 lifts


def test_past_unified_reset_is_ignored():
    ctx = extract_api_error_context(_err(_unified(reset=int(time.time()) - 60)))
    assert ctx["rate_limit_scope"] == "account"
    assert "reset_at" not in ctx


def test_retry_after_still_wins_over_unified_reset():
    ctx = extract_api_error_context(_err(_unified(reset=int(time.time()) + 3600, **{"retry-after": "30"})))
    assert abs(ctx["reset_at"] - (time.time() + 30)) < 5
    assert ctx["rate_limit_scope"] == "account"
