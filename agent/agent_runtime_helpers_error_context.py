"""Response-header readers for ``extract_api_error_context``: vendor reset windows and the
Anthropic unified (subscription) rate-limit verdict."""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from agent.credential_pool import _parse_absolute_timestamp

# OpenAI-style relative windows: "6m0s", "1.5s", "20ms", "1h2m3s" (also a bare number of seconds).
_DURATION_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
# Lowest-priority reset sources, after Retry-After and x-ratelimit-reset: OpenAI's per-bucket
# durations and Anthropic's per-bucket ISO-8601 timestamps. Plain OpenAI/Anthropic 429s often
# carry only these, and without them the retry status never names the reset window.
_VENDOR_RESET_HEADERS = (
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
    "anthropic-ratelimit-requests-reset", "anthropic-ratelimit-tokens-reset",
)
# Unified-limit claims that cover the whole subscription; ``seven_day_<family>`` (``seven_day_opus``)
# is a per-model-family allowance.
_ACCOUNT_CLAIMS = frozenset({"five_hour", "seven_day"})
_MODEL_CLAIM_PREFIX = "seven_day_"


def _duration_string_seconds(text: str) -> Optional[float]:
    raw = text.strip().lower()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    parts = _DURATION_COMPONENT_RE.findall(raw)
    if not parts or "".join(n + u for n, u in parts) != raw:
        return None
    return sum(float(n) * _DURATION_UNIT_SECONDS[u] for n, u in parts)


def set_reset_from_vendor_headers(context: dict[str, Any], headers: Any) -> None:
    for name in _VENDOR_RESET_HEADERS:
        value = headers.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        seconds = _duration_string_seconds(value)
        if seconds is None:
            absolute = _parse_absolute_timestamp(value)
            seconds = None if absolute is None else absolute - time.time()
        if seconds is not None and seconds > 0:
            context["reset_at"] = time.time() + seconds
            return


def set_unified_ratelimit_context(context: dict[str, Any], headers: Any) -> None:
    """Anthropic's unified (subscription) limit verdict, read only when it rejected the request.

    ``rate_limit_scope`` is ``"account"`` for a whole-subscription claim (``five_hour``,
    ``seven_day``) and ``"model"`` for a model-family claim (``seven_day_opus``); any other claim
    leaves it unset so the pool falls back to the body. ``rate_limit_claim`` keeps the raw claim for
    the pool log. The epoch ``anthropic-ratelimit-unified-reset`` fills ``reset_at`` when nothing
    earlier did; it is not a duration, so it stays out of ``_VENDOR_RESET_HEADERS``.
    """
    if str(headers.get("anthropic-ratelimit-unified-status") or "").strip().lower() != "rejected":
        return
    claim = str(headers.get("anthropic-ratelimit-unified-representative-claim") or "").strip().lower()
    if claim:
        context["rate_limit_claim"] = claim
    if claim in _ACCOUNT_CLAIMS:
        context["rate_limit_scope"] = "account"
    elif claim.startswith(_MODEL_CLAIM_PREFIX) and len(claim) > len(_MODEL_CLAIM_PREFIX):
        context["rate_limit_scope"] = "model"
    if "reset_at" in context:
        return
    try:
        reset = float(str(headers.get("anthropic-ratelimit-unified-reset") or "").strip())
    except ValueError:
        return
    if reset > time.time():
        context["reset_at"] = reset
