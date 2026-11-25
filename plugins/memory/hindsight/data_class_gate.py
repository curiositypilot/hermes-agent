"""Data-class egress gate for content the Hindsight server hands to an LLM (t_4ff4a177).

The data-class policy (``agent/provider_policy.py``) covers model routes; the Hindsight server's own
LLM reads every retained turn and every reflect query, so it is a route the policy must see too.
``server_llm_provider`` (config.json) names that provider; Hindsight Cloud counts as ``hindsight-cloud``.
"""
from __future__ import annotations

import logging
from typing import Callable

from agent.provider_policy import ProviderDenied, provider_permitted

logger = logging.getLogger(__package__)

CLOUD_PROVIDER = "hindsight-cloud"


def policy_egress_block(kind: str, data_class: Callable[[], str], mode: str, server_llm_provider: str | None,
                        *, log: str = "") -> str | None:
    """Why a *kind* (``retain`` | ``reflect`` | ``recall``) call may not reach the server, or None.

    *data_class* resolves the class at call time (the provider passes ``current_data_class``).
    A local server recalls with local embeddings and reranker only, so ``recall`` is checked in
    cloud mode alone. The allowlist decides, not ``is_restricted``: a restricted class whose
    providers include the server LLM keeps its tools. An unset provider passes only ``providers:
    any``; an unresolvable class blocks (fail closed). *log* set: a ``<log>: skipped (...)`` debug line.
    """
    cloud = mode == "cloud"
    if kind == "recall" and not cloud:
        return None
    provider = CLOUD_PROVIDER if cloud else server_llm_provider
    try:
        label = data_class()
    except ProviderDenied as exc:
        reason: str | None = f"data class unresolved, Hindsight server calls refused ({exc})"
    else:
        named = repr(provider) if provider else "(server_llm_provider unset)"
        reason = (None if provider_permitted(provider, label)
                  else f"data class {label!r} does not permit Hindsight server LLM {named}")
    if reason and log:
        logger.debug("%s: skipped (%s)", log, reason)
    return reason
