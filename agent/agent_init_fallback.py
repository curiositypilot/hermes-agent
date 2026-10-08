"""Init-time fallback chain: entry normalization, chain setup, and the fallback's api_mode."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger("run_agent")


def _fallback_entries(fallback_model) -> List[Dict[str, Any]]:
    """Normalize legacy single-dict ``fallback_model`` / list ``fallback_providers``."""
    if isinstance(fallback_model, dict):
        fallback_model = [fallback_model]
    if not isinstance(fallback_model, list):
        return []
    return [
        f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]


def _kanban_pinned_route() -> bool:
    """True for the dispatcher-owned worker of a card whose route is pinned
    (``HERMES_KANBAN_PINNED=1`` set by ``kanban_db_dispatch._default_spawn``)."""
    from agent.delegation_context import owned_kanban_task
    return bool(owned_kanban_task()) and os.environ.get("HERMES_KANBAN_PINNED") == "1"


def _init_fallback_chain(agent, fallback_model):
    # Stable pool-entry identity: OAuth refreshes can replace the token before a failed
    # request is recovered, so the key value alone can't attribute the failure.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    from agent.provider_policy import (
        ProviderDenied, assert_fallback_allowed, assert_provider_allowed, current_data_class,
    )

    sync_credential_pool_entry_id(agent)
    data_class = current_data_class()
    agent._data_class = data_class
    assert_provider_allowed(agent.provider, data_class, phase="main")

    # Ordered backups tried when the primary is exhausted (legacy single-dict or list).
    from hermes_cli.fallback_config import drop_chat_only_entries

    chain = []
    for fallback in drop_chat_only_entries(_fallback_entries(fallback_model),
                                           platform=getattr(agent, "platform", None)):
        try:
            assert_provider_allowed(fallback["provider"], data_class, phase="fallback")
            assert_fallback_allowed(fallback["provider"], data_class)
        except ProviderDenied:
            continue
        chain.append(fallback)
    # A pinned Kanban route is an explicit model choice: walking the global chain would
    # silently answer on another vendor (a critic pinned away from the author's vendor).
    # owned_kanban_task() is "" inside in-process delegate children, so their own chains
    # (inherited or declared) are untouched; same-provider pool rotation is unaffected.
    agent._kanban_pinned_route = _kanban_pinned_route()
    if agent._kanban_pinned_route:
        logger.info("kanban pinned route: fallback disabled (%d entries dropped)", len(chain))
        chain = []
    agent._fallback_chain = chain
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    chain = agent._fallback_chain
    if chain and not agent.quiet_mode:
        labels = [f"{f['model']} ({f['provider']})" for f in chain]
        if len(chain) == 1:
            print(f"🔄 Fallback model: {labels[0]}")
        else:
            print(f"🔄 Fallback chain ({len(chain)} providers): " + " → ".join(labels))


def recompute_init_fallback_api_mode(agent, fb_client) -> None:
    """Give an init-time fallback its own api_mode instead of the unreachable primary's.

    A Nous primary configured with ``api_mode: codex_responses`` otherwise leaves a Copilot
    gpt-5-mini fallback on the Responses API, which silently drops reasoning content (#46527).
    The per-turn path (``try_activate_fallback``) already recomputes via
    ``_fallback_api_mode_resolved``; this keeps init in agreement, and runs before the
    ``_primary_runtime`` snapshot so ``restore_primary_runtime`` replays the recomputed mode.
    A detection failure keeps the primary's mode and never blocks the fallback itself.
    """
    from agent.chat_completion_helpers import _fallback_api_mode_resolved
    try:
        agent.api_mode = _fallback_api_mode_resolved(
            agent, agent.provider, agent.model, str(getattr(fb_client, "base_url", "") or ""))
    except Exception:
        logger.debug("Init-time fallback api_mode detection failed for %s", agent.provider, exc_info=True)
        return
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
