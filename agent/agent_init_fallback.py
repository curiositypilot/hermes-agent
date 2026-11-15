"""Init-time fallback chain: entry normalization, chain setup, the init walk, and the fallback's api_mode."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("run_agent")


def _fallback_entries(fallback_model) -> list[dict[str, Any]]:
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
    (``HERMES_KANBAN_PINNED=1``; under owner-run the managed worker's
    ``gateway/session_kanban.bind_worker_context`` sets it from ``kanban_json``)."""
    from agent.delegation_context import owned_kanban_task
    return bool(owned_kanban_task()) and os.environ.get("HERMES_KANBAN_PINNED") == "1"


def _policy_filtered_entries(agent, fallback_model) -> list[dict[str, Any]]:
    """The chain after the data-class rules and the chat-only filter (sets ``agent._data_class``)."""
    from agent.provider_policy import (
        ProviderDenied, assert_fallback_allowed, assert_provider_allowed, current_data_class,
    )
    from hermes_cli.fallback_config import drop_chat_only_entries

    data_class = current_data_class()
    agent._data_class = data_class
    chain = []
    for fallback in drop_chat_only_entries(_fallback_entries(fallback_model),
                                           platform=getattr(agent, "platform", None)):
        try:
            assert_provider_allowed(fallback["provider"], data_class, phase="fallback")
            assert_fallback_allowed(fallback["provider"], data_class)
        except ProviderDenied:
            continue
        chain.append(fallback)
    return chain


def allowed_fallback_entries(agent, fallback_model) -> list[dict[str, Any]]:
    """The entries this agent may fall back to: the one filter for both the init-time walk
    (primary has no credential) and the per-turn chain, so neither can reach an entry the
    other refuses (Kanban pin, ``--no-fallback``, data-class rules, chat-only entries)."""
    from hermes_cli.fallback_config import fallback_disabled

    chain = _policy_filtered_entries(agent, fallback_model)
    return [] if _kanban_pinned_route() or fallback_disabled() else chain


def _init_fallback_chain(agent, fallback_model):
    # Stable pool-entry identity: OAuth refreshes can replace the token before a failed
    # request is recovered, so the key value alone can't attribute the failure.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    from agent.provider_policy import assert_provider_allowed, current_data_class

    sync_credential_pool_entry_id(agent)
    assert_provider_allowed(agent.provider, current_data_class(), phase="main")

    # Ordered backups tried when the primary is exhausted (legacy single-dict or list).
    chain = _policy_filtered_entries(agent, fallback_model)
    # A pinned Kanban route is an explicit model choice: walking the global chain would
    # silently answer on another vendor (a critic pinned away from the author's vendor).
    # owned_kanban_task() is "" inside in-process delegate children, so their own chains
    # (inherited or declared) are untouched; same-provider pool rotation is unaffected.
    # Keep in step with allowed_fallback_entries (the init-time walk's filter).
    agent._kanban_pinned_route = _kanban_pinned_route()
    # --no-fallback (HERMES_NO_FALLBACK=1) drops the chain the same way, but is not a Kanban pin:
    # the 429 fast-fail that requeues a card (turn_api_error) keys on _kanban_pinned_route only.
    from hermes_cli.fallback_config import fallback_disabled
    agent._fallback_disabled = agent._kanban_pinned_route or fallback_disabled()
    if agent._kanban_pinned_route:
        logger.info("kanban pinned route: fallback disabled (%d entries dropped)", len(chain))
        chain = []
    elif agent._fallback_disabled:
        logger.info("fallback disabled (--no-fallback): %d entries dropped", len(chain))
        chain = []
    agent._fallback_chain = chain
    agent._fallback_index = 0
    agent._primary_failure_reason = None
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

    A Nous primary configured with ``api_mode: codex_responses`` otherwise leaves a fallback whose
    profile serves that model on chat completions (``api_mode_for_model``) on the Responses API,
    which silently drops reasoning content (#46527).
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


# What an init-time entry overwrites; put back when its native client cannot be installed.
_INIT_IDENTITY = ("provider", "requested_provider", "model", "api_mode", "base_url")


def _install_native_fallback_client(agent, fb_client, before: dict[str, Any]) -> Optional[str]:
    """Install an init-time fallback whose wire mode is anthropic_messages / bedrock_converse.

    ``_build_client`` already took the OpenAI-wire branch from the unreachable primary's mode, so
    ``_init_anthropic_client`` never runs for this entry; install its client the way a per-turn
    fallback does. ``None`` on success. ``AuthError`` (no Anthropic token, raised before
    ``_swap_fallback_clients`` writes ``api_key`` / ``_anthropic_api_key``) puts the identity back
    to ``before`` and returns the refusal so the walk moves on. Any other exception is a bug and
    propagates: swallowing it would answer on the next entry with no record above debug.
    """
    from agent.client_lifecycle import _swap_fallback_clients
    from hermes_cli.auth_constants import AuthError
    try:
        agent.base_url = str(fb_client.base_url)
        _swap_fallback_clients(agent, fb_client, agent.provider, agent.model, agent.base_url, agent.api_mode)
    except AuthError as exc:
        # Same level as the per-turn path (chat_completion_helpers "Failed to activate fallback").
        logger.error("Init-time fallback entry %s client install failed: %s", agent.provider, exc)
        for name, value in before.items():
            setattr(agent, name, value)
        agent._fallback_activated = False
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        return str(exc) or type(exc).__name__
    return None


def walk_init_fallback_chain(agent, fallback_model, timeout, refused: list) -> tuple[bool, Optional[dict[str, Any]]]:
    """Bind the first usable allowed fallback entry when the primary resolved no client.

    ``(True, kwargs)`` when an entry was taken: OpenAI-client kwargs, or ``None`` when the agent is
    already bound (MoA facade, native Anthropic/Bedrock client). ``(False, None)`` when every
    entry refused; each refusal is appended to ``refused`` as ``(provider, reason)``.
    """
    from agent.agent_init import _client_kwargs_from_routed
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.fallback_config import resolve_entry_api_key

    for fb in allowed_fallback_entries(agent, fallback_model):
        fb_provider = str(fb["provider"])
        try:
            fb_client, fb_model = resolve_provider_client(
                fb["provider"], model=fb["model"], raw_codex=True,
                explicit_base_url=fb.get("base_url"), explicit_api_key=resolve_entry_api_key(fb),
            )
        except Exception as fb_exc:  # health: allow BLE001 -- moved verbatim from _routed_client_kwargs: a broken entry is refused, not fatal
            logger.debug("Init-time fallback entry %s failed: %s", fb_provider, fb_exc)
            # A bare exception (``KeyError()``) stringifies empty; name its type instead.
            refused.append((fb_provider, str(fb_exc) or type(fb_exc).__name__))
            continue
        if fb_client is None:
            # The router returns None when no credentials are usable for the entry — a skip
            # that leaves no trace otherwise, hiding key-less fallback entries from the log.
            logger.debug("Init-time fallback entry %s resolved no usable credentials", fb_provider)
            refused.append((fb_provider, "no usable credentials"))
            continue
        agent._fallback_activated = True
        if fb_provider.strip().lower() == "moa":
            # The chokepoint handed back the preset's aggregator client, which only proves the
            # preset resolves and its aggregator has credentials. A MoA entry means the preset
            # itself (same as ``provider: moa`` in config), so bind the facade, not the aggregator.
            from agent.moa_loop import bind_moa_runtime
            bind_moa_runtime(agent, fb["model"])
            return True, None
        before = {name: getattr(agent, name, None) for name in _INIT_IDENTITY}
        agent.provider = agent.requested_provider = fb["provider"]
        agent.model = fb_model or fb["model"]
        recompute_init_fallback_api_mode(agent, fb_client)
        if getattr(agent, "api_mode", None) not in ("anthropic_messages", "bedrock_converse"):
            return True, _client_kwargs_from_routed(fb_client, timeout)
        refusal = _install_native_fallback_client(agent, fb_client, before)
        if refusal is None:
            return True, None
        refused.append((fb_provider, refusal))
    return False, None
