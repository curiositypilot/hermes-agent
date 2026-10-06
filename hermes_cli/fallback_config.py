"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


def _normalized_base_url(value: Any) -> str:
    return value.strip().rstrip("/") if isinstance(value, str) else ""


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``api_key_env`` accepted as alias); None when neither
    yields a value so ``resolve_runtime_provider`` falls through to standard credential resolution.
    ``key_env`` goes through ``agent.secret_scope.get_secret``, not raw ``os.getenv``: in a
    multiplexed gateway a bare env read ignores the active profile's scope and can return another
    profile's credential.
    """
    if not isinstance(entry, dict):
        return None
    if inline := str(entry.get("api_key") or "").strip():
        return inline
    if key_env := str(entry.get("key_env") or entry.get("api_key_env") or "").strip():
        from agent.secret_scope import get_secret
        return (get_secret(key_env) or "").strip() or None
    return None


def effective_runtime_provider(
    entry: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> str:
    """Provider identity to persist/display for a resolved fallback entry.

    ``resolve_runtime_provider`` returns the bare billing class ``"custom"``
    for every named ``providers:`` / ``custom_providers:`` entry; the entry's
    configured id only survives in ``requested_provider``. Fallback resolvers
    that persist ``runtime["provider"]`` as the agent identity therefore label
    sessions/billing rows ``custom`` instead of the configured provider name —
    while the manual ``/model`` switch path correctly persists the named id
    (#98739). Same class as the delegation fix in ``tools/delegate_tool.py``.

    Returns the entry's requested identity when the resolved provider is the
    bare ``custom`` class; a genuinely ad-hoc endpoint (requested provider IS
    ``custom``) keeps the bare class unchanged.
    """
    runtime = runtime or {}
    resolved = str(runtime.get("provider") or "").strip()
    if resolved.lower() != "custom":
        return resolved
    requested = str(
        runtime.get("requested_provider")
        or (entry or {}).get("provider")
        or ""
    ).strip()
    if requested and requested.lower() != "custom":
        return requested
    return resolved


def pre_agent_fallback_notice(
    primary_provider: Any, primary_model: Any, fallback_provider: Any, fallback_model: Any
) -> str:
    """User-visible one-shot line for a provider switch made during credential resolution, before
    any AIAgent exists (#74349). Shared by the messaging gateway, the TUI/Desktop gateway and cron
    so the three pre-agent fallback paths cannot drift in wording."""
    primary_desc = "/".join(str(p).strip() for p in (primary_provider, primary_model) if p) or "primary"
    fallback_desc = "/".join(str(p).strip() for p in (fallback_provider, fallback_model) if p) or "fallback"
    return f"⚠️ Provider fallback: {primary_desc} unavailable; using {fallback_desc} for this response."



def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    candidates = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue
        normalized = {**entry, "provider": provider, "model": model}
        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url
        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


_NON_CHAT_PLATFORMS = frozenset({"cron", "subagent"})


def is_background_context(platform: str | None = None) -> bool:
    """True for unattended work: cron runs, delegated children, Kanban workers (and anything they spawn).

    ``HERMES_KANBAN_TASK`` is inherited by a worker's subprocesses on purpose: a child of a worker is
    still worker work. A chat surface (cli/desktop/telegram ...) with none of these markers is chat.
    """
    import os
    from agent.delegation_context import is_delegated_child_process_context

    return (
        (platform or "").strip().lower() in _NON_CHAT_PLATFORMS
        or bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())
        or is_delegated_child_process_context()
    )


def drop_chat_only_entries(chain: list[dict[str, Any]] | None, *, platform: str | None = None,
                           background: bool | None = None) -> list[dict[str, Any]]:
    """Remove ``chat_only: true`` entries outside interactive chat.

    A ``chat_only`` entry (e.g. a pay-per-use last resort) keeps chat alive when every subscription is
    exhausted, while unattended work stops and waits for a subscription to come back instead of
    burning metered credit. ``background`` overrides detection for callers that already know.
    """
    entries = list(chain or [])
    if not (is_background_context(platform) if background is None else background):
        return entries
    return [e for e in entries if not (isinstance(e, dict) and e.get("chat_only") is True)]


def same_provider_only_chain(
    chain: list[dict[str, Any]] | None, primary_provider: Any, primary_base_url: Any = ""
) -> bool:
    """True when every fallback entry sits on the primary's own endpoint, so none can take over
    when that provider fails (e.g. ``openai-codex -> openai-codex`` on one exhausted pool).

    Sameness is :func:`agent.backend_identity.should_skip_candidate` at ``FailureScope.ENDPOINT``
    (same label, or equal explicit base_urls). Empty chain or unknown primary -> False (fail open).
    """
    primary = str(primary_provider or "").strip()
    entries = [entry for entry in chain or () if isinstance(entry, dict)]
    if not entries or not primary:
        return False
    from agent.backend_identity import BackendIdentity, FailureScope, should_skip_candidate

    failed = BackendIdentity.build(primary, None, _normalized_base_url(primary_base_url) or None)
    return all(
        should_skip_candidate(
            BackendIdentity.build(entry.get("provider"), entry.get("model"), entry.get("base_url")),
            failed, FailureScope.ENDPOINT,
        )
        for entry in entries
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its order. Legacy
    ``fallback_model`` entries are appended afterwards unless they target the same
    provider/model/base_url route as an earlier entry. The returned list always contains fresh dict
    copies.
    """
    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain
