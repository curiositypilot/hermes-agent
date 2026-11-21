"""Which credential-pool rows count as capacity: a row the pool never leases is nothing to rotate to.

Every "sole credential?" count (the runtime bench TTL, the picker's limit state, Kanban routing, the
auxiliary diagnostics, the inventory) must apply this one rule. When the runtime counted a keyless
``env:ANTHROPIC_API_KEY`` row and routing did not, a model-less 429 benched the only real login for
60 min while routing called the pool free (t_282f4dff).
"""
from __future__ import annotations

from typing import Any, Iterable


def is_capacity(entry: Any) -> bool:
    """A loaded row that can re-enter rotation: not DEAD, and not an ``api_key`` row without a key.

    ``CredentialPool._available_entries`` never leases a keyless ``api_key`` row; ``env:`` rows are
    hydrated from their variable on ``load_pool()``, so on a loaded pool an empty key means unset.
    """
    from agent.credential_pool import AUTH_TYPE_API_KEY, STATUS_DEAD
    if entry.last_status == STATUS_DEAD:
        return False
    return not (entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key)


def capacity_rows(entries: Iterable[Any]) -> list:
    return [entry for entry in entries if is_capacity(entry)]


def env_row_unset(payload: dict) -> bool:
    """A raw on-disk ``api_key`` row seeded from ``env:<VAR>`` with no key on disk and ``VAR`` unset.

    The raw-payload form of :func:`is_capacity` for readers that skip ``load_pool()`` (env rows are
    secret-free on disk until hydrated). Other token-less rows (borrowed/singleton) hydrate on load.
    """
    source = str(payload.get("source") or "")
    if payload.get("auth_type") != "api_key" or payload.get("access_token") or not source.startswith("env:"):
        return False
    from agent.credential_pool import get_env_prefer_dotenv
    from agent.secret_scope import UnscopedSecretError
    try:
        return not get_env_prefer_dotenv(source[len("env:"):])
    except UnscopedSecretError:  # no profile scope to answer from: keep counting the row
        return False
