"""Adopting a token pair another process rotated into the auth.json singleton.

A rotated pair is usually a routine refresh, not a re-login: it must replace a consumed
refresh token, but it restores no credits, so it must not end a billing bench (t_7f8066a9).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from agent.credential_pool_timestamps import _parse_absolute_timestamp

logger = logging.getLogger("agent.credential_pool")

# Providers whose interactive logins stamp ``providers.<id>.logged_in_at`` (``login=True`` in
# ``_save_xai_oauth_tokens``). openai-codex has no stamp and keeps clear-on-rotation.
_LOGIN_STAMPED_PROVIDERS = frozenset({"xai-oauth"})


def _singleton_predates_entry(state: Any, entry: Any) -> bool:
    """True only when the auth.json singleton is PROVABLY older than *entry*.

    Both sides stamp ``last_refresh`` on every successful rotation. When
    either side lacks a parseable stamp this returns False (cannot prove),
    which keeps the historical adopt-on-difference behavior (#70111) intact
    for legacy writers.
    """
    entry_ts = _parse_absolute_timestamp(entry.last_refresh)
    if entry_ts is None:
        return False
    state_ts = _parse_absolute_timestamp(state.get("last_refresh") if isinstance(state, dict) else None)
    if state_ts is None:
        return False
    return state_ts < entry_ts


def is_billing_bench(entry: Any) -> bool:
    """*entry* is exhausted by a billing mark (402, or a 403 classified as billing)."""
    from agent.credential_pool import STATUS_EXHAUSTED, _is_billing_mark

    return entry.last_status == STATUS_EXHAUSTED and _is_billing_mark(
        entry.last_error_code, entry.extra.get("failure_reason"))


def rotation_is_refresh_only(provider: str, entry: Any, state: Any) -> bool:
    """The singleton's newer pair came from a token refresh, not a login after *entry*'s bench.

    True only for login-stamped providers when ``logged_in_at`` is absent, unparseable or not
    later than the bench start (``last_status_at``, else ``billing_streak_at``). Both sides go
    through ``_parse_absolute_timestamp``: the stamp is ISO-Z, the row fields epoch floats.
    Failing safe to "refresh": a bench still ends by TTL or ``hermes auth reset xai-oauth``.

    ``hermes auth add xai-oauth`` never reaches this: it adds its own ``manual:device_code``
    row with a fresh status, which rotation selects while the ``device_code`` row stays benched.
    """
    if provider not in _LOGIN_STAMPED_PROVIDERS:
        return False
    bench_start = (_parse_absolute_timestamp(entry.last_status_at)
                   or _parse_absolute_timestamp(entry.extra.get("billing_streak_at")))
    logged_in_at = _parse_absolute_timestamp(state.get("logged_in_at") if isinstance(state, dict) else None)
    if logged_in_at is None:
        return True
    return bench_start is not None and logged_in_at <= bench_start


def note_kept_billing_bench(provider: str, entry: Any) -> None:
    """INFO once per adoption that kept a billing bench: name when it ends and what clears it."""
    from agent.credential_pool import _exhausted_until

    until = _exhausted_until(entry)
    when = datetime.fromtimestamp(until, UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if until else "its TTL"
    logger.info(
        "%s: token refresh keeps billing bench on %s until %s; re-login or "
        "`hermes auth reset %s` clears it", provider, entry.id, when, provider)
