"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.

An account-wide 429 is the exception: a subscription window wall (unified-limit
claim ``five_hour`` / ``seven_day``, or, without those headers, an OAuth row's
"exceed your account's rate limit" body) refuses every model until the window
resets, so it benches the whole credential like a billing failure does
(t_e0c3a616). A ``seven_day_<family>`` claim stays a per-model cooldown.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

# A usage-window plan limit: this model is refused until the allowance resets; every other model
# (a Free or non-Large one) stays usable on the same credential.
FAILURE_REASON_USAGE_WINDOW = "usage_window"
# The OAuth subscription wall's body. API-key rows get the same wording for per-model limits, so it
# only decides scope on an OAuth row whose 429 carried no unified-limit header verdict.
_ACCOUNT_WALL_MESSAGE_RE = re.compile(r"exceed your account's rate limit", re.IGNORECASE)

logger = logging.getLogger(__name__)


def pool_failure_reason(reason: Any, error_context: Optional[dict[str, Any]], billing_unverified: bool) -> str:
    """The pool's failure reason for a rotation (*reason* is a ``FailoverReason``). A usage-window 402
    benches only the refused model; an ambiguous billing body is sized as transient, not a 1-hour bench."""
    from agent.error_classifier import FailoverReason
    if reason == FailoverReason.billing and (error_context or {}).get("usage_window"):
        return FAILURE_REASON_USAGE_WINDOW
    if reason == FailoverReason.billing and billing_unverified:
        from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
        return FAILURE_REASON_BILLING_UNVERIFIED
    return reason.value


def model_cooldown_until(entry: PooledCredential, model: Optional[str]) -> Optional[float]:
    """Active cooldown blocking *entry* for *model*, or ``None``.

    Callers that do not know the model stay conservative: any active model
    cooldown blocks them, so an unscoped route cannot reuse the credential.
    """
    cooldowns = entry.model_cooldowns or {}
    values = cooldowns.values() if not model else (cooldowns.get(model),)
    now = time.time()
    active = [float(until) for until in values if isinstance(until, (int, float)) and until > now]
    return max(active) if active else None


def merge_model_cooldowns(*maps: Any) -> dict[str, float]:
    """Latest reset per model across snapshots — each writer only observed its own model."""
    merged: dict[str, float] = {}
    for cooldowns in maps:
        if not isinstance(cooldowns, dict):
            continue
        for model, until in cooldowns.items():
            if isinstance(until, (int, float)):
                merged[model] = max(float(until), merged.get(model, 0.0))
    return merged


def _is_account_wide_429(error_context: Optional[dict[str, Any]], entry: Optional[PooledCredential]) -> bool:
    """A 429 that walls the whole credential: header verdict first, else the OAuth account-wall body."""
    context = error_context if isinstance(error_context, dict) else {}
    scope = context.get("rate_limit_scope")
    if scope:
        return scope == "account"
    message = context.get("message")
    return (
        getattr(entry, "auth_type", None) == "oauth"
        and isinstance(message, str) and bool(_ACCOUNT_WALL_MESSAGE_RE.search(message))
    )


class CredentialPoolModelCooldownMixin:
    def token_is_blocked(self, token: str, *, model: Optional[str] = None) -> bool:
        """Whether a pool cooldown blocks *token* for *model*: a model cooldown, or a credential-wide
        429 bench (an account wall refuses every model).

        Closes the paths that hand out a native Anthropic token without
        selecting it from the pool (env / borrowed credentials). Tokens the
        pool does not know fail open: no row can attribute a cooldown to them.
        """
        with self._lock:
            return self.account_wall_until(token) is not None or any(
                entry.runtime_api_key == token and model_cooldown_until(entry, model) is not None
                for entry in self._entries
            )

    def account_wall_until(self, token: Optional[str]) -> Optional[float]:
        """End of an active credential-wide 429 bench on a row holding *token*, else ``None``.

        Billing 429s and auth benches are not walls: they keep their own handling.
        """
        from agent.credential_pool import (
            FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED, STATUS_EXHAUSTED, _exhausted_until,
        )

        if not token:
            return None
        now = time.time()
        with self._lock:
            sole = self._is_sole_credential()
            walls = [
                until for entry in self._entries
                if entry.runtime_api_key == token and entry.last_status == STATUS_EXHAUSTED
                and entry.last_error_code == 429
                and entry.failure_reason not in (FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED)
                and (until := _exhausted_until(entry, sole_credential=sole)) and until > now
            ]
        return max(walls) if walls else None

    def _is_model_scoped_failure(
        self, status_code: Optional[int], model: Optional[str], failure_reason: Optional[str],
        error_context: Optional[dict[str, Any]] = None, entry: Optional[PooledCredential] = None,
    ) -> bool:
        """Anthropic per-model 429s and a usage-window plan limit: the credential stays valid for
        every other model. (Fork: a Codex model entitlement 400 no longer benches anything, see
        error_classifier.)

        An account-wide Anthropic 429 is not model-scoped (logged here): the unified-limit header
        verdict ``rate_limit_scope == "account"`` (claim ``five_hour`` / ``seven_day``), or, with no
        header verdict, an OAuth *entry* whose body says "exceed your account's rate limit".
        ``rate_limit_scope == "model"`` (a ``seven_day_<family>`` claim) stays model-scoped, and so
        does an API-key row without headers, whose per-model limits use the same wording.
        """
        from agent.credential_pool import FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED

        if not model:
            return False
        if failure_reason == FAILURE_REASON_USAGE_WINDOW:
            return True
        if not (
            self.provider == "anthropic" and status_code == 429
            and failure_reason not in (FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED)
        ):
            return False
        if _is_account_wide_429(error_context, entry):
            self._log_account_wall(entry, error_context)
            return False
        return True

    @staticmethod
    def _log_account_wall(entry: Optional[PooledCredential], error_context: Optional[dict[str, Any]]) -> None:
        from agent.credential_pool import _normalize_error_context

        reset_at = _normalize_error_context(error_context).get("reset_at")
        until = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(reset_at)) if reset_at else "the provider TTL"
        label = (entry.label or entry.id[:8]) if entry is not None else "?"
        logger.info(
            "credential pool: %s account-wide rate limit (claim=%s) until %s; benching all models",
            label, (error_context or {}).get("rate_limit_claim") or "none", until,
        )

    def _cool_down_model(
        self, entry: PooledCredential, model: str, error_context: Optional[dict[str, Any]],
    ) -> None:
        """Record a cooldown for *model* on *entry* and every sibling sharing its key.

        Same TTL policy as a credential-wide 429 (provider ``reset_at`` wins, a
        sole credential keeps its short bench).
        Siblings matter because a ``model_config`` twin seeded from the same key
        would otherwise be re-selected for the very model that just failed.
        Caller holds the lock.
        """
        from agent.credential_pool import _exhausted_ttl, _normalize_error_context

        until = _normalize_error_context(error_context).get("reset_at") or (
            time.time() + _exhausted_ttl(429, sole_credential=self._is_sole_credential())
        )
        failed_key = entry.runtime_api_key
        for scoped in list(self._entries):
            if scoped.id != entry.id and not (failed_key and scoped.runtime_api_key == failed_key):
                continue
            cooldowns = merge_model_cooldowns(scoped.model_cooldowns, {model: until})
            self._adopt(scoped, persist=False, model_cooldowns=cooldowns)
        self._persist()

    def limit_state(self, models: Iterable[str]) -> Optional[dict[str, Any]]:
        """What a picker should say about this pool's rate limits, or ``None`` when nothing is limited.

        ``{"scope": "account", "resets_at": epoch}`` when every live entry is benched credential-wide
        (the whole login is out; another model won't help), else ``{"scope": "models", "models":
        {model: epoch}}`` for the given *models* no usable entry can serve until a model cooldown
        ends. Read-only: never clears or persists a cooldown.
        """
        from agent.credential_pool import _exhausted_until
        from agent.credential_pool_capacity import capacity_rows

        now = time.time()
        with self._lock:
            live = capacity_rows(self._entries)
            sole = len(live) <= 1
            benched = {entry.id: until for entry in live
                       if (until := _exhausted_until(entry, sole_credential=sole)) and until > now}
            if live and len(benched) == len(live):
                return {"scope": "account", "resets_at": min(benched.values())}
            usable = [entry for entry in live if entry.id not in benched]
            cooled: dict[str, float] = {}
            for model in models:
                waits = [model_cooldown_until(entry, model) for entry in usable]
                if waits and all(waits):
                    cooled[model] = min(waits)
        return {"scope": "models", "models": cooled} if cooled else None
