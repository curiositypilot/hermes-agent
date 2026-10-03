"""Complexity-tier model routing for Kanban workers.

A task labelled ``S``/``M``/``L`` (``tasks.complexity``) with no pinned model
gets its worker model from the ``kanban.routing.tiers`` list for that label.
The dispatcher walks the list in order and takes the first candidate whose
provider is currently usable, so a quota wall on one provider moves the next
card to the next model instead of parking it behind a fixed cooldown.

"Usable" is judged from two read-only signals, both already written by the
normal runtime:

* the assignee profile's credential pool (``auth.json``): a provider whose
  every credential is ``exhausted`` (429 / quota) or benched for this model is
  skipped until its reset;
* the board's own history: a candidate whose worker exited ``rate_limited``
  (EX_TEMPFAIL) within ``cooldown_seconds`` is skipped. The dispatcher records
  the chosen model on every routed run (``routed`` event, ``run_id`` scoped),
  which is what makes this lookup possible.

Nothing here writes to ``auth.json`` or refreshes tokens; routing never costs
an API call. A provider with no pool entries (key inline in config, local
proxy) is treated as available — the worker's own error handling still
applies.

Precedence: ``model_override`` (pinned) > tier routing > the profile's own
model. The routed model is applied to the in-memory task at spawn time only;
it is never persisted on the card, so every retry routes afresh.

Review lane: ``kanban.routing.review`` is an ordered candidate list for the
reviewer run (sdlc-review), walked the same way. When set it wins over the
card's pin and reasoning: those were chosen for the implementer, and a
reviewer on a different model than the implementer catches different defects.
Unset, review runs keep the card pin / profile model.

Quota gate (``kanban.routing.quota_gate``): a provider whose live headroom
(``data/quota/headroom.json``, written by the quota-snapshot cron) is below a
band only takes cards whose priority meets that band's ``min_priority``. A
gated candidate is skipped like an unavailable one, so the card moves to a
provider with headroom; when every candidate is gated (or rate-limited with at
least one gate skip) the card is held as ``quota_held`` — no failure counted,
``on_exhausted: profile`` does not override it, and pay-as-you-go providers
are not used as a fallthrough for it. Unknown headroom (no/stale/unreadable
file, provider missing, ``ok: false``) leaves the gate open.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

TIER_ORDER = ("S", "M", "L")
DEFAULT_COOLDOWN_SECONDS = 900
VALID_UNLABELED = ("profile", *TIER_ORDER)
VALID_ON_EXHAUSTED = ("wait", "profile")


# --- Config ---------------------------------------------------------------


@dataclass(frozen=True)
class TierCandidate:
    model: str
    provider: Optional[str] = None
    reasoning: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.provider else self.model

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "provider": self.provider, "reasoning": self.reasoning}


DEFAULT_QUOTA_MAX_AGE_SECONDS = 1800
DEFAULT_QUOTA_BANDS: dict[str, tuple[tuple[float, int], ...]] = {
    "default": ((0.25, 1), (0.10, 2), (0.03, 3)),
}


@dataclass(frozen=True)
class QuotaGateConfig:
    enabled: bool = False
    headroom_file: str = ""
    """``""`` = ``<HERMES_HOME>/data/quota/headroom.json``."""
    max_age_seconds: int = DEFAULT_QUOTA_MAX_AGE_SECONDS
    payg_providers: tuple[str, ...] = ("openrouter",)
    bands: dict[str, tuple[tuple[float, int], ...]] = field(
        default_factory=lambda: dict(DEFAULT_QUOTA_BANDS))
    """provider (or ``default``) -> ``((below, min_priority), ...)``."""

    def bands_for(self, provider: str) -> tuple[tuple[float, int], ...]:
        if provider in self.bands:
            return self.bands[provider]
        return self.bands.get("default") or ()


@dataclass
class RoutingConfig:
    enabled: bool = False
    tiers: dict[str, tuple[TierCandidate, ...]] = field(default_factory=dict)
    review: tuple[TierCandidate, ...] = ()
    unlabeled: str = "profile"
    escalate: bool = True
    on_exhausted: str = "wait"
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    auto_label: bool = False
    auto_label_per_tick: int = 3
    quota_gate: QuotaGateConfig = field(default_factory=QuotaGateConfig)


def _parse_bands(raw: Any) -> dict[str, tuple[tuple[float, int], ...]]:
    if raw is None:
        return dict(DEFAULT_QUOTA_BANDS)
    if not isinstance(raw, dict):
        logger.warning("kanban.routing.quota_gate.bands: expected a mapping, got %r", raw)
        return dict(DEFAULT_QUOTA_BANDS)
    out: dict[str, tuple[tuple[float, int], ...]] = {}
    for provider, entries in raw.items():
        parsed: list[tuple[float, int]] = []
        for entry in (entries if isinstance(entries, list) else [entries]):
            try:
                parsed.append((float(entry["below"]), int(entry["min_priority"])))
            except (TypeError, KeyError, ValueError):
                logger.warning("kanban.routing.quota_gate.bands.%s: ignoring bad band %r", provider, entry)
        out[str(provider).strip().lower()] = tuple(parsed)
    return out


def _parse_quota_gate(raw: Any) -> QuotaGateConfig:
    if not isinstance(raw, dict):
        return QuotaGateConfig()
    try:
        max_age = int(raw.get("max_age_seconds", DEFAULT_QUOTA_MAX_AGE_SECONDS))
    except (TypeError, ValueError):
        max_age = DEFAULT_QUOTA_MAX_AGE_SECONDS
    payg = raw.get("payg_providers", ["openrouter"])
    payg_list = payg if isinstance(payg, list) else [payg] if payg else []
    return QuotaGateConfig(
        enabled=bool(raw.get("enabled", False)),
        headroom_file=str(raw.get("headroom_file") or "").strip(),
        max_age_seconds=max_age if max_age > 0 else DEFAULT_QUOTA_MAX_AGE_SECONDS,
        payg_providers=tuple(str(p).strip().lower() for p in payg_list if str(p).strip()),
        bands=_parse_bands(raw.get("bands")),
    )


def _parse_candidate(raw: Any, where: str) -> Optional[TierCandidate]:
    """One candidate mapping; ``where`` is its config path under kanban.routing (``tiers.S``, ``review``)."""
    if not isinstance(raw, dict):
        logger.warning("kanban.routing.%s: ignoring non-mapping entry %r", where, raw)
        return None
    model = str(raw.get("model") or "").strip()
    if not model:
        logger.warning("kanban.routing.%s: ignoring entry without a model: %r", where, raw)
        return None
    provider = str(raw.get("provider") or "").strip() or None
    reasoning = None
    if raw.get("reasoning") not in (None, ""):
        from hermes_cli.kanban_db import normalize_reasoning_effort
        try:
            reasoning = normalize_reasoning_effort(str(raw["reasoning"]))
        except ValueError as exc:
            logger.warning("kanban.routing.%s: %s (ignored for %s)", where, exc, model)
    return TierCandidate(model=model, provider=provider, reasoning=reasoning)


def _parse_candidates(entries: Any, where: str) -> tuple[TierCandidate, ...]:
    if entries in (None, ""):
        return ()
    parsed = [_parse_candidate(e, where) for e in (entries if isinstance(entries, list) else [entries])]
    return tuple(c for c in parsed if c is not None)


def load_routing_config(kanban_cfg: Optional[dict] = None) -> RoutingConfig:
    """Parse ``kanban.routing``; any malformed piece degrades to disabled/defaults, never raises."""
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config_readonly
            kanban_cfg = (load_config_readonly() or {}).get("kanban") or {}
        except Exception:
            kanban_cfg = {}
    raw = kanban_cfg.get("routing") if isinstance(kanban_cfg, dict) else None
    if not isinstance(raw, dict):
        return RoutingConfig()
    tiers: dict[str, tuple[TierCandidate, ...]] = {}
    raw_tiers = raw.get("tiers")
    for key, entries in (raw_tiers.items() if isinstance(raw_tiers, dict) else ()):
        tier = str(key).strip().upper()
        if tier not in TIER_ORDER:
            logger.warning("kanban.routing.tiers: unknown tier %r (expected S, M, L)", key)
            continue
        tiers[tier] = _parse_candidates(entries, f"tiers.{tier}")
    unlabeled = str(raw.get("unlabeled") or "profile").strip()
    unlabeled = unlabeled.upper() if unlabeled.upper() in TIER_ORDER else unlabeled.lower()
    if unlabeled not in VALID_UNLABELED:
        logger.warning("kanban.routing.unlabeled=%r invalid; using 'profile'", raw.get("unlabeled"))
        unlabeled = "profile"
    on_exhausted = str(raw.get("on_exhausted") or "wait").strip().lower()
    if on_exhausted not in VALID_ON_EXHAUSTED:
        logger.warning("kanban.routing.on_exhausted=%r invalid; using 'wait'", raw.get("on_exhausted"))
        on_exhausted = "wait"

    def _int(key: str, default: int, minimum: int) -> int:
        try:
            value = int(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= minimum else default

    return RoutingConfig(
        enabled=bool(raw.get("enabled", False)),
        tiers=tiers,
        review=_parse_candidates(raw.get("review"), "review"),
        unlabeled=unlabeled,
        escalate=bool(raw.get("escalate", True)),
        on_exhausted=on_exhausted,
        cooldown_seconds=_int("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS, 0),
        auto_label=bool(raw.get("auto_label", False)),
        auto_label_per_tick=_int("auto_label_per_tick", 3, 1),
        quota_gate=_parse_quota_gate(raw.get("quota_gate")),
    )


# --- Availability -----------------------------------------------------------


@dataclass
class Availability:
    available: bool
    reason: str
    until: Optional[float] = None


def _custom_pool_keys(provider: str) -> list[str]:
    """Pool keys a named ``providers:`` entry may live under (profile config view)."""
    try:
        from hermes_cli.config import load_config_readonly
        from agent.credential_pool import custom_provider_pool_key_candidates
        providers = (load_config_readonly() or {}).get("providers") or {}
        entry = providers.get(provider) if isinstance(providers, dict) else None
        if not isinstance(entry, dict):
            return []
        base_url = entry.get("api") or entry.get("base_url") or ""
        return list(custom_provider_pool_key_candidates(base_url, provider))
    except Exception:
        return []


def pool_availability(provider: Optional[str], model: Optional[str], *, now: Optional[float] = None) -> Availability:
    """Read-only credential-pool check under the CURRENT home scope.

    Mirrors ``CredentialPool._available_entries`` without its side effects
    (no persistence, no token refresh, no env seeding): DEAD rows never count,
    ``exhausted`` rows count again once their reset / TTL passes, and a
    per-model bench blocks only that model.
    """
    if not provider:
        return Availability(True, "no provider pinned (not checked)")
    now = time.time() if now is None else now
    try:
        from hermes_cli.auth import read_credential_pool
        from agent.credential_pool import (
            STATUS_DEAD, STATUS_EXHAUSTED, PooledCredential, _exhausted_until,
        )
        from agent.credential_pool_model_cooldowns import model_cooldown_until
    except Exception as exc:  # pragma: no cover - import guard
        return Availability(True, f"pool check unavailable ({type(exc).__name__})")
    key = provider.strip().lower()
    raw_entries: list = []
    for candidate in (key, *_custom_pool_keys(provider)):
        try:
            raw_entries = [e for e in read_credential_pool(candidate) if isinstance(e, dict)]
        except Exception:
            raw_entries = []
        if raw_entries:
            key = candidate
            break
    if not raw_entries:
        return Availability(True, "no pool entries (not checked)")
    entries = []
    for payload in raw_entries:
        try:
            entries.append(PooledCredential.from_dict(key, payload))
        except Exception:
            continue
    live = [e for e in entries if e.last_status != STATUS_DEAD]
    if not live:
        return Availability(False, "all credentials dead (re-auth needed)")
    sole = len(live) <= 1
    waits: list[float] = []
    for entry in live:
        bench = model_cooldown_until(entry, model)
        if bench is not None:
            waits.append(bench)
            continue
        if entry.last_status == STATUS_EXHAUSTED:
            until = _exhausted_until(entry, sole_credential=sole)
            if until is not None and now < until:
                waits.append(until)
                continue
        return Availability(True, "pool ok")
    return Availability(False, "credential pool exhausted", until=min(waits) if waits else None)


def _profile_model(profile_home: Optional[str]) -> Optional[TierCandidate]:
    """The assignee profile's own ``model.default``/``model.provider`` (best effort)."""
    try:
        from hermes_cli.config import load_config_readonly
        with _home_scope(profile_home):
            model_cfg = (load_config_readonly() or {}).get("model") or {}
        if not isinstance(model_cfg, dict):
            return None
        model = str(model_cfg.get("default") or "").strip()
        provider = str(model_cfg.get("provider") or "").strip() or None
        return TierCandidate(model=model, provider=provider) if model else None
    except Exception:
        return None


@contextlib.contextmanager
def _home_scope(profile_home: Optional[str]):
    if not profile_home:
        yield
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(profile_home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def resolve_profile_home(assignee: Optional[str]) -> Optional[str]:
    if not assignee:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
        return resolve_profile_env(normalize_profile_name(assignee))
    except Exception:
        return None


# --- Decision ---------------------------------------------------------------


@dataclass
class RouteDecision:
    source: str
    """``pinned`` (card has model_override) | ``tier`` | ``review`` (review-lane
    candidate) | ``profile`` (unlabeled or fallback) | ``exhausted`` (every
    candidate unavailable; card held) | ``quota_held`` (nothing usable and the
    quota gate refused at least one candidate; card held)."""
    lane: str = "ready"
    requested_tier: Optional[str] = None
    tier: Optional[str] = None
    candidate: Optional[TierCandidate] = None
    skipped: list[dict] = field(default_factory=list)
    retry_at: Optional[float] = None
    note: Optional[str] = None

    @property
    def applies_model(self) -> bool:
        return self.source in ("tier", "review") and self.candidate is not None

    def label(self) -> str:
        if self.source == "quota_held":
            return f"quota held — {self.note}" if self.note else "quota held"
        if self.source == "review" and self.candidate:
            return f"review -> {self.candidate.label}"
        if self.source == "exhausted" and self.lane == "review":
            return "review: no candidate available"
        if self.source == "tier" and self.candidate:
            esc = f" (escalated from {self.requested_tier})" if self.tier != self.requested_tier else ""
            return f"tier {self.tier}{esc} -> {self.candidate.label}"
        if self.source == "exhausted":
            return f"tier {self.requested_tier}: no candidate available"
        if self.source == "pinned":
            return f"pinned -> {self.candidate.label if self.candidate else '?'}"
        base = f"profile default{f' ({self.candidate.label})' if self.candidate else ''}"
        return f"{base}{f' — {self.note}' if self.note else ''}"

    def event_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"source": self.source}
        if self.lane != "ready":
            payload["lane"] = self.lane
        if self.requested_tier:
            payload["requested_tier"] = self.requested_tier
        if self.tier:
            payload["tier"] = self.tier
        if self.candidate:
            payload.update({k: v for k, v in self.candidate.as_dict().items() if v})
        if self.skipped:
            payload["skipped"] = self.skipped
        if self.note:
            payload["note"] = self.note
        return payload


def _tier_walk(requested: str, escalate: bool) -> tuple[str, ...]:
    start = TIER_ORDER.index(requested)
    return TIER_ORDER[start:] if escalate else (requested,)


# --- Quota gate ---------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _parse_iso_epoch(raw: Any) -> Optional[float]:
    if not raw:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


@dataclass
class QuotaSnapshot:
    """What the headroom file says this tick. ``providers`` holds only known
    headroom; anything else (missing, ``ok: false``, null binding) is unknown =
    gate open, listed in ``unknown`` with the reason."""
    path: str
    status: str
    """``ok`` | ``disabled`` | ``missing`` | ``unreadable`` | ``stale``."""
    age_seconds: Optional[float] = None
    providers: dict[str, tuple[float, Optional[float]]] = field(default_factory=dict)
    """provider -> ``(headroom 0..1, resets_at epoch or None)``."""
    unknown: dict[str, str] = field(default_factory=dict)


def default_headroom_path() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "data" / "quota" / "headroom.json")


def read_headroom(gate: QuotaGateConfig, *, now: Optional[float] = None) -> QuotaSnapshot:
    """Read the quota-snapshot file. Never raises, never touches the network."""
    import os
    now = time.time() if now is None else now
    path = os.path.expanduser(gate.headroom_file) if gate.headroom_file else default_headroom_path()
    if not gate.enabled:
        return QuotaSnapshot(path, "disabled")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return QuotaSnapshot(path, "missing")
    except (OSError, ValueError):
        return QuotaSnapshot(path, "unreadable")
    if not isinstance(data, dict) or not isinstance(data.get("providers"), dict):
        return QuotaSnapshot(path, "unreadable")
    try:
        age = now - float(data["written_at_epoch"])
    except (TypeError, KeyError, ValueError):
        return QuotaSnapshot(path, "unreadable")
    if age > gate.max_age_seconds:
        return QuotaSnapshot(path, "stale", age_seconds=age)
    snap = QuotaSnapshot(path, "ok", age_seconds=age)
    for name, entry in data["providers"].items():
        name = str(name)
        if not isinstance(entry, dict) or not entry.get("ok"):
            snap.unknown[name] = "fetch failed (ok: false)"
            continue
        binding = entry.get("binding")
        if not isinstance(binding, dict):
            snap.unknown[name] = "no binding window"
            continue
        try:
            headroom = float(binding["headroom"])
        except (TypeError, KeyError, ValueError):
            snap.unknown[name] = "no binding window"
            continue
        snap.providers[name.strip().lower()] = (headroom, _parse_iso_epoch(binding.get("resets_at")))
    return snap


def required_priority(bands: tuple[tuple[float, int], ...], headroom: float) -> tuple[int, Optional[float]]:
    """``(min priority, the band threshold that set it)``; ``(0, None)`` when no band matches."""
    matched = [(below, prio) for below, prio in bands if headroom < below]
    if not matched:
        return 0, None
    required = max(prio for _below, prio in matched)
    return required, min(below for below, prio in matched if prio == required)


@dataclass
class QuotaVerdict:
    """One provider refused by the gate for one card."""
    provider: str
    headroom: float
    below: float
    required: int
    resets_at: Optional[float] = None

    def describe(self) -> str:
        return f"{self.provider} headroom {_pct(self.headroom)} < {_pct(self.below)} needs P{self.required}"


@dataclass
class _Walk:
    chosen: Optional[tuple[Optional[str], TierCandidate]] = None
    skipped: list[dict] = field(default_factory=list)
    waits: list[float] = field(default_factory=list)
    gated: list[QuotaVerdict] = field(default_factory=list)


def _task_priority(task: Any) -> int:
    try:
        return int(getattr(task, "priority", 0) or 0)
    except (TypeError, ValueError):
        return 0


class RoutingContext:
    """Per-tick routing state: config, board rate-limit history, memoised checks."""

    def __init__(self, conn: sqlite3.Connection, cfg: RoutingConfig, *, now: Optional[float] = None) -> None:
        self.conn = conn
        self.cfg = cfg
        self.now = time.time() if now is None else now
        self._recent: Optional[dict[tuple[str, str], int]] = None
        self._memo: dict[tuple[Optional[str], Optional[str], str], Availability] = {}
        self._quota: Optional[QuotaSnapshot] = None

    def quota(self) -> QuotaSnapshot:
        """The headroom file, read once per context (per dispatcher tick)."""
        if self._quota is None:
            self._quota = read_headroom(self.cfg.quota_gate, now=self.now)
        return self._quota

    def quota_verdict(self, provider: Optional[str], priority: int) -> Optional[QuotaVerdict]:
        """The gate's refusal of ``provider`` for a card of ``priority``, or None
        (gate off, headroom unknown, or the card's priority meets the band)."""
        gate = self.cfg.quota_gate
        if not gate.enabled or not provider:
            return None
        snap = self.quota()
        if snap.status != "ok":
            return None
        key = provider.strip().lower()
        known = snap.providers.get(key)
        if known is None:
            return None
        headroom, resets_at = known
        required, below = required_priority(gate.bands_for(key), headroom)
        if below is None or priority >= required:
            return None
        return QuotaVerdict(key, headroom, below, required, resets_at)

    def _is_payg(self, provider: Optional[str]) -> bool:
        return bool(provider) and provider.strip().lower() in self.cfg.quota_gate.payg_providers

    def _walk(self, entries: list[tuple[Optional[str], TierCandidate]], task: Any,
              profile_home: Optional[str]) -> _Walk:
        """First usable candidate of ``(tier, candidate)`` entries in order. A
        quota-gated candidate is skipped like an unavailable one; after the
        first gate skip, pay-as-you-go providers are skipped too."""
        walk = _Walk()
        priority = _task_priority(task)
        for tier, cand in entries:
            where = {"tier": tier} if tier else {}
            if walk.gated and self._is_payg(cand.provider):
                walk.skipped.append({**where, "model": cand.model, "provider": cand.provider,
                                     "reason": "quota: no paid fallthrough"})
                continue
            verdict = self.quota_verdict(cand.provider, priority)
            if verdict is not None:
                walk.gated.append(verdict)
                walk.skipped.append({**where, "model": cand.model, "provider": cand.provider,
                                     "reason": f"quota: {verdict.describe()}"})
                continue
            avail = self.availability(cand, profile_home)
            if avail.available:
                walk.chosen = (tier, cand)
                return walk
            walk.skipped.append({**where, "model": cand.model, "provider": cand.provider, "reason": avail.reason})
            if avail.until:
                walk.waits.append(avail.until)
        return walk

    def _quota_held(self, task: Any, gated: list[QuotaVerdict], *, lane: str = "ready",
                    requested_tier: Optional[str] = None, skipped: Optional[list[dict]] = None,
                    candidate: Optional[TierCandidate] = None) -> RouteDecision:
        seen: dict[str, QuotaVerdict] = {}
        for v in gated:
            seen.setdefault(v.provider, v)
        resets = [v.resets_at for v in seen.values() if v.resets_at]
        note = "; ".join(v.describe() for v in seen.values()) + f" (card P{_task_priority(task)})"
        return RouteDecision("quota_held", lane=lane, requested_tier=requested_tier, candidate=candidate,
                             skipped=list(skipped or []), retry_at=min(resets) if resets else None, note=note)

    def _gate_fixed(self, task: Any, decision: RouteDecision) -> RouteDecision:
        """Pinned / profile routes have no alternative: a gated provider holds the card."""
        cand = decision.candidate
        verdict = self.quota_verdict(cand.provider if cand else None, _task_priority(task))
        if verdict is None:
            return decision
        skipped = [*decision.skipped, {"model": cand.model, "provider": cand.provider,
                                       "reason": f"quota: {verdict.describe()}"}]
        return self._quota_held(task, [verdict], lane=decision.lane, requested_tier=decision.requested_tier,
                                skipped=skipped, candidate=cand)

    def recently_rate_limited(self) -> dict[tuple[str, str], int]:
        """``(provider, model) -> latest ended_at`` of routed runs that exited
        ``rate_limited`` within ``cooldown_seconds`` (whole board)."""
        if self._recent is not None:
            return self._recent
        recent: dict[tuple[str, str], int] = {}
        if self.cfg.cooldown_seconds > 0:
            try:
                rows = self.conn.execute(
                    "SELECT e.payload AS payload, r.ended_at AS ended_at FROM task_runs r "
                    "JOIN task_events e ON e.run_id = r.id AND e.kind = 'routed' "
                    "WHERE r.outcome = 'rate_limited' AND r.ended_at >= ?",
                    (int(self.now) - self.cfg.cooldown_seconds,),
                ).fetchall()
            except sqlite3.Error:
                rows = []
            for row in rows:
                try:
                    payload = json.loads(row["payload"] or "{}")
                except (TypeError, ValueError):
                    continue
                model = str(payload.get("model") or "")
                if not model:
                    continue
                k = (str(payload.get("provider") or ""), model)
                recent[k] = max(recent.get(k, 0), int(row["ended_at"] or 0))
        self._recent = recent
        return recent

    def availability(self, cand: TierCandidate, profile_home: Optional[str]) -> Availability:
        memo_key = (cand.provider, cand.model, str(profile_home or ""))
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached
        ended = self.recently_rate_limited().get((cand.provider or "", cand.model))
        if ended is not None:
            result = Availability(False, "rate_limited on this board", until=ended + self.cfg.cooldown_seconds)
        else:
            with _home_scope(profile_home):
                result = pool_availability(cand.provider, cand.model, now=self.now)
        self._memo[memo_key] = result
        return result

    def requested_tier(self, task: Any) -> Optional[str]:
        tier = getattr(task, "complexity", None)
        if tier in TIER_ORDER:
            return tier
        return self.cfg.unlabeled if self.cfg.unlabeled in TIER_ORDER else None

    def decide(self, task: Any, profile_home: Optional[str] = None) -> RouteDecision:
        if getattr(task, "model_override", None):
            return self._gate_fixed(task, RouteDecision(
                "pinned", candidate=TierCandidate(task.model_override, getattr(task, "provider_override", None))))
        requested = self.requested_tier(task)
        if requested is None:
            return self._gate_fixed(task, RouteDecision("profile", candidate=_profile_model(profile_home),
                                                        note="unlabeled"))
        tiers = _tier_walk(requested, self.cfg.escalate)
        if not any(self.cfg.tiers.get(t) for t in tiers):
            return self._gate_fixed(task, RouteDecision(
                "profile", requested_tier=requested, candidate=_profile_model(profile_home),
                note=f"no candidates configured for tier {requested}"))
        walk = self._walk([(t, c) for t in tiers for c in self.cfg.tiers.get(t, ())], task, profile_home)
        if walk.chosen is not None:
            tier, cand = walk.chosen
            return RouteDecision("tier", requested_tier=requested, tier=tier, candidate=cand, skipped=walk.skipped)
        if walk.gated:
            # Never on_exhausted=profile here: that would spend the gated quota anyway.
            return self._quota_held(task, walk.gated, requested_tier=requested, skipped=walk.skipped)
        if self.cfg.on_exhausted == "profile":
            return self._gate_fixed(task, RouteDecision(
                "profile", requested_tier=requested, candidate=_profile_model(profile_home),
                skipped=walk.skipped, note="all tier candidates unavailable"))
        return RouteDecision("exhausted", requested_tier=requested, skipped=walk.skipped,
                             retry_at=min(walk.waits) if walk.waits else None)

    def decide_review(self, task: Any, profile_home: Optional[str] = None) -> Optional[RouteDecision]:
        """Reviewer model from ``kanban.routing.review``; None when no review
        candidates are configured (the run keeps the card pin / profile model).
        Deliberately ignores the card's pin and complexity: both describe the
        implementer's run. The quota gate applies with the card's priority."""
        if not self.cfg.review:
            return None
        walk = self._walk([(None, c) for c in self.cfg.review], task, profile_home)
        if walk.chosen is not None:
            return RouteDecision("review", lane="review", candidate=walk.chosen[1], skipped=walk.skipped)
        if walk.gated:
            return self._quota_held(task, walk.gated, lane="review", skipped=walk.skipped)
        if self.cfg.on_exhausted == "profile":
            return RouteDecision("profile", lane="review", skipped=walk.skipped,
                                 note="all review candidates unavailable")
        return RouteDecision("exhausted", lane="review", skipped=walk.skipped,
                             retry_at=min(walk.waits) if walk.waits else None)

    def decide_for_lane(self, task: Any, lane: str, profile_home: Optional[str] = None) -> Optional[RouteDecision]:
        return self.decide_review(task, profile_home) if lane == "review" else self.decide(task, profile_home)


def apply_route(task: Any, decision: Optional[RouteDecision]) -> None:
    """Pin the routed model on the IN-MEMORY task used to build the worker argv.
    A task-level ``reasoning_effort`` still wins over a tier candidate's; a
    review candidate replaces it (the card's effort was set for the implementer)."""
    if decision is None or not decision.applies_model or decision.candidate is None:
        return
    cand = decision.candidate
    task.model_override = cand.model
    task.provider_override = cand.provider
    if decision.source == "review":
        task.reasoning_effort = cand.reasoning
    elif cand.reasoning and not getattr(task, "reasoning_effort", None):
        task.reasoning_effort = cand.reasoning


# --- Complexity estimate (auxiliary model) ----------------------------------

ESTIMATE_SYSTEM_PROMPT = (
    "You estimate how much work an autonomous coding agent will spend on a "
    "kanban task. Given the task title and description, respond with STRICT "
    "JSON only (no prose, no code fence):\n"
    '{"est_tokens": <integer total tokens across the whole run>, '
    '"complexity": "S"|"M"|"L", '
    '"rationale": "<one short sentence>"}\n'
    "Base the token figure on a realistic multi-turn agent run (reading files, "
    "tool calls, edits, retries) — not a single reply. S≈small/localized, "
    "M≈multi-file, L≈broad or ambiguous. Be honest that this is a rough guess.")


def _cap(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def estimate_complexity(title: str, body: Optional[str], *, task_id: Optional[str]) -> dict:
    """Aux-model S/M/L + token estimate. Never raises — errors come back as
    ``{"ok": False, "reason"}``. Shared by the dashboard and auto-labelling."""
    if not (title or "").strip():
        return {"ok": False, "reason": "a title is required to estimate"}
    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return {"ok": False, "reason": "auxiliary client unavailable"}
    user_msg = f"Title: {_cap(title, 400)}\n\nDescription:\n{_cap(body, 4000) or '(none)'}"
    # Headless aux calls need a relay affinity key (#112043); the create dialog
    # has no task yet, so it shares one stable key.
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{task_id or 'estimate'}")
    try:
        resp = call_llm(
            task="kanban_estimator",
            messages=[{"role": "system", "content": ESTIMATE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.0, max_tokens=300, timeout=60)
    except Exception as exc:
        return {"ok": False, "reason": f"LLM error: {type(exc).__name__}"}
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    try:
        raw = (resp.choices[0].message.content or "").strip()
        model = getattr(resp, "model", None)
    except Exception:
        raw, model = "", None
    try:
        m = None if raw.lstrip().startswith("{") else re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        parsed = obj if isinstance(obj, dict) else None
    except Exception:
        parsed = None
    if not parsed:
        return {"ok": False, "reason": "could not parse an estimate from the model"}
    try:
        est_tokens = int(parsed.get("est_tokens") or 0)
    except (TypeError, ValueError):
        est_tokens = 0
    complexity = str(parsed.get("complexity") or "").strip().upper()
    return {
        "ok": True, "est_tokens": est_tokens, "complexity": complexity if complexity in TIER_ORDER else None,
        "rationale": str(parsed.get("rationale") or "").strip() or None, "model": model}


def unlabeled_ready_ids(conn: sqlite3.Connection, limit: int) -> list[str]:
    """Ready/todo cards with no label and no pinned model, dispatch order."""
    rows = conn.execute(
        "SELECT id FROM tasks WHERE status IN ('ready', 'todo') AND claim_lock IS NULL "
        "AND (complexity IS NULL OR complexity = '') "
        "AND (model_override IS NULL OR model_override = '') "
        "AND NOT EXISTS (SELECT 1 FROM task_events e WHERE e.task_id = tasks.id "
        "                AND e.kind IN ('complexity_set', 'complexity_estimate_failed')) "
        "ORDER BY CASE status WHEN 'ready' THEN 0 ELSE 1 END, priority DESC, created_at ASC LIMIT ?",
        (int(limit),),
    ).fetchall()
    return [r["id"] for r in rows]


def auto_label(conn: sqlite3.Connection, limit: int) -> list[tuple[str, Optional[str]]]:
    """Estimate + persist S/M/L for up to ``limit`` unlabeled cards. A failed
    estimate is recorded once (``complexity_estimate_failed``) so the card is
    not re-estimated every tick; ``set-complexity`` still works on it."""
    from hermes_cli import kanban_db as kb
    out: list[tuple[str, Optional[str]]] = []
    for task_id in unlabeled_ready_ids(conn, limit):
        task = kb.get_task(conn, task_id)
        if task is None:
            continue
        est = estimate_complexity(task.title, task.body, task_id=task_id)
        if est.get("ok") and est.get("complexity"):
            kb.set_complexity(conn, task_id, est["complexity"], source="estimator")
            out.append((task_id, est["complexity"]))
        else:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "complexity_estimate_failed",
                                 {"reason": est.get("reason") or "no complexity in reply"})
            out.append((task_id, None))
    return out
