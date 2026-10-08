"""Hold a job's fires while a provider's usage window is known to be closed (#89376).

A quota-exhausted provider answers with an explicit ``retry after <N>s`` (Codex 429: the
``AuthError`` from ``hermes_cli.auth_codex._codex_quota_exhausted_error``). When the whole
fallback chain is unavailable, re-firing on cadence is guaranteed to fail identically until
the window reopens — every fire is a usage probe plus a delivered failure alert. The failing
run's alert says the job is held; ``mark_job_run`` then parks ``next_run_at`` at the recovery
boundary (or the first legal occurrence after it, when several fall inside the window) and
stamps ``quota_hold_until`` so the stale-error re-arm
(``cron.jobs._job_is_stale_error_recurring``) does not pull the job back early.

The park is bounded: a provider that reports a multi-week window (a Codex monthly 429 says
``retry after 2581776s``) parks a job for at most ``MAX_HOLD_SECONDS`` (plus, for a cron schedule,
up to one cadence to the next legal occurrence), after which the job re-probes. Repointing a job to
another provider/model clears the hold (``cron.jobs.update_job``). The hold is stamped with the
route it was measured on (``ROUTE_KEY``); a due scan releases it once the job resolves to another
route through config (main ``model.provider`` or ``cron.model_provider``), and a run that was
repointed while in flight is not parked on the old route's window (``cron.jobs.mark_job_run``).

Complement to ``cron/unreachable_retry.py``: this one moves ``next_run_at`` out of a known
closed provider window. Any run that reaches the model clears the marker.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from hermes_time import now as _hermes_now, safe_strftime

logger = logging.getLogger("cron.scheduler")

# Persisted while a hold is active: ISO instant the job was parked at.
STATE_KEY = "quota_hold_until"
SCHEDULE_EXPR_KEY = "quota_hold_cron_expr"

# Persisted beside STATE_KEY: the route (``route_of``) the window was measured on. A due scan
# releases a hold whose job no longer resolves to this route (main-model or ``cron.model_provider``
# switch); a hold without it (written before the stamp existed) is kept until it expires.
ROUTE_KEY = "quota_hold_route"

# The provider's remaining seconds were measured when the probe ran; by the time the run is
# recorded a little wall clock has passed, so land clearly past the boundary.
HOLD_SLACK_SECONDS = 60

# Upper bound on how long ONE failed run parks a job, whatever window the provider reports. A
# monthly quota would otherwise park an every-15-minute job for 30 days with no way to notice the
# window reopened early or the job was repointed. Capped holds re-probe once per cap while the
# window stays closed; whether a re-probe alerts follows the normal failure-incident rules. A
# module constant, not a config key: nobody needs to tune it yet (#133454).
MAX_HOLD_SECONDS = 24 * 3600.0

_RETRY_AFTER_RE = re.compile(r"retry after (\d+)s", re.IGNORECASE)


def hold_seconds_from_failure(exc: BaseException) -> Optional[float]:
    """Seconds the provider said it will stay closed, or None when *exc* (or anything in its
    cause chain) is not a rate-limited ``AuthError`` carrying a wait hint. Anchored on the
    AuthError itself, never on arbitrary text, so an unrelated "retry after" in an agent's
    output cannot park a job."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, AuthError) and is_rate_limited_auth_error(cur):
            hint = getattr(cur, "retry_after", None)
            if hint is None:
                m = _RETRY_AFTER_RE.search(str(cur))
                hint = float(m.group(1)) if m else None
            return float(hint) if hint is not None and float(hint) > 0 else None
        cur = cur.__cause__ or cur.__context__
    return None


def hold_active(job: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """True while the job is parked inside a provider window (an expired marker is inert)."""
    from cron.jobs import _instant_after, _parse_aware  # late: jobs imports this module's helpers

    until = _parse_aware(job.get(STATE_KEY)) if job.get(STATE_KEY) else None
    return until is not None and _instant_after(until, now or _hermes_now())


def clear_state(job: Dict[str, Any]) -> None:
    job.pop(STATE_KEY, None)
    job.pop(ROUTE_KEY, None)
    job.pop(SCHEDULE_EXPR_KEY, None)


_UNSET = object()


def route_of(job: Dict[str, Any], cron_default_provider: Any = _UNSET) -> str:
    """``"<provider>@<base_url>"`` the job's provider resolve is sent to — the identity a quota
    window belongs to (the model is not part of it). Provider precedence mirrors the scheduler's
    ``_resolve_job_runtime``: per-job provider > ``cron.model_provider`` > the main
    ``model.provider`` (then ``HERMES_INFERENCE_PROVIDER``, else ``"auto"``). The scheduler passes
    the ``cron.model_provider`` it dispatched with (``_CronJobConfig.cron_default_provider``) so
    the stamp describes the fire, not config read later; otherwise it is read from config.

    Config read only, no credential resolution (``cron.jobs._main_model_pin`` resolves credentials
    and would swallow the very quota error). Known gap: an unset main provider reads ``"auto"``
    on both sides of a switch, so such a hold is kept until it expires (``MAX_HOLD_SECONDS``).
    This is the canonical identity for the hold's stamp; ``cron.jobs._route_key`` is
    ``update_job``'s per-job edit detector and is not compared against it. May raise; callers
    holding the jobs lock guard it."""
    from cron.jobs import _normalize_base_url, _normalize_job_optional_text
    from hermes_cli.runtime_provider import resolve_requested_provider

    requested = _normalize_job_optional_text(job.get("provider"))
    if not requested:
        requested = (_cron_default_provider() if cron_default_provider is _UNSET
                     else _normalize_job_optional_text(cron_default_provider))
    provider = resolve_requested_provider(requested or None)
    return f"{provider}@{_normalize_base_url(job.get('base_url')) or ''}"


def _cron_default_provider() -> Optional[str]:
    """``cron.model_provider``, read the way ``cron.scheduler._load_cron_job_config`` reads it."""
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_hermes_home

    cfg_path = get_hermes_home() / "config.yaml"
    cfg = load_user_config_effective(cfg_path) if cfg_path.exists() else {}
    cron_cfg = cfg.get("cron") if isinstance(cfg, dict) else None
    if not isinstance(cron_cfg, dict):
        return None
    return str(cron_cfg.get("model_provider") or "").strip() or None


def is_recovery_fire(job: Dict[str, Any], next_run: str) -> bool:
    """True for the exact off-lattice cron fire parked by ``plan_hold``.

    The expression fingerprint keeps a direct ``jobs.json`` schedule edit from inheriting the
    exception: edited schedules must still re-anchor without firing.
    """
    schedule = job.get("schedule") or {}
    return (
        schedule.get("kind") == "cron"
        and job.get(STATE_KEY) == next_run
        and job.get(SCHEDULE_EXPR_KEY) == schedule.get("expr")
    )


def _effective_hold_seconds(hold_seconds: float) -> float:
    """The window a job is parked for: the provider's figure, bounded by ``MAX_HOLD_SECONDS``."""
    return min(float(hold_seconds), MAX_HOLD_SECONDS)


def _window_end(hold_seconds: float, now: Optional[datetime] = None) -> datetime:
    """End of the (capped) window in real time, so a fall-back hour does not shift it."""
    from cron.jobs import _seconds_after

    return _seconds_after(now or _hermes_now(), _effective_hold_seconds(hold_seconds) + HOLD_SLACK_SECONDS)


def _recovery_worthwhile(
    job: Dict[str, Any], natural_next: datetime, window_end: datetime,
) -> bool:
    """One off-lattice recovery fire, and only for a sparse schedule.

    Bounded: a job already carrying ``quota_hold_until`` IS the recovery fire failing again, so
    it waits for the natural schedule instead of re-parking at every hold boundary (the
    probe-per-window cost the hold exists to prevent). Sparse: the natural occurrence must be at
    least half a cadence period past the boundary — the same half-period rule as
    ``cron.jobs._compute_grace_seconds`` — otherwise the recovery fire is a near-duplicate of the
    natural one (hourly job, hold ending at :58, would fire :58 AND :00).
    """
    from cron.jobs import _elapsed_seconds, _schedule_cadence_seconds

    if job.get(STATE_KEY):
        return False
    cadence = _schedule_cadence_seconds(job.get("schedule") or {})
    return bool(cadence) and _elapsed_seconds(natural_next, window_end) >= cadence / 2


def _parked_at(
    job: Dict[str, Any], natural_next: Optional[datetime], hold_seconds: float, now: datetime,
    recover_consumed_fire: bool,
) -> Tuple[Optional[str], bool]:
    """``(parked_iso, is_recovery_fire)`` for a recurring job failing at *now*, or ``(None,
    False)`` when no park applies (not recurring, or the natural next run already lands past the
    effective window and no recovery fire is due). One owner of the schedule/window arithmetic
    for ``plan_hold`` and ``hold_notice``; the caller owns the terminal-state and paused gates."""
    from cron.jobs import _instant_before, compute_next_run

    schedule = job.get("schedule") or {}
    kind = schedule.get("kind")
    if kind not in {"cron", "interval"}:
        return None, False
    window_end = _window_end(hold_seconds, now)
    blocked = natural_next is None or _instant_before(natural_next, window_end)
    recover = (kind == "cron" and not blocked and recover_consumed_fire
               and _recovery_worthwhile(job, natural_next, window_end))
    if not blocked and not recover:
        return None, False
    if kind == "cron" and blocked:
        # First LEGAL cron occurrence after the window; parking at the boundary would fire at a
        # time the expression excludes. ``compute_next_run`` returns None only when croniter is
        # missing or the schedule has no ``expr``: the job has no computable occurrence to park
        # at, so it is not parked and keeps its natural next run.
        return compute_next_run(schedule, window_end.isoformat()), False
    return window_end.isoformat(), recover


def plan_hold(
    job: Dict[str, Any], hold_seconds: float, *, recover_consumed_fire: bool = False,
) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the schedule's natural
    ``next_run_at`` for a failed run. A scheduled sparse cron may retry its consumed fire at the
    recovery boundary; manual runs keep the natural schedule. Otherwise park through the
    (``MAX_HOLD_SECONDS``-capped) window. Returns True when parked."""
    from cron.jobs import _elapsed_seconds, _parse_aware

    parked, recover = None, False
    now = _hermes_now()
    if job.get("state") != "paused":
        parked, recover = _parked_at(
            job, _parse_aware(job.get("next_run_at")), hold_seconds, now, recover_consumed_fire)
    if parked is None:
        clear_state(job)
        return False
    if recover:
        # Only the recovery fire is off-lattice; the coalesced instant is a legal occurrence.
        job[SCHEDULE_EXPR_KEY] = (job.get("schedule") or {}).get("expr")
    else:
        job.pop(SCHEDULE_EXPR_KEY, None)
    job["next_run_at"] = parked
    job[STATE_KEY] = parked
    logger.warning(
        "Job '%s': provider usage window closed for %.0fs (holding for %.0fs) — holding fires "
        "until %s instead of failing on every cadence tick",
        job.get("name", job.get("id", "?")), float(hold_seconds),
        _elapsed_seconds(_parse_aware(parked), now), parked)
    return True


def hold_notice(job: Dict[str, Any], hold_seconds: Optional[float]) -> str:
    """Line appended to the failure alert delivered on entering the hold, else "" when the job
    will not be held: not recurring, paused, not blocked by the window, or retired by this very
    run (repeat limit reached, no next run computable). Uses the same logic as ``plan_hold``, on
    the delivery-time job and clock (``mark_job_run`` reads its own, later); a scheduled fire
    (``_scheduled_instant``) may get the recovery fire, as in ``cron.scheduler``.

    Inside the cap the provider's window is quoted and no further alert is promised. Over the cap
    the job is held for ``MAX_HOLD_SECONDS`` only, then re-probes; a window that is still closed
    parks it again. Whether that re-probe alerts follows the normal failure-incident rules, so
    the notice does not promise it."""
    from cron.jobs import _elapsed_seconds, _parse_aware, compute_next_run, terminal_after_run

    schedule = job.get("schedule") or {}
    if not hold_seconds or schedule.get("kind") not in {"cron", "interval"} \
            or job.get("state") == "paused" or terminal_after_run(job):
        return ""
    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(schedule, now.isoformat()))
    parked, _recover = _parked_at(
        job, natural_next, hold_seconds, now, bool(job.get("_scheduled_instant")))
    if parked is None:
        return ""
    if float(hold_seconds) <= MAX_HOLD_SECONDS:
        hours = float(hold_seconds) / 3600.0
        return (
            f"\nThe provider's usage window is closed for about {hours:.1f}h. This job is held "
            f"through {safe_strftime(_window_end(hold_seconds, now), '%Y-%m-%d %H:%M %Z')} and "
            "resumes at the first safe opportunity afterwards — no further alerts until then."
        )
    parked_dt = _parse_aware(parked)
    hours = _elapsed_seconds(parked_dt, now) / 3600.0
    return (
        "\nThe provider's usage window is closed for longer than this job waits at once. This "
        f"job is held for about {hours:.1f}h (through {safe_strftime(parked_dt, '%Y-%m-%d %H:%M %Z')}) "
        "and then re-probes; if the window is still closed it is held again. Repoint the job to "
        "another provider or model to clear the hold (another model on the same provider may "
        "share the same quota)."
    )
