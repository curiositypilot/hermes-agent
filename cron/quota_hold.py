"""Hold a job's fires while a provider's usage window is known to be closed (#89376).

A quota-exhausted provider answers with an explicit ``retry after <N>s`` (Codex 429: the
``AuthError`` from ``hermes_cli.auth_codex._codex_quota_exhausted_error``). When the whole
fallback chain is unavailable, re-firing on cadence is guaranteed to fail identically until
the window reopens — every fire is a usage probe plus a delivered failure alert. The failing
run's alert says the job is held; ``mark_job_run`` then parks ``next_run_at`` at the first
scheduled occurrence after the window and stamps ``quota_hold_until`` so the stale-error
re-arm (``cron.jobs._job_is_stale_error_recurring``) does not pull the job back early.

The park is bounded: a provider that reports a multi-week window (a Codex monthly 429 says
``retry after 2581776s``) parks a job for at most ``MAX_HOLD_SECONDS`` (plus, for a cron schedule,
up to one cadence to the next legal occurrence), after which the job re-probes. Repointing a job to
another provider/model clears the hold (``cron.jobs.update_job``).

Mirror of ``cron/unreachable_retry.py`` (which pulls ``next_run_at`` EARLIER); this one pushes
it LATER. Any run that reaches the model clears the marker.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# Persisted while a hold is active: ISO instant the job was parked at.
STATE_KEY = "quota_hold_until"

# The provider's remaining seconds were measured when the probe ran; by the time the run is
# recorded a little wall clock has passed, so land clearly past the boundary.
HOLD_SLACK_SECONDS = 60

# Upper bound on how long ONE failed run parks a job, whatever window the provider reports. A
# monthly quota would otherwise park an every-15-minute job for 30 days with no way to notice the
# window reopened early or the job was repointed. Capped holds re-probe once per cap (one failed
# run, one alert) while the window stays closed. A module constant, not a config key: nobody needs
# to tune it yet (#133454).
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
    from cron.jobs import _parse_aware  # late: jobs imports this module's helpers

    until = _parse_aware(job.get(STATE_KEY)) if job.get(STATE_KEY) else None
    return until is not None and until > (now or _hermes_now())


def clear_state(job: Dict[str, Any]) -> None:
    job.pop(STATE_KEY, None)


def _effective_hold_seconds(hold_seconds: float) -> float:
    """The window a job is parked for: the provider's figure, bounded by ``MAX_HOLD_SECONDS``."""
    return min(float(hold_seconds), MAX_HOLD_SECONDS)


def _parked_at(
    schedule: Dict[str, Any], natural_next: Optional[datetime], hold_seconds: float, now: datetime,
) -> Optional[str]:
    """Where a recurring job failing at *now* is parked, or None when no park applies (the
    schedule is not recurring, or its natural next run already lands past the effective window).
    One owner for ``plan_hold`` and ``hold_notice`` so the alert never promises a park the store
    will not make."""
    from cron.jobs import compute_next_run

    if schedule.get("kind") not in {"cron", "interval"}:
        return None
    window_end = now + timedelta(seconds=_effective_hold_seconds(hold_seconds) + HOLD_SLACK_SECONDS)
    if natural_next is not None and natural_next >= window_end:
        return None
    if schedule.get("kind") == "interval":
        return window_end.isoformat()
    # First LEGAL cron occurrence after the window; parking at the boundary would fire at a time
    # the expression excludes.
    return compute_next_run(schedule, window_end.isoformat()) or window_end.isoformat()


def plan_hold(job: Dict[str, Any], hold_seconds: float) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the schedule's natural
    ``next_run_at`` for a failed run. Parks a recurring job at its first occurrence after the
    provider window (bounded by ``MAX_HOLD_SECONDS``) when that is later than the natural one.
    Returns True when parked."""
    from cron.jobs import _parse_aware

    schedule = job.get("schedule") or {}
    parked = None
    if job.get("state") != "paused":
        parked = _parked_at(
            schedule, _parse_aware(job.get("next_run_at")), hold_seconds, _hermes_now())
    if parked is None:
        clear_state(job)
        return False
    job["next_run_at"] = parked
    job[STATE_KEY] = parked
    logger.warning(
        "Job '%s': provider usage window closed for %.0fs (holding for %.0fs) — holding fires "
        "until %s instead of failing on every cadence tick",
        job.get("name", job.get("id", "?")), float(hold_seconds),
        _effective_hold_seconds(hold_seconds), parked)
    return True


def hold_notice(job: Dict[str, Any], hold_seconds: Optional[float]) -> str:
    """Line appended to the ONE failure alert delivered on entering the hold, else "" (also when
    ``plan_hold`` will not park this job, so the alert never promises a hold that is not made).

    Inside the cap the provider's window is quoted and no further alert is promised. Over the cap
    the job is held for ``MAX_HOLD_SECONDS`` only, then re-probes; a window that is still closed
    parks it again with a fresh alert, so the notice promises one alert per hold, not silence."""
    from cron.jobs import _parse_aware, compute_next_run

    schedule = job.get("schedule") or {}
    if not hold_seconds or schedule.get("kind") not in {"cron", "interval"} \
            or job.get("state") == "paused":
        return ""
    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(schedule, now.isoformat()))
    parked = _parked_at(schedule, natural_next, hold_seconds, now)
    if parked is None:
        return ""
    if float(hold_seconds) <= MAX_HOLD_SECONDS:
        window_end = now + timedelta(seconds=float(hold_seconds) + HOLD_SLACK_SECONDS)
        hours = float(hold_seconds) / 3600.0
        return (
            f"\nThe provider's usage window is closed for about {hours:.1f}h. This job is held "
            f"until its first scheduled run after {window_end.strftime('%Y-%m-%d %H:%M %Z')} — "
            "no further alerts until then."
        )
    parked_dt = _parse_aware(parked)
    hours = (parked_dt - now).total_seconds() / 3600.0
    return (
        "\nThe provider's usage window is closed for longer than this job waits at once. This "
        f"job is held for about {hours:.1f}h (through {parked_dt.strftime('%Y-%m-%d %H:%M %Z')}) "
        "and then re-probes; if the window is still closed it is held again, with one alert per "
        "hold. Repoint the job to another provider or model to clear the hold."
    )
