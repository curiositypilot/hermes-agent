"""``hermes gateway restart --when-idle``: hold new Kanban work, wait for the rest, then restart.

A plain restart refuses while work is live and leaves the dispatcher spawning, so a retry loop waits
for an idle instant that a busy board may never offer (t_50ffb4fd). This mode engages the scoped
dispatch hold (``kanban_restart_hold``) on the gateway the restart targets, polls its live work until
none is left, then runs the normal restart path unchanged (``refuse_while_sessions_live`` re-checks,
then the graceful restart). Chat and cron are never held; their gaps are short enough for the poll.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

POLL_S = 10
REPORT_EVERY_S = 60
HOLD_SLACK_S = 120


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _now() -> float:
    return time.monotonic()


def preflight(args) -> None:
    """Exit 2 on a ``--when-idle`` that cannot work, before anything is held."""
    if not getattr(args, "when_idle", False):
        return
    from hermes_cli.gateway import print_error

    if getattr(args, "interrupt", False) or getattr(args, "all", False):
        print_error("--when-idle waits for live work to finish; it cannot be combined with --yes or --all.")
        sys.exit(2)
    task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if task:
        # The caller's own managed run counts as live work, so the wait could only time out with
        # every profile's dispatch held. Its own turn is reported by the gateway pid, not the
        # worker's, so it cannot be excluded from the count: refuse instead.
        print_error(f"--when-idle refused inside Kanban task {task}: the wait counts this worker's own run, "
                    "so it could never finish. Start it outside the worker (a detached `systemd-run --user` unit).")
        sys.exit(2)


def refuse_for_served_profile(args, name: str) -> None:
    """Exit 2: a served profile restarts inside the host, so holding the host's single dispatcher for
    one profile's drain would freeze every profile's Kanban work."""
    if not getattr(args, "when_idle", False):
        return
    from hermes_cli.gateway import print_error

    print_error(f"--when-idle is not supported for profile '{name}': the host gateway serves it, and its "
                "Kanban dispatcher is shared by every profile. Restart it without --when-idle, or restart "
                "the host with `hermes gateway restart --when-idle`.")
    sys.exit(2)


def _print_units(home: Path, work: dict[str, Any], waited_s: float) -> None:
    from hermes_cli.gateway_lifecycle_guards import live_count
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit

    print(f"  waiting {int(waited_s)}s: {live_count(work)} live unit(s), new Kanban work held")
    for unit in work.get("active_work") or []:
        if isinstance(unit, dict):
            print(f"    • {describe_active_work_unit(unit, home)}")


def _wait_for_idle(home: Path, timeout_s: float) -> None:
    """Return once ``home``'s gateway reports no live work (or stops answering); exit 1 on timeout."""
    from hermes_cli import gateway_lifecycle_guards as guards

    start = _now()
    last_report = None
    while True:
        work = guards.live_work(home)
        if work is None or not guards.live_count(work):
            return
        waited = _now() - start
        if waited >= timeout_s:
            from hermes_cli.gateway import print_error

            lines = guards._work_lines("restart", home, work, f"hermes gateway {{verb}} {guards.INTERRUPT_FLAG}")
            print_error(f"--when-idle: still busy after {int(timeout_s)}s; the gateway was not restarted.")
            for line in lines:
                print(line)
            sys.exit(1)
        if last_report is None or waited - last_report >= REPORT_EVERY_S:
            _print_units(home, work, waited)
            last_report = waited
        _sleep(POLL_S)


@contextmanager
def dispatch_hold_until_idle(args) -> Iterator[None]:
    """With ``--when-idle``: hold the target gateway's dispatch, wait until idle, run the body, release.

    The target is the first answering gateway among the homes the restart guard checks, so the pid
    the hold names and the count polled come from the gateway the re-check reads. No pid (no gateway
    answers, or an older gateway answering from its state file): no hold, the normal path runs.
    """
    if not getattr(args, "when_idle", False):
        yield
        return
    from hermes_cli import gateway_lifecycle_guards as guards, kanban_restart_hold

    found = guards.first_live_work(guards._candidate_homes(getattr(args, "system", False)))
    pid = found[1].get("pid") if found else None
    if found is None or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        print("--when-idle: no gateway reported its pid; restarting without a Kanban dispatch hold.")
        yield
        return
    home = found[0]
    timeout_s = max(int(getattr(args, "idle_timeout", 3600) or 0), 0)
    try:
        kanban_restart_hold.engage(pid, timeout_s + HOLD_SLACK_S, f"hermes gateway restart --when-idle (pid {os.getpid()})")
        print(f"--when-idle: new Kanban work held in gateway pid {pid}; waiting up to {timeout_s}s for live work to finish.")
        _wait_for_idle(home, timeout_s)
        yield
    finally:
        kanban_restart_hold.release()
