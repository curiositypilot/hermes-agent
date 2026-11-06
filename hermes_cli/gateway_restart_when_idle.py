"""``hermes gateway restart --when-idle``: hold new Kanban work, wait for the rest, then restart.

A plain restart refuses while work is live and leaves the dispatcher spawning, so a retry loop waits
for an idle instant that a busy board may never offer (t_50ffb4fd). This mode engages the scoped
dispatch hold (``kanban_restart_hold``) on the gateway the restart targets, polls its live work until
none is left, then runs the normal restart path unchanged (``refuse_while_sessions_live`` re-checks,
then the graceful restart). Chat and cron are never held; their gaps are short enough for the poll.

The hold is bounded (t_3d310b19): one long Kanban run that will not end inside ``--idle-timeout``
used to freeze every new spawn for the whole timeout and then exit 1 with no restart. After
``--hold-window`` seconds held, the Kanban runs this gateway claimed before the hold (read from
``task_runs``) are blockers: the hold is released while they run, then re-engaged when they end or
the gateway reports no live work, and the round repeats. ``--idle-timeout`` caps the whole call.
Accepted trade-off: a card claimed while unheld must finish too, so a restart today's hold would
land when the pre-hold runs end can be missed if that card outlives the cap (exit 1, as on any
timeout). ``--hold-window 0`` keeps the old hold-until-timeout behaviour.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

POLL_S = 10
REPORT_EVERY_S = 60
HOLD_SLACK_S = 120
HOLD_WINDOW_S = 600


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _now() -> float:
    return time.monotonic()


def _wall() -> float:
    """Epoch seconds, the clock ``task_runs.started_at`` is written in."""
    return time.time()


def preflight(args) -> None:
    """Exit 2 on a ``--when-idle`` that cannot work, before anything is held."""
    from hermes_cli.gateway import print_error

    if int(getattr(args, "hold_window", HOLD_WINDOW_S) or 0) < 0:
        print_error("--hold-window must be 0 (hold until --idle-timeout) or a positive number of seconds.")
        sys.exit(2)
    if not getattr(args, "when_idle", False):
        return
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


def gateway_kanban_runs(gateway_pid: int) -> Optional[list[dict[str, Any]]]:
    """Running Kanban runs claimed by ``gateway_pid`` on this host, across every live board.

    The gateway's ``active_work`` reports managed runs under its own pid, so the board is the only
    place that names them. Each DB is opened read-only; a missing board DB has no runs. None when a
    board DB cannot be read: callers treat that as "no identifiable blocker" and keep today's hold.
    """
    from hermes_cli import kanban_db, kanban_db_boards

    lock = f"{kanban_db._host_prefix()}{int(gateway_pid)}"
    runs: list[dict[str, Any]] = []
    for board in kanban_db_boards.list_boards(include_archived=False):
        slug = board["slug"]
        path = kanban_db.kanban_db_path(board=slug)
        if not path.is_file():
            continue
        try:
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT id, task_id, started_at FROM task_runs "
                    "WHERE status = 'running' AND ended_at IS NULL AND claim_lock = ?", (lock,)).fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        runs += [{"board": slug, "run_id": int(r[0]), "task_id": str(r[1]), "started_at": float(r[2])} for r in rows]
    return runs


def _run_line(run: dict[str, Any]) -> str:
    from hermes_cli.update_cmd_drain_report import _fmt_elapsed

    return f"{run['task_id']} run {run['run_id']} (running {_fmt_elapsed(_wall() - run['started_at'])})"


def _print_units(home: Path, work: dict[str, Any], waited_s: float, pid: int, held: bool) -> None:
    from hermes_cli.gateway_lifecycle_guards import live_count
    from hermes_cli.update_cmd_drain_report import _fmt_elapsed, describe_active_work_unit

    runs = gateway_kanban_runs(pid) or []
    state = "new Kanban work held" if held else "new Kanban work not held"
    print(f"  waiting {int(waited_s)}s: {live_count(work)} live unit(s), {state}")
    for unit in work.get("active_work") or []:
        # Managed units carry only the gateway pid; the board rows below name them instead.
        if isinstance(unit, dict) and not (runs and unit.get("kind") == "managed"):
            print(f"    • {describe_active_work_unit(unit, home)}")
    for run in runs:
        print(f"    • kanban run {run['task_id']} run {run['run_id']}, "
              f"running {_fmt_elapsed(_wall() - run['started_at'])}")


def _poll(home: Path, pid: int, start: float, timeout_s: float, held: bool,
          stop_when: Optional[Callable[[], bool]] = None) -> bool:
    """Poll until the gateway reports no live work (True) or ``stop_when()`` holds (False).

    Exit 1 once ``timeout_s`` has passed since ``start`` (the whole call, held and unheld time).
    """
    from hermes_cli import gateway_lifecycle_guards as guards

    last_report = None
    while True:
        work = guards.live_work(home)
        if work is None or not guards.live_count(work):
            return True
        waited = _now() - start
        if waited >= timeout_s:
            from hermes_cli.gateway import print_error

            lines = guards._work_lines("restart", home, work, f"hermes gateway {{verb}} {guards.INTERRUPT_FLAG}")
            print_error(f"--when-idle: still busy after {int(timeout_s)}s; the gateway was not restarted.")
            for line in lines:
                print(line)
            sys.exit(1)
        if stop_when is not None and stop_when():
            return False
        if last_report is None or waited - last_report >= REPORT_EVERY_S:
            _print_units(home, work, waited, pid, held)
            last_report = waited
        _sleep(POLL_S)


def _blockers_gone(pid: int, blockers: list[dict[str, Any]]) -> bool:
    """True once none of ``blockers`` is still running; an unreadable board re-engages the hold."""
    runs = gateway_kanban_runs(pid)
    if runs is None:
        return True
    live = {(r["board"], r["run_id"]) for r in runs}
    return not any((b["board"], b["run_id"]) in live for b in blockers)


def _wait_for_idle(home: Path, pid: int, timeout_s: float, hold_window_s: float) -> None:
    """Return with the hold engaged once ``home``'s gateway reports no live work; exit 1 on timeout."""
    from hermes_cli import kanban_restart_hold

    start = _now()
    while True:
        remaining = max(timeout_s - (_now() - start), 0)
        kanban_restart_hold.engage(pid, remaining + HOLD_SLACK_S, f"hermes gateway restart --when-idle (pid {os.getpid()})")
        held_since, t0 = _now(), _wall()
        window = (lambda since=held_since: _now() - since >= hold_window_s) if hold_window_s > 0 else None
        if _poll(home, pid, start, timeout_s, held=True, stop_when=window):
            return
        blockers = [r for r in gateway_kanban_runs(pid) or [] if r["started_at"] < t0]
        if not blockers:
            # Only chat, cron, delegation or runs claimed since T0 are left: keep holding, as before.
            _poll(home, pid, start, timeout_s, held=True)
            return
        kanban_restart_hold.release()
        print(f"--when-idle: releasing the hold after {int(_now() - held_since)}s; waiting unheld for "
              + ", ".join(_run_line(b) for b in blockers) + "…")
        # Idle here (e.g. a crashed worker whose row the dispatcher has not ended yet) also re-engages:
        # the held poll then sees the zero answer at once and the restart runs held, as before.
        _poll(home, pid, start, timeout_s, held=False, stop_when=lambda held=blockers: _blockers_gone(pid, held))
        print("--when-idle: re-engaging the Kanban dispatch hold.")


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
    hold_window_s = max(int(getattr(args, "hold_window", HOLD_WINDOW_S) or 0), 0)
    try:
        print(f"--when-idle: new Kanban work held in gateway pid {pid}; waiting up to {timeout_s}s for live work to finish"
              + (f" (hold released after {hold_window_s}s while older Kanban runs keep going)." if hold_window_s else "."))
        _wait_for_idle(home, pid, timeout_s, hold_window_s)
        yield
    finally:
        kanban_restart_hold.release()
