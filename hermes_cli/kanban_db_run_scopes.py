"""A Kanban run's systemd scope: minted per run at spawn, stopped once the run has ended.

The dispatcher wraps each worker in ``hermes-worker-kanban-<task>-run-<run>.scope``
(``--collect``), and a collected scope only goes away once every process inside it has
exited. A background shell the worker left behind therefore kept its scope, and itself,
alive for days after the card was done. A run's scope dies with its run, whatever the
outcome and even for ``persist_on_release`` jobs: work that must outlive its run is deployed.

Split out of ``hermes_cli.kanban_db_dispatch``; the facade is late-imported.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import subprocess
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

logger = logging.getLogger(__name__)


def _restart_safe_worker_argv(task: Task, command: list[str]) -> list[str]:
    """Wrap a systemd-hosted dispatcher's worker in the shared restart-safe scope.

    Kanban workers are long-lived agentic runs that outlive the dispatcher
    tick, so they never take cron's degraded mode under the managed gateway:
    ``require_restart_safe_scope=True`` makes the helper raise
    ``RestartSafeScopeUnavailable`` there (an infrastructure spawn failure the
    dispatcher does not charge to the card). Under any other systemd unit
    (``Type=oneshot`` dispatch timers, #113612) ``outlives_parent=True`` gets the
    worker its own scope so the unit's cgroup teardown cannot kill it.
    """
    from tools.process_registry import restart_safe_gateway_child_argv

    if task.current_run_id is None:
        # Outside managed systemd this is harmless, but a managed dispatch must
        # never mint an untraceable worker.  Check topology through the shared
        # helper first, using a placeholder suffix that cannot be launched.
        dispatch = restart_safe_gateway_child_argv(
            command,
            unit_suffix=f"kanban-{task.id}-run-missing",
            require_restart_safe_scope=True,
            outlives_parent=True,
        )
        if dispatch.mode != "in_process":
            raise RuntimeError(
                "cannot create restart-safe systemd scope for Kanban worker: "
                "the claimed task has no current run id"
            )
        return command

    return restart_safe_gateway_child_argv(
        command,
        unit_suffix=f"kanban-{task.id}-run-{task.current_run_id}",
        require_restart_safe_scope=True,
        outlives_parent=True,
    ).argv


# Task ids are ``t_<hex>``; ``run-missing`` (no current run id) never matches, so it is never stopped.
_RUN_SCOPE_UNIT = re.compile(r"^hermes-worker-kanban-(t_[0-9a-f]+)-run-([0-9]+)\.scope$")
RUN_SCOPE_REAP_INTERVAL_SECONDS = 300
# Keyed per board: the gateway ticks every board in one process back to back, so a single
# process-wide stamp would let the first board starve the rest.
_last_scan: dict[str, float] = {}


def _board_key(conn: sqlite3.Connection) -> str:
    """The board's DB file (``PRAGMA database_list``); an in-memory DB keys by connection."""
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main" and row[2]:
            return row[2]
    return f"conn:{id(conn)}"


def _active_run_scope_units() -> Optional[list[str]]:
    """Names of the user manager's ``hermes-worker-kanban-*-run-*.scope`` units, or None
    when ``systemctl`` is missing or the listing failed."""
    binary = shutil.which("systemctl")
    if binary is None:
        return None
    from tools.process_registry import systemd_user_bus_env

    result = subprocess.run(
        [binary, "--user", "list-units", "--plain", "--no-legend", "hermes-worker-kanban-*-run-*.scope"],
        capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL, env=systemd_user_bus_env(),
    )
    if result.returncode != 0:
        logger.debug("systemctl --user list-units exited %d: %s", result.returncode, result.stderr.strip())
        return None
    return [line.split()[0] for line in result.stdout.splitlines() if line.strip()]


def reap_ended_run_scopes(conn: sqlite3.Connection) -> list[str]:
    """Stop the scope of each run on THIS board that ended more than
    ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago. Several boards share one user manager, so a
    unit is stopped only when this board's ``task_runs`` holds its exact (run id, task id)
    pair; an unknown pair or a run with ``ended_at IS NULL`` (a live worker, this tick's
    caller included) is skipped. Runs at most once per ``RUN_SCOPE_REAP_INTERVAL_SECONDS``
    per board. Returns the stopped unit names."""
    board = _board_key(conn)
    now_mono = time.monotonic()
    last = _last_scan.get(board)
    if last is not None and now_mono - last < RUN_SCOPE_REAP_INTERVAL_SECONDS:
        return []
    _last_scan[board] = now_mono
    try:
        units = _active_run_scope_units()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("kanban dispatch: listing worker scopes failed: %s", exc)
        return []
    if not units:
        return []
    from hermes_cli.kanban_db_dispatch import TERMINAL_WORKER_REAP_GRACE_SECONDS

    cutoff = int(time.time()) - TERMINAL_WORKER_REAP_GRACE_SECONDS
    reaped: list[str] = []
    for unit in units:
        match = _RUN_SCOPE_UNIT.match(unit)
        if match is None:
            continue
        task_id, run_id = match.group(1), int(match.group(2))
        try:
            _reap_run_scope(conn, unit, task_id, run_id, cutoff, reaped)
        except (OSError, subprocess.SubprocessError, sqlite3.Error) as exc:
            # One unit's failure must not stop the others; a programming error still raises.
            logger.warning("kanban dispatch: run scope reap failed for %s: %s", unit, exc)
    return reaped


def _reap_run_scope(conn, unit: str, task_id: str, run_id: int, cutoff: int, reaped: list[str]) -> None:
    row = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id = ? AND task_id = ?", (run_id, task_id),
    ).fetchone()
    if row is None or row["ended_at"] is None or int(row["ended_at"]) > cutoff:
        return
    from hermes_cli import kanban_db as _kb
    from tools.process_registry import _stop_systemd_unit

    if not _stop_systemd_unit(unit):
        return
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, "run_scope_reaped", {"unit": unit, "run_id": run_id}, run_id=run_id)
    reaped.append(unit)
