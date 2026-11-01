"""Scoped Kanban dispatch hold for ``hermes gateway restart --when-idle``.

A plain restart refuses while work is live, but the embedded dispatcher keeps spawning new runs, so
a restart retried every few minutes only lands on an idle instant by luck (t_50ffb4fd: 86 minutes on
a busy board). The hold stops new Kanban spawns in ONE gateway process while the restart waits for
the runs already going to finish; chat, cron and running workers are untouched (unlike ESTOP).

The file lives next to the default board DB. It counts only for the gateway whose pid it names and
only until ``expires_at``, so the replacement gateway (new pid) and a killed CLI (TTL) both clear it
without anyone releasing it. Anything unreadable counts as no hold: the hold is an optimisation, not
a safety gate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

HOLD_FILE = "restart_hold.json"


def hold_path() -> Path:
    from hermes_cli.kanban_db import kanban_home

    return kanban_home() / HOLD_FILE


def engage(gateway_pid: int, ttl_s: float, reason: str) -> dict[str, Any]:
    """Hold dispatch in the gateway ``gateway_pid`` for ``ttl_s`` seconds; returns the record written."""
    from utils import atomic_json_write

    now = time.time()
    record = {"gateway_pid": int(gateway_pid), "expires_at": now + float(ttl_s), "reason": reason, "engaged_at": now}
    path = hold_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, record)
    return record


def release() -> None:
    """Remove the hold; a missing file is fine (idempotent)."""
    hold_path().unlink(missing_ok=True)


def active_hold(pid: Optional[int] = None) -> Optional[dict[str, Any]]:
    """The hold record when it names ``pid`` (default: this process) and has not expired, else None."""
    try:
        record = json.loads(hold_path().read_text(encoding="utf-8-sig"))
        holds = (int(record["gateway_pid"]) == (os.getpid() if pid is None else pid)
                 and time.time() < float(record["expires_at"]))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return record if holds else None


def active(pid: Optional[int] = None) -> bool:
    return active_hold(pid) is not None
