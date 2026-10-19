"""Route cron failure notices to a card instead of a chat (``cron.failure_route``).

When ``cron.failure_route.command`` is configured, every failure-category notice that
``_deliver_result(for_failure=True)`` would send (run failures, crashed runs, blocked-config and
drift alerts) is first handed to that command as JSON on stdin::

    {"source": "cron", "job_id", "job_name", "text", "output_file"}

Exit 0 means the command took ownership (it filed or updated a card): nothing is sent to the
chat. Any other outcome (not configured, non-zero exit, timeout, crash) returns ``None`` and the
caller delivers exactly as before, so an error is never dropped without either a card or the old
chat notice. ``failure_deliver: local`` jobs are routed too: their failures used to vanish.

Config (``config.yaml``)::

    cron:
      failure_route:
        command: [/path/to/python, /path/to/router.py, route, --json]
        timeout: 180          # seconds, default 180
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180


def _route_config(cfg: Optional[dict]) -> tuple[list[str], int]:
    route = ((cfg or {}).get("cron") or {}).get("failure_route") or {}
    if isinstance(route, (str, list)):
        route = {"command": route}
    if not isinstance(route, dict) or route.get("enabled") is False:
        return [], DEFAULT_TIMEOUT
    cmd = route.get("command") or []
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    try:
        timeout = max(5, int(route.get("timeout") or DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    return [str(c) for c in cmd], timeout


def _latest_output(job_id: str) -> Optional[str]:
    try:
        from cron.jobs import OUTPUT_DIR
        files = sorted(Path(OUTPUT_DIR, job_id).glob("*"), key=lambda p: p.stat().st_mtime)
        return str(files[-1]) if files else None
    except Exception:
        return None


def route_failure(job: dict, content: str, cfg: Optional[dict] = None) -> Optional[dict]:
    """Hand one failure notice to the configured router. Returns its JSON result (truthy) when it
    took ownership, else ``None`` (caller falls back to chat delivery). Never raises."""
    try:
        if cfg is None:
            from cron import scheduler as _sched
            cfg = _sched.load_config()
        cmd, timeout = _route_config(cfg)
        if not cmd or not (content or "").strip():
            return None
        payload = {"source": "cron", "job_id": str(job.get("id") or ""),
                   "job_name": str(job.get("name") or job.get("id") or ""),
                   "text": content, "output_file": _latest_output(str(job.get("id") or ""))}
        proc = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Job '%s': failure router timed out; delivering the notice to chat instead",
                       job.get("id", "?"))
        return None
    except Exception as exc:
        logger.warning("Job '%s': failure router unavailable (%s); delivering the notice to chat",
                       job.get("id", "?"), exc)
        return None
    if proc.returncode != 0:
        logger.warning("Job '%s': failure router exit %s (%s); delivering the notice to chat",
                       job.get("id", "?"), proc.returncode, (proc.stderr or "").strip()[-300:])
        return None
    result: Any
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {"raw": proc.stdout.strip()[-300:]}
    if not isinstance(result, dict):
        result = {"raw": str(result)}
    result = result or {"routed": True}
    logger.info("Job '%s': failure routed to card %s (%s)", job.get("id", "?"),
                result.get("card", "?"), result.get("action", "routed"))
    return result
