"""Route failure notices to a card instead of a chat (``cron.failure_route``).

When ``cron.failure_route.command`` is configured, every failure-category notice that
``_deliver_result(for_failure=True)`` would send (run failures, crashed runs, blocked-config and
drift alerts) is first handed to that command as JSON on stdin::

    {"source": "cron", "job_id", "job_name", "text", "output_file"[, "signature"]}

Side notices of a SUCCESSFUL run are routed the same way before the report is delivered
(``cron/scheduler_side_notices.py``): the agent's file-mutation verifier footer and the
pre-agent provider-fallback line. They carry a per-job ``signature`` so repeats fold into one card.

Exit 0 means the command took ownership (it filed or updated a card): nothing is sent to the
chat. Any other outcome (not configured, non-zero exit, timeout, crash) returns ``None`` and the
caller delivers exactly as before, so an error is never dropped without either a card or the old
chat notice. ``failure_deliver: local`` jobs are routed too: their failures used to vanish.

The same key covers the gateway-side error notices: a Kanban ``gave_up`` ping is routed with
``source: kanban`` (``route_event``); ``crashed``/``timed_out`` pings (retries in flight) and the
"cron run interrupted by a restart" notice are skipped when a route is configured
(``route_configured``), since the card's events and the run record already hold them.

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
# What reading config.yaml can raise: unreadable file, InvalidUserConfigError (a RuntimeError).
_CONFIG_ERRORS = (OSError, RuntimeError, ImportError)


def _route_config(cfg: Optional[dict]) -> tuple[list[str], int]:
    cron_cfg = cfg.get("cron") if isinstance(cfg, dict) else None
    route = cron_cfg.get("failure_route") if isinstance(cron_cfg, dict) else None
    if isinstance(route, (str, list)):
        route = {"command": route}
    if not isinstance(route, dict) or route.get("enabled") is False:
        return [], DEFAULT_TIMEOUT
    cmd = route.get("command") or []
    if isinstance(cmd, str):
        try:
            cmd = shlex.split(cmd)
        except ValueError:
            logger.warning("cron.failure_route.command is not valid shell syntax: %r", cmd)
            return [], DEFAULT_TIMEOUT
    if not isinstance(cmd, list):
        logger.warning("cron.failure_route.command must be a string or list, got %r", cmd)
        return [], DEFAULT_TIMEOUT
    try:
        timeout = max(5, int(route.get("timeout") or DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    return [str(c) for c in cmd], timeout


def _load_config(cfg: Optional[dict]) -> Optional[dict]:
    if cfg is not None:
        return cfg
    from cron import scheduler as _sched
    return _sched.load_config()


def _latest_output(job_id: str) -> Optional[str]:
    try:
        from cron.jobs import OUTPUT_DIR
        files = sorted(Path(OUTPUT_DIR, job_id).glob("*"), key=lambda p: p.stat().st_mtime)
        return str(files[-1]) if files else None
    except (ImportError, OSError):
        return None


def route_configured(cfg: Optional[dict] = None) -> bool:
    """True when a failure route command is set (and not ``enabled: false``). Never raises."""
    try:
        return bool(_route_config(_load_config(cfg))[0])
    except _CONFIG_ERRORS:
        logger.warning("failure route config unreadable; treating it as unset", exc_info=True)
        return False


def route_event(payload: dict, cfg: Optional[dict] = None) -> Optional[dict]:
    """Hand one notice (``source, job_id, job_name, text, output_file``) to the configured router.
    Returns its JSON result (truthy) when it took ownership, else ``None`` (caller falls back to
    its chat delivery). Never raises."""
    label = f"{payload.get('source') or '?'} '{payload.get('job_id') or '?'}'"
    try:
        cmd, timeout = _route_config(_load_config(cfg))
        if not cmd or not str(payload.get("text") or "").strip():
            return None
        proc = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("%s: failure router timed out; delivering the notice to chat instead", label)
        return None
    except (*_CONFIG_ERRORS, subprocess.SubprocessError, ValueError, TypeError):
        # OSError: command missing / not executable; TypeError: payload not JSON-serialisable.
        logger.warning("%s: failure router unavailable; delivering the notice to chat", label,
                       exc_info=True)
        return None
    except Exception:
        # Any other error is a bug here, logged with its traceback. Raising would lose the notice:
        # the callers (cron delivery, the Kanban notifier tick) treat ``None`` as "send it to chat".
        logger.exception("%s: failure router crashed; delivering the notice to chat", label)
        return None
    if proc.returncode != 0:
        logger.warning("%s: failure router exit %s (%s); delivering the notice to chat",
                       label, proc.returncode, (proc.stderr or "").strip()[-300:])
        return None
    result: Any
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {"raw": proc.stdout.strip()[-300:]}
    if not isinstance(result, dict):
        result = {"raw": str(result)}
    result = result or {"routed": True}
    logger.info("%s: failure routed to card %s (%s)", label, result.get("card", "?"),
                result.get("action", "routed"))
    return result


def route_failure(job: dict, content: str, cfg: Optional[dict] = None, *,
                  signature: Optional[str] = None) -> Optional[dict]:
    """``route_event`` for one cron job's failure notice (``source: cron``). Never raises.
    ``signature`` (optional) is the router's dedupe key: repeats of one signature fold into one
    card instead of one card per distinct text."""
    job_id = str(job.get("id") or "")
    payload = {"source": "cron", "job_id": job_id,
               "job_name": str(job.get("name") or job.get("id") or ""),
               "text": content, "output_file": _latest_output(job_id)}
    if signature:
        payload["signature"] = signature
    return route_event(payload, cfg)
