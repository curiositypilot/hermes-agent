"""Fork-patch smoke: ``hermes cron run`` and chat ``/cron run`` route through REAL processes.

Pins fork patches efb1d42436 (``hermes cron run`` hands the run to the live gateway scheduler via
``trigger_job``; ``--inline`` keeps the in-process run; no live scheduler -> warn and run here) and
e07b4c5273 (chat ``/cron run`` routes the same way, ``hermes_cli/cron.py::chat_run_handoff``).
Their unit tests (``tests/hermes_cli/test_cron.py``) mock the ticker, so an upstream move of the
ticker, the liveness probe or the ``/cron`` dispatch would leave ``Queued job`` printed with nothing
ever running, or a run executed by the CLI under a live gateway, with every unit test green.
Regression suite for intake t_ac698836; conventions (real ``hermes`` processes in an isolated
``Home``, warm-gateway rule) are this directory's, see ``test_no_fallback_e2e.py``.

Observable: the executions ledger (``<home>/cron/executions.db``, table ``executions``), read-only.
The process that FIRES a run writes its row: the gateway ticker writes ``source='builtin'`` with
the gateway's pid (``cron/scheduler.py`` tick), an in-process ``run_one_job`` writes
``source='direct'`` with its own pid (``cron/executions.py`` stamps ``os.getpid()``). Which process
runs the agent turn is NOT the observable: even an inline run sends its turn to the gateway daemon
(``cron/scheduler_authority.py::run_canonical_job``). So:

- (a) queue: warm gateway; ``hermes cron run`` prints ``Queued job:`` (no ``Ran now:``) and the
  ledger gets a completed ``builtin`` row stamped with the gateway's pid, not the CLI's, within
  one ticker interval (60 s) plus the turn; the job's output file exists.
- (b) ``--inline`` under a warm gateway: ``Ran now: succeeded.``, no queueing, no
  no-scheduler warning; the row is ``direct`` with the CLI's pid.
- (c) no gateway: the no-live-scheduler warning, ``Ran now:``, a ``direct`` row with the CLI's pid.
- (d) chat ``/cron run`` under a warm gateway, sent as ``slash.exec`` the way the classic CLI's chat
  sends it (``_CHAT_DRIVER``): ``Queued job:`` and the same ledger proof as (a).

Warm gateway = auto-started by a flagless ``hermes -z`` control turn and proven live by
``live_gateway_pid_for_home`` before the run under test. Every case stops the home's gateway in
``finally``; every wait raises ``HarnessError`` on timeout (never xfail, never retry).
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.status import live_gateway_pid_for_home
from tests.e2e.core.providers._openai_helpers import (
    HarnessError,
    Home,
    custom_chat_config,
    hermes_argv,
    wait_until,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = pytest.mark.platforms("linux")

REPLY = "CRON-E2E-OK"
CLI_BUDGET = 150.0
GATEWAY_LIVE_BUDGET = 15.0
# Ticker interval (gateway/run_bootstrap.py SupervisedTickerThread) is 60 s, plus the turn.
QUEUED_RUN_BUDGET = 150.0
QUEUED = "Queued job:"
RAN = "Ran now:"
NO_SCHEDULER = "No live scheduler for this profile; running here."
_JOB_ID = re.compile(r"Created job: (\S+)")

# The classic CLI's chat path for /cron (hermes_cli/gateway_plugin_commands.py::call_unbounded): the
# chat's own connection creates (and so subscribes to) the session, and the command goes out as
# ``slash.exec`` on a second socket of its own (GatewayConnection.call_until). A write-form /cron sent
# on the subscribed viewer connection is refused by design (gateway/session_commands.py
# _OWNER_COMMANDS), and the stdio tui_gateway backend refuses to start under a live gateway, so this
# is the one chat path that runs /cron run in the gateway.
_CHAT_DRIVER = """
import asyncio, json, os, sys, uuid
from hermes_cli.gateway_client import connect_gateway

async def main(command):
    async with connect_gateway() as viewer:
        snapshot = await viewer.rpc("session.create", request_id=uuid.uuid4().hex, source="cli", cwd=os.getcwd())
        async with connect_gateway() as socket:
            reply = await socket.rpc("slash.exec", _timeout=float("inf"), session_id=snapshot["session_id"],
                                     command=command)
    print(json.dumps(reply))

asyncio.run(main(sys.argv[1]))
"""


def _run(h: Home, argv: list[str]) -> tuple[subprocess.CompletedProcess, int]:
    """Run one real process in the home; returns its result and its pid."""
    proc = subprocess.Popen(argv, cwd=h.project, env=h.env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    try:
        out, err = proc.communicate(timeout=CLI_BUDGET)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.communicate()
        raise HarnessError(f"{argv[1:]!r} still running after {CLI_BUDGET}s") from exc
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err), proc.pid


def _hermes(h: Home, *argv: str) -> tuple[subprocess.CompletedProcess, int]:
    return _run(h, hermes_argv(*argv))


def _describe(p: subprocess.CompletedProcess) -> str:
    return f"exit={p.returncode}\nstdout={p.stdout[-1500:]!r}\nstderr={p.stderr[-3000:]}"


def _stop_gateway(h: Home) -> None:
    subprocess.run(hermes_argv("gateway", "stop"), cwd=h.project, env=h.env(), capture_output=True,
                   text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)


def _create_job(h: Home, name: str) -> str:
    created, _ = _hermes(h, "cron", "create", "1d", f"{name}: reply with the canary", "--name", name,
                         "--deliver", "local")
    match = _JOB_ID.search(created.stdout)
    if created.returncode != 0 or not match:
        raise HarnessError(f"hermes cron create failed\n{_describe(created)}")
    return match.group(1)


def _warm_gateway(h: Home) -> int:
    """Auto-start the home's gateway with a flagless control turn; its proven-live pid."""
    control, _ = _hermes(h, "-z", "warm up")
    if control.returncode != 0 or REPLY not in control.stdout:
        raise HarnessError(f"flagless control turn failed (fixture does not bite)\n{_describe(control)}")
    return wait_until(lambda: live_gateway_pid_for_home(h.hermes_home), GATEWAY_LIVE_BUDGET,
                      "a live gateway for the home after the control turn")


def _executions(h: Home, job_id: str) -> list[dict]:
    path = h.hermes_home / "cron" / "executions.db"
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT job_id, source, pid, status FROM executions WHERE job_id=?", (job_id,))
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _outputs(h: Home, job_id: str) -> list[Path]:
    return sorted((h.hermes_home / "cron" / "output" / job_id).glob("*.md"))


def _assert_queued_run_fired_by_gateway(h: Home, job_id: str, gateway_pid: int, caller_pid: int) -> None:
    """The ledger proof for a queued run: a completed ``builtin`` row stamped with the gateway's pid."""
    try:
        row = wait_until(lambda: next((r for r in _executions(h, job_id) if r["status"] == "completed"), None),
                         QUEUED_RUN_BUDGET, f"a completed execution of queued job {job_id}")
    except HarnessError as exc:
        raise HarnessError(f"{exc}; ledger rows: {_executions(h, job_id)}") from exc
    assert row["source"] == "builtin" and row["pid"] == gateway_pid and row["pid"] != caller_pid, (
        f"queued run was not fired by the gateway ticker (gateway pid {gateway_pid}, caller pid "
        f"{caller_pid}): {row}")
    assert _outputs(h, job_id), f"queued run completed but wrote no cron/output/{job_id}/*.md"


def _assert_direct_run_by(h: Home, job_id: str, cli_pid: int, gateway_pid: int | None) -> None:
    rows = _executions(h, job_id)
    assert len(rows) == 1, f"expected exactly one execution of {job_id}: {rows}"
    row = rows[0]
    assert row["source"] == "direct" and row["pid"] == cli_pid and row["status"] == "completed", (
        f"in-process run was not fired by the CLI (pid {cli_pid}): {row}")
    assert row["pid"] != gateway_pid, f"in-process run carries the gateway's pid {gateway_pid}: {row}"


@pytest.fixture
def home(tmp_path):
    with FakeLLMServer(default_text=REPLY) as srv:
        h = Home(tmp_path).write(custom_chat_config(srv.base_url), {"OPENAI_API_KEY": "sk-fake"})
        try:
            yield h
        finally:
            _stop_gateway(h)


def test_cron_run_queues_for_the_warm_gateway_ticker(home: Home) -> None:
    job_id = _create_job(home, "queued")
    gateway_pid = _warm_gateway(home)
    ran, cli_pid = _hermes(home, "cron", "run", job_id)
    assert ran.returncode == 0 and QUEUED in ran.stdout and RAN not in ran.stdout, (
        f"hermes cron run under a live gateway did not queue\n{_describe(ran)}")
    _assert_queued_run_fired_by_gateway(home, job_id, gateway_pid, cli_pid)


def test_cron_run_inline_runs_in_the_cli_under_a_warm_gateway(home: Home) -> None:
    job_id = _create_job(home, "inline")
    gateway_pid = _warm_gateway(home)
    ran, cli_pid = _hermes(home, "cron", "run", job_id, "--inline")
    assert (ran.returncode == 0 and f"{RAN} succeeded." in ran.stdout and QUEUED not in ran.stdout
            and NO_SCHEDULER not in ran.stdout), f"hermes cron run --inline did not run in the CLI\n{_describe(ran)}"
    _assert_direct_run_by(home, job_id, cli_pid, gateway_pid)


def test_cron_run_without_gateway_warns_and_runs_in_the_cli(home: Home) -> None:
    job_id = _create_job(home, "cold")
    assert live_gateway_pid_for_home(home.hermes_home) is None, "cold case: a gateway already serves the home"
    ran, cli_pid = _hermes(home, "cron", "run", job_id)
    assert ran.returncode == 0 and NO_SCHEDULER in ran.stdout and RAN in ran.stdout and QUEUED not in ran.stdout, (
        f"hermes cron run with no live scheduler did not warn and run here\n{_describe(ran)}")
    _assert_direct_run_by(home, job_id, cli_pid, None)


def test_chat_cron_run_queues_for_the_warm_gateway_ticker(home: Home) -> None:
    job_id = _create_job(home, "chat")
    gateway_pid = _warm_gateway(home)
    driven, driver_pid = _run(home, [sys.executable, "-c", _CHAT_DRIVER, f"/cron run {job_id}"])
    if driven.returncode != 0:
        raise HarnessError(f"chat slash.exec driver failed\n{_describe(driven)}")
    reply = json.loads(driven.stdout.strip().splitlines()[-1])
    output = str(reply.get("output", ""))
    assert QUEUED in output and RAN not in output, f"chat /cron run under a live gateway did not queue: {reply!r}"
    _assert_queued_run_fired_by_gateway(home, job_id, gateway_pid, driver_pid)
