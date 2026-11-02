"""Fork-patch smoke: ``--data-class`` (Kanban card, cron job) provider policy through a real turn.

Regression for intake t_a0cb6cce. The class is enforced inside the turn: ``agent/agent_init_fallback.py``
``_init_fallback_chain`` drops every fallback entry the class refuses (and records a
``provider_denied`` card event), ``agent/chat_completion_helpers.py`` re-checks at activation. Two
seams carry the class into that turn, and the unit tests call neither through a real process:

* Kanban: the turn runs in the profile owner's managed worker, and ``gateway/session_kanban.py``
  ``bind_worker_context`` binds the card's class there from its task row. Before that, the worker
  inherited the owner daemon's env, and a daemon cold-started by an ``internal`` card's client kept
  ``HERMES_DATA_CLASS=internal`` (``DAEMON_STRIPPED_ENV`` did not list it), which wins over the row:
  a later ``confidential`` card in that daemon fell back cross-vendor.
* Cron: ``cron/scheduler.py`` ``_CronRunScope`` binds the job's class as a ContextVar inside the
  daemon's ``run_job``.

Warm-gateway rule (every fork e2e, see ``test_no_fallback_e2e.py``): the classed turn must hit a
daemon that was started WITHOUT it. A flagless control (a card or ``-z`` turn) runs first, reaches
the fallback (the fixture bites) and auto-starts the daemon; the daemon is proven live, and only
then the ``confidential`` card or job runs in the same home.

Fixture: a primary that answers every request with 404 ``model_not_found``, one ``custom``
fallback, and a ``confidential`` class that allows ``custom`` but sets ``fallback: false``; primary
and fallback share the provider, so the fallback switch is the only knob under test.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from gateway.status import live_gateway_pid_for_home
from tests.e2e.core.kanban._helpers import PY, Board
from tests.e2e.core.kanban._helpers import wait_until as board_wait_until
from tests.e2e.core.providers._openai_helpers import (
    HarnessError,
    Home,
    custom_chat_config,
    hermes_argv,
    wait_until,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.fakes.providers.chat_variants import CError, FakeChatVariantServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs the board's reparented workers
]

FALLBACK_TEXT = "FROM-FALLBACK"
# A 404 primary is not retried, so a turn overrunning this is a hang (HarnessError), never a pass.
TURN_BUDGET = 120.0
GATEWAY_LIVE_BUDGET = 15.0

POLICIES = {
    "default": "internal",
    "classes": {
        "internal": {"providers": "any", "fallback": True, "auxiliary": "any"},
        "confidential": {"providers": ["custom"], "fallback": False, "auxiliary": "same_provider_or_fail"},
    },
}

_TASK_RE = re.compile(r"work kanban task \(?(t_[0-9a-f]+)")
_JOB_RE = re.compile(r"Created job: (\S+)")


def _dead_primary() -> FakeChatVariantServer:
    return FakeChatVariantServer(lambda _r: CError(404, "model not found", code="model_not_found"))


def _fallback_entry(base_url: str) -> dict:
    return {"provider": "custom", "model": "fallback-model", "base_url": base_url}


# Kanban ----------------------------------------------------------------------------------------


def _board_config(fallback_url: str) -> str:
    """Board ``extra_config``. ``fallback_providers`` is a root key, so it is appended as its own
    block and the harness's ``model:`` section stays the only one (no duplicate key). The harness
    drops its ``dispatch_in_gateway: false`` once ``kanban:`` is present, so this block restates it:
    the owner's embedded dispatcher would race the explicit ticks below."""
    internal, confidential = POLICIES["classes"]["internal"], POLICIES["classes"]["confidential"]
    return (
        "kanban:\n"
        "  dispatch_in_gateway: false\n"
        "  data_policies:\n"
        f"    default: {POLICIES['default']}\n"
        "    classes:\n"
        "      internal:\n"
        f"        providers: {internal['providers']}\n"
        "        fallback: true\n"
        f"        auxiliary: {internal['auxiliary']}\n"
        "      confidential:\n"
        f"        providers: [{', '.join(confidential['providers'])}]\n"
        "        fallback: false\n"
        f"        auxiliary: {confidential['auxiliary']}\n"
        "fallback_providers:\n"
        "  - provider: custom\n"
        "    model: fallback-model\n"
        f"    base_url: {fallback_url}\n"
    )


def _completing_responder(rec: dict):
    if rec["body"]["messages"][-1].get("role") == "tool":
        return Text(FALLBACK_TEXT)
    return ToolCall("kanban_complete", {"summary": "data class probe"})


def _task_id(body: dict) -> str | None:
    for m in body.get("messages", []):
        hit = _TASK_RE.search(str(m.get("content")))
        if hit:
            return hit.group(1)
    return None


def _tagged(srv, tid: str) -> int:
    return sum(1 for body in srv.main_requests() if _task_id(body) == tid)


def _run_one_card(board: Board, tid: str) -> None:
    board.dispatch()
    pid = board.task(tid)["worker_pid"]
    assert pid, board.diag(tid)
    board.wait_worker_exit(tid, int(pid), timeout=TURN_BUDGET)


def _stop_board_gateway(board: Board) -> None:
    subprocess.run([PY, "-m", "hermes_cli.main", "gateway", "stop"], cwd=str(board.root), env=board.env(),
                   capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)


def test_kanban_card_data_class_refuses_fallback_through_warm_owner(tmp_path) -> None:
    with _dead_primary() as primary, FakeLLMServer(_completing_responder) as fallback:
        board = Board(tmp_path, primary.base_url, extra_config=_board_config(fallback.base_url))
        try:
            # (a) flagless control card: resolves `internal`, falls back, and its client cold-starts
            # the owner daemon (with HERMES_DATA_CLASS=internal in the client's env).
            control = board.create("data class control card")
            _run_one_card(board, control)
            assert _tagged(fallback, control) >= 1 and board.task(control)["status"] == "done", (
                f"control card did not complete on the fallback (fixture does not bite)\n{board.diag(control)}")

            # (b) warm-gateway rule: the classed card below hits this already-running owner.
            owner = board_wait_until(lambda: live_gateway_pid_for_home(board.hermes_home), GATEWAY_LIVE_BUDGET,
                                     "a live owner daemon for the home after the control card")

            # (c) confidential card in the warm owner: primary only, the fallback refused and recorded.
            tid = board.create("confidential card", "--data-class", "confidential")
            _run_one_card(board, tid)
            board.dispatch("--max", "0")  # reap the exited worker so its run is booked
            n_fallback = _tagged(fallback, tid)
            assert n_fallback == 0, (
                f"confidential card in warm owner pid {owner} was answered by fallback_providers "
                f"({n_fallback} fallback requests): the card's class did not reach the owner-run worker\n"
                f"{board.diag(tid)}")
            assert _tagged(primary, tid) >= 1, f"confidential card never reached the primary\n{board.diag(tid)}"
            assert board.task(tid)["status"] != "done", board.diag(tid)
            assert all(r["outcome"] != "completed" for r in board.runs(tid)), board.diag(tid)
            denied = [e["payload"] for e in board.events(tid, "provider_denied")]
            assert any(p.get("data_class") == "confidential" and p.get("phase") == "fallback" for p in denied), (
                f"no provider_denied(confidential, fallback) event: {denied}\n{board.diag(tid)}")
        finally:
            board.kill_workers()
            _stop_board_gateway(board)


# Cron ------------------------------------------------------------------------------------------


def _run(h: Home, *argv: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(hermes_argv(*argv), cwd=h.project, env=h.env(), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=TURN_BUDGET,
                              stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"hermes {argv!r} still running after {TURN_BUDGET}s") from exc


def _describe(p: subprocess.CompletedProcess) -> str:
    return f"exit={p.returncode}\nstdout={p.stdout[-1500:]!r}\nstderr={p.stderr[-3000:]}"


def _with_canary(srv, canary: str) -> int:
    return sum(1 for body in srv.main_requests() if canary in str(body.get("messages")))


def _cron_job(h: Home, canary: str, name: str, *extra: str) -> str:
    created = _run(h, "cron", "create", "1d", canary, "--name", name, "--deliver", "local", *extra)
    hit = _JOB_RE.search(created.stdout)
    assert created.returncode == 0 and hit, f"cron create {name} failed\n{_describe(created)}"
    return hit.group(1)


def test_cron_job_data_class_refuses_fallback_through_warm_gateway(tmp_path) -> None:
    with _dead_primary() as primary, FakeLLMServer(default_text=FALLBACK_TEXT) as fallback:
        cfg = custom_chat_config(primary.base_url)
        cfg["fallback_providers"] = [_fallback_entry(fallback.base_url)]
        cfg["kanban"] = {"data_policies": POLICIES}
        h = Home(tmp_path).write(cfg, {"OPENAI_API_KEY": "sk-fake"})
        try:
            # (a) flagless control turn: the fixture bites and the turn auto-starts the gateway.
            control = _run(h, "-z", "CANARY-PROMPT say hi")
            assert control.returncode == 0 and FALLBACK_TEXT in control.stdout, (
                f"flagless control turn did not reach the fallback (fixture does not bite)\n{_describe(control)}")
            pid = wait_until(lambda: live_gateway_pid_for_home(h.hermes_home), GATEWAY_LIVE_BUDGET,
                             "a live gateway for the home after the flagless control turn")

            # (b) classless cron job: still reaches the fallback through the warm gateway.
            canary_a = "CRON-CANARY-A-7c1e say hi"
            job_a = _cron_job(h, canary_a, "dc-control")
            ran_a = _run(h, "cron", "run", job_a, "--inline")
            assert ran_a.returncode == 0 and _with_canary(fallback, canary_a) >= 1, (
                f"classless cron job did not reach the fallback\n{_describe(ran_a)}")

            # (c) confidential cron job: primary only, no fallback, and the run is not a success.
            canary_b = "CRON-CANARY-B-9d2f say hi"
            job_b = _cron_job(h, canary_b, "dc-confidential", "--data-class", "confidential")
            ran_b = _run(h, "cron", "run", job_b, "--inline")
            n_fallback = _with_canary(fallback, canary_b)
            assert n_fallback == 0, (
                f"confidential cron job through warm gateway pid {pid} was answered by fallback_providers "
                f"({n_fallback} fallback requests)\n{_describe(ran_b)}")
            assert _with_canary(primary, canary_b) >= 1, f"confidential job never reached the primary\n{_describe(ran_b)}"
            assert ran_b.returncode != 0 and "succeeded" not in ran_b.stdout, (
                f"confidential cron job was reported succeeded with its primary dead\n{_describe(ran_b)}")
            assert live_gateway_pid_for_home(h.hermes_home) == pid, "the warm gateway was replaced during the cron runs"
        finally:
            subprocess.run(hermes_argv("gateway", "stop"), cwd=h.project, env=h.env(), capture_output=True,
                           text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)
