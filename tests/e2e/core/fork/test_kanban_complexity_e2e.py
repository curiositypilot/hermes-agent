"""Fork-patch smoke: ``kanban create --complexity S|M`` reaches the owner-run worker's model request.

Regression for intake t_0510731c. Fork routing (``hermes_cli/kanban_routing.py``) picks the
worker model in the dispatcher and records it ONLY on the run's ``routed`` event; the card never
carries it. Since the 2026-10-30 sync the worker is a thin client with no model of its own, and
the profile owner rebuilds the session from the board in ``gateway/session_kanban.py``
``build_kanban_policy``. The routed model reaches that session through one hand-attached fork
hook there: ``replay_recorded_route(conn, task, task.current_run_id)``. The unit tests call
``build_kanban_policy`` by name, so an upstream merge that moves worker execution again can drop
the replay with every unit test green, and routed cards silently run on the profile default.
This file asserts on the request the fake provider actually receives.

Warm-gateway rule (every fork e2e, see ``test_no_fallback_e2e.py``): the routed turns must hit
an owner daemon that was started WITHOUT them. A flagless control card runs first (its worker
auto-starts the owner), the owner is proven live, and only then the S and M cards run.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from gateway.status import live_gateway_pid_for_home
from tests.e2e.core.kanban._helpers import PY, Board, wait_until
from tests.fakes.fake_llm_provider import MODEL_ID, FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs this board's reparented workers
]

TIER_MODELS = {"S": "fake-model-s", "M": "fake-model-m"}
# The harness only drops its own ``dispatch_in_gateway: false`` when ``kanban:`` is absent, so this
# block restates it: the owner's embedded dispatcher would race the explicit ticks below.
ROUTING_CONFIG = (
    "kanban:\n"
    "  dispatch_in_gateway: false\n"
    "  routing:\n"
    "    enabled: true\n"
    "    unlabeled: profile\n"
    "    escalate: false\n"
    "    tiers:\n"
    + "".join(f"      {tier}:\n        - model: {model}\n" for tier, model in TIER_MODELS.items())
)
GATEWAY_LIVE_BUDGET = 15.0

_TASK_RE = re.compile(r"work kanban task \(?(t_[0-9a-f]+)")


def _task_id(body: dict) -> str | None:
    for m in body.get("messages", []):
        hit = _TASK_RE.search(str(m.get("content")))
        if hit:
            return hit.group(1)
    return None


def _completing_responder(rec: dict):
    if rec["body"]["messages"][-1].get("role") == "tool":
        return Text("routed run done")
    return ToolCall("kanban_complete", {"summary": "complexity routing probe"})


def _run_one_card(board: Board, tid: str) -> None:
    board.dispatch()
    pid = board.task(tid)["worker_pid"]
    assert pid, board.diag(tid)  # no worker: a routing hold (routing_held) reads in the events
    board.wait_worker_exit(tid, int(pid))


def _models_for(srv: FakeLLMServer, tid: str) -> list[str | None]:
    return [body.get("model") for body in srv.main_requests() if _task_id(body) == tid]


def _stop_gateway(board: Board) -> None:
    subprocess.run([PY, "-m", "hermes_cli.main", "gateway", "stop"], cwd=str(board.root), env=board.env(),
                   capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)


def test_complexity_tier_model_reaches_warm_owner_worker(tmp_path) -> None:
    with FakeLLMServer(_completing_responder) as srv:
        board = Board(tmp_path, srv.base_url, extra_config=ROUTING_CONFIG)
        try:
            # (a) control: unlabelled -> profile default. Proves the fixture tells models apart
            # and auto-starts the owner daemon without any routed card.
            control = board.create("unlabelled control card")
            _run_one_card(board, control)
            models = _models_for(srv, control)
            assert models and set(models) == {MODEL_ID}, (
                f"control card requests carried {models}, expected only {MODEL_ID}\n{board.diag(control)}")
            assert board.task(control)["status"] == "done", board.diag(control)

            # (b) warm-gateway rule: the routed cards below hit this already-running owner.
            owner = wait_until(lambda: live_gateway_pid_for_home(board.hermes_home), GATEWAY_LIVE_BUDGET,
                               "a live owner daemon for the home after the control card")

            # (c)/(d) each tier's model must be on every main request of its card's worker.
            for tier, model in TIER_MODELS.items():
                tid = board.create(f"tier {tier} card", "--complexity", tier)
                _run_one_card(board, tid)
                routed = board.events(tid, "routed")
                assert routed, f"no routed event for the {tier} card\n{board.diag(tid)}"
                payload = routed[-1]["payload"]
                assert (payload.get("source"), payload.get("tier"), payload.get("model")) == ("tier", tier, model), (
                    f"routed event {payload} does not name tier {tier} -> {model}\n{board.diag(tid)}")
                models = _models_for(srv, tid)
                assert models and set(models) == {model}, (
                    f"{tier} card's worker (owner pid {owner}) requested {models}, expected only {model}: "
                    f"the routed model did not reach the owner-run session (replay_recorded_route)\n"
                    f"{board.diag(tid)}")
                assert board.task(tid)["status"] == "done", board.diag(tid)
        finally:
            board.kill_workers()
            _stop_gateway(board)
