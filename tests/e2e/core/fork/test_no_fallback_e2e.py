"""Fork-patch flag smoke: ``--no-fallback`` through REAL ``hermes`` processes and a warm gateway.

``tests/e2e/core/fork/`` is the fork-patch CLI flag smoke suite: one file per CLI flag a fork
patch adds, each driving real ``hermes`` processes in an isolated ``Home``. Run it after every
upstream merge (``scripts/run_tests.sh tests/e2e/core/fork/``). A fork patch's unit tests pin
the seam the patch was written against; an upstream merge can move the production path (CLI
-> gateway, in-process -> managed worker) without touching that seam, and only real processes
notice. Regression for intake t_d3009878: upstream bf3d69137d moved ``-z`` and ``chat -Q``
turns into the shared gateway, ``--no-fallback`` still set only the CLI's env, and a pinned dead
model was answered by ``fallback_providers`` while ``tests/hermes_cli/test_no_fallback_flag.py``
stayed green.

Warm-gateway rule (every fork e2e): run the flagged turn against a gateway that was started
WITHOUT the flag's env. A cold start, where the flagged CLI is the first process in the home,
auto-starts the gateway daemon with the CLI's env (``HERMES_NO_FALLBACK=1``), so the flag works
by accident and the bug stays hidden. Production gateways are started by systemd, flagless.
So each case runs a flagless control turn first (it auto-starts the gateway), proves that
gateway is live, and only then runs the flagged turn in the same home. The ``hermes --tui`` case
gets its warm gateway from a flagless TUI A's auto-start; TUI B (``--no-fallback``) reuses it.

Fixture: a primary that answers every request with 404 ``model_not_found`` and a fallback
that answers ``FALLBACK_TEXT``. The control turn must reach the fallback (the fixture bites);
the ``--no-fallback`` turn must exit non-zero without a single new fallback request.
"""

from __future__ import annotations

import subprocess

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
from tests.fakes.providers.chat_variants import CError, FakeChatVariantServer

pytestmark = pytest.mark.platforms("linux")

FALLBACK_TEXT = "FROM-FALLBACK"
PROMPT = "CANARY-PROMPT say hi"
# The cold control turn includes the gateway auto-start; a 404 primary is not retried, so a
# turn overrunning this is a hang (HarnessError), never a pass or an xfail.
TURN_BUDGET = 90.0
GATEWAY_LIVE_BUDGET = 15.0

SURFACES = {
    "z": ["-z", PROMPT],
    "chat_Q": ["chat", "-q", PROMPT, "-Q"],
}


def _turn(h: Home, argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(hermes_argv(*argv), cwd=h.project, env=h.env(), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=TURN_BUDGET,
                              stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"turn {argv!r} still running after {TURN_BUDGET}s") from exc


def _describe(p: subprocess.CompletedProcess) -> str:
    return f"exit={p.returncode}\nstdout={p.stdout[-1500:]!r}\nstderr={p.stderr[-3000:]}"


def _stop_gateway(h: Home) -> None:
    subprocess.run(hermes_argv("gateway", "stop"), cwd=h.project, env=h.env(), capture_output=True,
                   text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)


def _warm_gateway_pid(h: Home) -> int:
    """PID of the live gateway the control turn auto-started. ``live_gateway_pid_for_home``
    proves liveness and identity (start-time reuse guard, a gateway command line owned by this
    home), not merely that ``gateway.pid`` exists. None means turns no longer go through a
    gateway daemon: the warm-gateway premise is gone, which is a harness failure, not a pass."""
    return wait_until(lambda: live_gateway_pid_for_home(h.hermes_home), GATEWAY_LIVE_BUDGET,
                      "a live gateway for the home after the flagless control turn")


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_no_fallback_refuses_fallback_through_warm_gateway(tmp_path, surface: str) -> None:
    argv = SURFACES[surface]
    with FakeChatVariantServer(lambda _r: CError(404, "model not found", code="model_not_found")) as primary, \
            FakeLLMServer(default_text=FALLBACK_TEXT) as fallback:
        cfg = custom_chat_config(primary.base_url)
        cfg["fallback_providers"] = [{"provider": "custom", "model": "fallback-model", "base_url": fallback.base_url}]
        h = Home(tmp_path).write(cfg, {"OPENAI_API_KEY": "sk-fake"})
        try:
            control = _turn(h, argv)
            n_control = len(fallback.main_requests())
            assert control.returncode == 0 and FALLBACK_TEXT in control.stdout and n_control >= 1, (
                f"flagless control turn did not reach the fallback (fixture does not bite)\n{_describe(control)}")
            pid = _warm_gateway_pid(h)
            flagged = _turn(h, [*argv, "--no-fallback"])
            n_new = len(fallback.main_requests()) - n_control
            assert flagged.returncode != 0 and n_new == 0, (
                f"--no-fallback turn through warm gateway pid {pid} was answered by fallback_providers "
                f"(new fallback requests={n_new})\n{_describe(flagged)}")
        finally:
            _stop_gateway(h)


def _turn_settled(text: str) -> bool:
    return "HTTP 404" in text or FALLBACK_TEXT in text


@pytest.mark.live_system_guard_bypass
def test_no_fallback_tui_warm_gateway(tmp_path) -> None:
    """``hermes --tui --no-fallback``: the TUI builds ``no_fallback`` for ``session.create`` in
    TypeScript (``ui-tui/src/canonicalGateway.ts``), a client the ``-z``/``chat`` cases cannot
    reach. Intake t_3b23f827; it fails at 702ebed4d7, before merge e2582abe3b (t_044478d3)."""
    # tui_pty's helpers import termios at module level; keep them out of this module's import.
    from tests.e2e.core.tui_pty._helpers import TmuxTui, private_tui_dir, require_tui

    require_tui()
    tui_dir = private_tui_dir(tmp_path)
    with FakeChatVariantServer(lambda _r: CError(404, "model not found", code="model_not_found")) as primary, \
            FakeLLMServer(default_text=FALLBACK_TEXT) as fallback:
        fb = (f"fallback_providers:\n  - provider: custom\n    model: fallback-model\n"
              f"    base_url: {fallback.base_url}\n")
        a = TmuxTui(tmp_path / "a", primary.base_url, extra_config=fb, tui_dir=tui_dir)
        b = None
        try:
            # (a) flagless TUI A: the fixture bites, and A's auto-start is the warm gateway.
            a.wait_ready()
            a.submit("CANARY-PROMPT say hi")
            a.wait_for(FALLBACK_TEXT, timeout=TURN_BUDGET)
            n_control = len(fallback.main_requests())
            assert n_control >= 1, f"flagless TUI turn did not reach the fallback\n{a.dump()}"
            assert (problem := a.exit_problem()) == "", problem
            # (b) the gateway A started outlives A.
            pid = wait_until(lambda: live_gateway_pid_for_home(a.hermes_home), GATEWAY_LIVE_BUDGET,
                             "a live gateway for the home after the flagless TUI exited")
            # (c) TUI B with --no-fallback on the same home and gateway.
            p_control = len(primary.main_requests())
            b = TmuxTui(tmp_path / "b", primary.base_url, args=("--yolo", "--no-fallback"),
                        env_extra={"HOME": str(a.home), "HERMES_HOME": str(a.hermes_home)},
                        write_home=False, tui_dir=tui_dir)
            b.wait_ready()
            b.submit("CANARY-PROMPT say hi again")
            try:
                b.wait_for(_turn_settled, timeout=TURN_BUDGET)
                b.wait_quiet(2.0, timeout=60)
            except AssertionError as exc:
                raise HarnessError(f"--no-fallback TUI turn never settled: {exc}") from exc
            n_new = len(fallback.main_requests()) - n_control
            screen = b.text()
            assert len(primary.main_requests()) > p_control, f"the TUI turn never reached the primary\n{b.dump()}"
            assert n_new == 0 and FALLBACK_TEXT not in screen, (
                f"--no-fallback TUI turn through warm gateway pid {pid} was answered by fallback_providers "
                f"(new fallback requests={n_new})\n{b.dump()}")
            assert "HTTP 404" in screen, f"the primary's 404 is not on screen\n{b.dump()}"
            assert live_gateway_pid_for_home(a.hermes_home) == pid, "TUI B did not run on A's warm gateway"
        finally:
            if b is not None:
                b.close()
            a.close()
