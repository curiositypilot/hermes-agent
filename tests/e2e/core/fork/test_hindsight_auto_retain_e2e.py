"""Fork-patch env smoke: ``HINDSIGHT_AUTO_RETAIN=0`` through REAL ``hermes`` processes and a warm gateway.

Regression for intake t_248c3f9e: since upstream bf3d69137d a ``-z`` / ``chat -Q`` turn runs in the
shared gateway, whose Hindsight provider read the kill switch from its own process env only, while
the CLI's ``client_env`` allowlist dropped the name. ``HINDSIGHT_AUTO_RETAIN=0 hermes chat -Q``
(the idle model-eval prefix) still wrote a document into bank ``main``, and
``test_kill_switch_disables_auto_retain_but_not_the_tool`` stayed green because it sets the variable
in the provider's own process. Same class as t_d3009878 (``--no-fallback``).

Warm-gateway rule (docstring of ``test_no_fallback_e2e.py``): the switched turn must run against a
gateway started WITHOUT the switch. A cold start would auto-start the daemon with the CLI's env and
the switch would work by accident. So a flagless control turn runs first (it auto-starts the
gateway and proves the fixture bites: its canary reaches the fake Hindsight retain endpoint), the
gateway is proven live, and only then the switched turn runs in the same home. A second flagless
turn bounds the wait without a sleep: once its retain arrives, the switched turn's would have too.

``test_skip_sources_drop_oneshot_turns_on_a_warm_gateway`` (intake t_4d646807) applies the same rule to
``auto_retain_skip_sources: [oneshot, tool]``: ``chat -Q -q`` and ``-z`` with no switch are stored as
source ``oneshot`` and must not reach Hindsight; ``--source cli`` controls before and after must.

Fixture: ``FakeLLMServer`` as the model, ``FakeHindsightServer`` as the memory API (cloud mode,
``retain_contexts: [primary]``, ``retain_async: false``, ``auto_recall: false`` so a recall query
cannot carry the canary either). Assert: no recorded request carries the switched turn's canary.
"""

from __future__ import annotations

import json
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
from tests.fakes.fake_hindsight_server import FakeHindsightServer
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = pytest.mark.platforms("linux")

REPLY = "PONG-REPLY"
# The cold control turn includes the gateway auto-start; overrunning it is a hang (HarnessError).
TURN_BUDGET = 90.0
GATEWAY_LIVE_BUDGET = 15.0
RETAIN_BUDGET = 30.0

SURFACES = {
    "z": lambda prompt: ["-z", prompt],
    "chat_Q": lambda prompt: ["chat", "-q", prompt, "-Q"],
}


def _turn(h: Home, argv: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(hermes_argv(*argv), cwd=h.project, env=h.env(extra_env), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=TURN_BUDGET,
                              stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"turn {argv!r} still running after {TURN_BUDGET}s") from exc


def _describe(p: subprocess.CompletedProcess) -> str:
    return f"exit={p.returncode}\nstdout={p.stdout[-1500:]!r}\nstderr={p.stderr[-3000:]}"


def _stop_gateway(h: Home) -> None:
    subprocess.run(hermes_argv("gateway", "stop"), cwd=h.project, env=h.env(), capture_output=True,
                   text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)


def _home(tmp_path, llm: FakeLLMServer, hs: FakeHindsightServer, **hindsight_cfg) -> Home:
    cfg = custom_chat_config(llm.base_url)
    cfg["memory"] = {"provider": "hindsight"}
    h = Home(tmp_path).write(cfg, {"OPENAI_API_KEY": "sk-fake"})
    hindsight = h.hermes_home / "hindsight"
    hindsight.mkdir(parents=True, exist_ok=True)
    (hindsight / "config.json").write_text(json.dumps({
        "mode": "cloud", "api_url": hs.base_url, "apiKey": "hs-fake", "bank_id": "e2e-bank",
        "retain_contexts": ["primary"], "retain_async": False, "auto_recall": False,
        "retain_indicator": False, **hindsight_cfg,
    }), encoding="utf-8")
    return h


def _retained(hs: FakeHindsightServer, canary: str, after: str) -> list[dict]:
    return [r for r in hs.retain_requests() if canary in r["raw"]] or wait_until(
        lambda: [r for r in hs.retain_requests() if canary in r["raw"]], RETAIN_BUDGET,
        f"a retain request carrying {canary!r} ({after})")


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_auto_retain_switch_reaches_warm_gateway(tmp_path, surface: str) -> None:
    argv = SURFACES[surface]
    with FakeLLMServer(default_text=REPLY) as llm, FakeHindsightServer() as hs:
        h = _home(tmp_path, llm, hs)
        try:
            # (a) flagless control: auto-starts the gateway and proves the fixture bites.
            control = _turn(h, argv("CTRL-A say pong"))
            assert control.returncode == 0 and REPLY in control.stdout, (
                f"flagless control turn failed\n{_describe(control)}")
            _retained(hs, "CTRL-A", f"the flagless control turn; fixture does not bite\n{_describe(control)}")
            pid = wait_until(lambda: live_gateway_pid_for_home(h.hermes_home), GATEWAY_LIVE_BUDGET,
                             "a live gateway for the home after the flagless control turn")
            # (b) the switch in the CLI's env only.
            switched = _turn(h, argv("OFF-B say pong"), {"HINDSIGHT_AUTO_RETAIN": "0"})
            assert switched.returncode == 0 and REPLY in switched.stdout, (
                f"switched turn failed\n{_describe(switched)}")
            # (c) a second flagless turn bounds the wait for any late retain of (b).
            after = _turn(h, argv("CTRL-C say pong"))
            assert after.returncode == 0, f"second control turn failed\n{_describe(after)}"
            _retained(hs, "CTRL-C", f"the second flagless turn\n{_describe(after)}")
            assert live_gateway_pid_for_home(h.hermes_home) == pid, "the turns did not share the warm gateway"
            leaked = hs.requests_containing("OFF-B")
            assert not leaked, (
                f"HINDSIGHT_AUTO_RETAIN=0 turn through warm gateway pid {pid} reached Hindsight: "
                f"{[(r['method'], r['path']) for r in leaked]}")
        finally:
            _stop_gateway(h)


def test_skip_sources_drop_oneshot_turns_on_a_warm_gateway(tmp_path) -> None:
    with FakeLLMServer(default_text=REPLY) as llm, FakeHindsightServer() as hs:
        h = _home(tmp_path, llm, hs, auto_retain_skip_sources=["oneshot", "tool"])
        try:
            # (a) explicit --source cli control: auto-starts the gateway and proves the fixture bites.
            control = _turn(h, ["chat", "-Q", "-q", "CLI-A say pong", "--source", "cli"])
            assert control.returncode == 0 and REPLY in control.stdout, f"cli control failed\n{_describe(control)}"
            _retained(hs, "CLI-A", f"the --source cli control turn; fixture does not bite\n{_describe(control)}")
            pid = wait_until(lambda: live_gateway_pid_for_home(h.hermes_home), GATEWAY_LIVE_BUDGET,
                             "a live gateway for the home after the cli control turn")
            # (b) agent-run one-shots with no switch: stored as source oneshot.
            for canary, argv in (("ONE-B", SURFACES["chat_Q"]), ("ONE-Z", SURFACES["z"])):
                oneshot = _turn(h, argv(f"{canary} say pong"))
                assert oneshot.returncode == 0 and REPLY in oneshot.stdout, (
                    f"{canary} turn failed\n{_describe(oneshot)}")
            # (c) a second cli control bounds the wait for any late retain of (b).
            after = _turn(h, ["chat", "-Q", "-q", "CLI-C say pong", "--source", "cli"])
            assert after.returncode == 0, f"second cli control failed\n{_describe(after)}"
            _retained(hs, "CLI-C", f"the second cli control turn\n{_describe(after)}")
            assert live_gateway_pid_for_home(h.hermes_home) == pid, "the turns did not share the warm gateway"
            leaked = {c: hs.requests_containing(c) for c in ("ONE-B", "ONE-Z") if hs.requests_containing(c)}
            assert not leaked, (
                f"oneshot turns through warm gateway pid {pid} reached Hindsight despite auto_retain_skip_sources: "
                f"{ {c: [(r['method'], r['path']) for r in rs] for c, rs in leaked.items()} }")
        finally:
            _stop_gateway(h)
