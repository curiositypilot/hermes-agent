"""Fork-patch smoke: ``hermes gateway restart --when-idle`` against a REAL gateway in a fixture Home.

Fork patch 40053d6584 (intake t_50ffb4fd) adds ``--when-idle``: engage the pid-keyed Kanban
dispatch hold (``restart_hold.json`` next to the board DB, ``hermes_cli/kanban_restart_hold.py``),
poll the target gateway's live work, then run the normal drain-aware restart and release the hold.
Its unit tests mock the gateway. The fork-landing retry units and the upstream-sync cutover call
this path, so an upstream change to the live-work source (control socket ``active-work``,
``gateway_state.json``), the restart path or ``gateway.pid`` could make it restart mid-turn, never
restart, or leave the hold behind while those unit tests stay green. Regression for intake
t_01f7e2a1.

Warm-gateway rule (every fork e2e, see ``test_no_fallback_e2e.py``): a flagless ``-z`` turn
auto-starts the gateway first, and each case acts on that live gateway.

Fixture: a fake provider whose reply blocks on a ``threading.Event`` when the user message carries
``SLOW_MARKER``, so the test decides when the live turn ends. Three cases:

- idle: the restart exits 0, a new gateway pid comes up, no hold file is left, a turn still answers.
- busy: the hold names the old pid; the restart is still waiting after more than one poll; once the
  turn ends the restart exits 0, the pid changes and the hold is gone.
- timeout: the hold names the old pid; the restart exits 1 with the gateway untouched and the hold
  released; the held turn still finishes.

Not covered here: the dispatcher actually skipping a ready task while held (needs a board and a
dispatch tick in the Home; unit-tested via ``_kanban_dispatch_allowed``; t_7efd5009#uf1).
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gateway.status import live_gateway_pid_for_home
from hermes_cli.gateway_restart_when_idle import POLL_S
from hermes_cli.kanban_restart_hold import HOLD_FILE
from tests.e2e.core.providers._openai_helpers import (
    HarnessError,
    Home,
    custom_chat_config,
    hermes_argv,
    wait_until,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.waiting import wait_until as poll_until

# spawns_gateway_lookalike: each case auto-starts a gateway daemon in its own fixture Home and the
# restart spawns its detached replacement there; ``_warm_rig`` stops it (``gateway stop --yes``).
pytestmark = [pytest.mark.platforms("linux"), pytest.mark.spawns_gateway_lookalike]

SLOW_MARKER = "SLOWTURN"
FAST_TEXT = "FAST-OK"
SLOW_TEXT = "SLOW-DONE"
TURN_BUDGET = 90.0  # includes the cold gateway auto-start on the warm-up turn
GATEWAY_LIVE_BUDGET = 15.0
HOLD_BUDGET = 30.0
NEW_PID_BUDGET = 60.0
# After the turn ends: at most one POLL_S sleep, then the graceful restart.
RESTART_EXIT_BUDGET = 120.0
GATE_BUDGET = 180.0


def _last_user_text(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


class _Proc:
    """A real ``hermes`` process writing to files, so a failure can quote its output while it runs."""

    def __init__(self, h: Home, logs: Path, name: str, argv: list[str]) -> None:
        self.argv = argv
        self.out, self.err = logs / f"{name}.out", logs / f"{name}.err"
        with self.out.open("w", encoding="utf-8") as out, self.err.open("w", encoding="utf-8") as err:
            self.p = subprocess.Popen(hermes_argv(*argv), cwd=h.project, env=h.env(), stdout=out, stderr=err,
                                      stdin=subprocess.DEVNULL)

    @property
    def stdout(self) -> str:
        return self.out.read_text(encoding="utf-8", errors="replace")

    def describe(self) -> str:
        err = self.err.read_text(encoding="utf-8", errors="replace")
        return f"{self.argv!r} exit={self.p.poll()}\nstdout={self.stdout[-1500:]!r}\nstderr={err[-3000:]}"

    def finish(self, timeout: float) -> int:
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(f"still running after {timeout}s: {self.describe()}") from exc

    def exits_within(self, timeout: float) -> bool:
        try:
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True


@dataclass
class _Rig:
    h: Home
    gate: threading.Event
    slow_seen: threading.Event
    old_pid: int
    logs: Path
    procs: list[_Proc] = field(default_factory=list)

    @property
    def hold(self) -> Path:
        return self.h.hermes_home / HOLD_FILE

    def spawn(self, name: str, *argv: str) -> _Proc:
        proc = _Proc(self.h, self.logs, name, list(argv))
        self.procs.append(proc)
        return proc

    def turn(self, name: str, prompt: str) -> _Proc:
        return self.spawn(name, "-z", prompt)

    def restart(self, idle_timeout: int) -> _Proc:
        return self.spawn("restart", "gateway", "restart", "--when-idle", "--idle-timeout", str(idle_timeout),
                          "--detach")

    def live_pid(self) -> int | None:
        return live_gateway_pid_for_home(self.h.hermes_home)

    def read_hold(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.hold.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def start_slow_turn(self) -> _Proc:
        slow = self.turn("slow", f"{SLOW_MARKER} please")
        if not self.slow_seen.wait(TURN_BUDGET):
            raise HarnessError(f"the slow turn never reached the provider: {slow.describe()}")
        return slow

    def await_hold(self, restart: _Proc) -> dict[str, Any]:
        """The hold record, read while the restart still runs. A restart that exits first (e.g. refused
        because it did not wait for the live turn) is named as such, not as a missing hold."""
        poll_until(lambda: self.read_hold() is not None or restart.p.poll() is not None, timeout=HOLD_BUDGET,
                   interval=0.05, what="the hold file or the restart exiting")
        record = self.read_hold()
        assert record is not None, f"restart exited without holding while a turn was live\n{restart.describe()}"
        assert record.get("gateway_pid") == self.old_pid, (
            f"hold names {record.get('gateway_pid')}, gateway pid is {self.old_pid}\n{restart.describe()}")
        return record

    def new_pid(self) -> int | None:
        try:
            return poll_until(lambda: (pid := self.live_pid()) and pid != self.old_pid and pid, timeout=NEW_PID_BUDGET,
                              interval=0.1, what="a replacement gateway pid")
        except AssertionError:
            return None


@contextmanager
def _warm_rig(tmp_path: Path) -> Iterator[_Rig]:
    gate, slow_seen = threading.Event(), threading.Event()

    def responder(req: dict[str, Any]) -> Text:
        if SLOW_MARKER in _last_user_text(req.get("body") or {}):
            slow_seen.set()
            gate.wait(GATE_BUDGET)
            return Text(SLOW_TEXT)
        return Text(FAST_TEXT)

    with FakeLLMServer(responder) as fake:
        h = Home(tmp_path).write(custom_chat_config(fake.base_url), {"OPENAI_API_KEY": "sk-fake"})
        logs = tmp_path / "logs"
        logs.mkdir()
        rig: _Rig | None = None
        try:
            warm = _Proc(h, logs, "warm", ["-z", "warm up"])
            if warm.finish(TURN_BUDGET) != 0 or FAST_TEXT not in warm.stdout:
                raise HarnessError(f"warm-up turn failed: {warm.describe()}")
            old = wait_until(lambda: live_gateway_pid_for_home(h.hermes_home), GATEWAY_LIVE_BUDGET,
                             "a live gateway for the home after the warm-up turn")
            rig = _Rig(h, gate, slow_seen, old, logs)
            yield rig
        finally:
            # A restart CLI left waiting would restart the gateway after the stop below: kill it first.
            for proc in rig.procs if rig else []:
                if proc.p.poll() is None:
                    proc.p.kill()
                    proc.p.wait(timeout=30)
            gate.set()
            # --yes: stop even with a turn still live; it targets the Home, so it also stops a replacement.
            subprocess.run(hermes_argv("gateway", "stop", "--yes"), cwd=h.project, env=h.env(), capture_output=True,
                           text=True, timeout=60, stdin=subprocess.DEVNULL, check=False)


def test_when_idle_restarts_an_idle_gateway(tmp_path: Path) -> None:
    with _warm_rig(tmp_path) as rig:
        restart = rig.restart(idle_timeout=30)
        assert restart.finish(RESTART_EXIT_BUDGET) == 0, restart.describe()
        new = rig.new_pid()
        assert new, f"gateway pid still {rig.live_pid()} (was {rig.old_pid})\n{restart.describe()}"
        assert not rig.hold.exists(), f"hold left behind: {rig.read_hold()}\n{restart.describe()}"
        after = rig.turn("after", "after restart")
        assert after.finish(TURN_BUDGET) == 0 and FAST_TEXT in after.stdout, after.describe()


def test_when_idle_waits_for_the_live_turn_then_restarts(tmp_path: Path) -> None:
    with _warm_rig(tmp_path) as rig:
        slow = rig.start_slow_turn()
        restart = rig.restart(idle_timeout=120)
        rig.await_hold(restart)
        # Longer than one POLL_S sleep: the restart re-polled and saw the turn still live.
        assert not restart.exits_within(POLL_S + 2), f"restart returned while the turn was live\n{restart.describe()}"
        assert rig.live_pid() == rig.old_pid, f"gateway replaced mid-turn\n{restart.describe()}"
        rig.gate.set()
        assert slow.finish(TURN_BUDGET) == 0 and SLOW_TEXT in slow.stdout, slow.describe()
        assert restart.finish(RESTART_EXIT_BUDGET) == 0, restart.describe()
        new = rig.new_pid()
        assert new, f"gateway pid still {rig.live_pid()} (was {rig.old_pid})\n{restart.describe()}"
        assert not rig.hold.exists(), f"hold left behind: {rig.read_hold()}\n{restart.describe()}"


def test_when_idle_timeout_exits_1_and_releases_the_hold(tmp_path: Path) -> None:
    with _warm_rig(tmp_path) as rig:
        slow = rig.start_slow_turn()
        restart = rig.restart(idle_timeout=3)
        rig.await_hold(restart)
        assert restart.finish(RESTART_EXIT_BUDGET) == 1, restart.describe()
        assert not rig.hold.exists(), f"hold left behind after the timeout: {rig.read_hold()}\n{restart.describe()}"
        assert rig.live_pid() == rig.old_pid, f"gateway restarted despite the timeout\n{restart.describe()}"
        rig.gate.set()
        assert slow.finish(TURN_BUDGET) == 0 and SLOW_TEXT in slow.stdout, slow.describe()
