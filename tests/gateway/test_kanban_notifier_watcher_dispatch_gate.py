"""Notifier polling has an independent gateway config gate; dispatch has the ESTOP and restart-hold gate."""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


def _make_runner(with_adapter=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()} if with_adapter else {}
    runner._kanban_sub_fail_counts = {}
    return runner


def test_notifier_watcher_skips_when_notifications_disabled():
    runner = _make_runner(with_adapter=True)

    with patch(
        "hermes_cli.config.load_config",
        return_value={"kanban": {"notify_in_gateway": False}},
    ):
        with patch("hermes_cli.kanban_db.list_boards") as list_boards:
            asyncio.run(runner._kanban_notifier_watcher())

    list_boards.assert_not_called()


def test_notifier_watcher_polls_without_dispatch_ownership():
    """A profile gateway still polls its profile-owned subscriptions."""
    runner = _make_runner(with_adapter=True)
    past_gate = []
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        # Stop after the initial delay + first per-interval sleep so the loop
        # body runs exactly once.
        if len(sleep_calls) >= 2:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import hermes_cli.kanban_db as _kb

    with patch(
        "hermes_cli.config.load_config",
        return_value={
            "kanban": {
                "dispatch_in_gateway": False,
                "notify_in_gateway": True,
            }
        },
    ):
        with patch.object(
            _kb, "list_boards",
            side_effect=lambda *a, **kw: past_gate.append(True) or [],
        ):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with patch("asyncio.to_thread", side_effect=fake_to_thread):
                    asyncio.run(runner._kanban_notifier_watcher())

    assert past_gate, (
        "gateways without the dispatch lock must still poll owned subscriptions"
    )


# ── Dispatch gate: `gateway restart --when-idle` hold (t_50ffb4fd) ───────────────────────────────


@pytest.fixture
def hold(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    from hermes_cli import kanban_restart_hold
    yield kanban_restart_hold
    kanban_restart_hold.release()


def test_dispatch_is_held_while_a_restart_hold_names_this_gateway(hold):
    from gateway.kanban_watchers_common import _kanban_dispatch_allowed

    assert _kanban_dispatch_allowed() is True
    hold.engage(os.getpid(), 60, "gateway restart --when-idle")
    assert _kanban_dispatch_allowed() is False
    hold.release()
    assert _kanban_dispatch_allowed() is True


@pytest.mark.parametrize("pid_offset, ttl", [(0, -1), (1, 60)], ids=["expired", "other-gateway"])
def test_dispatch_ignores_an_expired_hold_or_one_for_another_gateway(hold, pid_offset, ttl):
    from gateway.kanban_watchers_common import _kanban_dispatch_allowed

    hold.engage(os.getpid() + pid_offset, ttl, "not for this gateway")
    assert _kanban_dispatch_allowed() is True


def test_estop_still_pauses_dispatch_without_a_hold(hold):
    from agent import estop
    from gateway.kanban_watchers_common import _kanban_dispatch_allowed

    estop.engage(reason="test")
    try:
        assert _kanban_dispatch_allowed() is False
    finally:
        estop.disengage()
    assert _kanban_dispatch_allowed() is True


def test_hold_engagement_is_logged_once_and_release_once(hold, caplog):
    from gateway.kanban_watchers_common import _kanban_dispatch_allowed

    hold.engage(os.getpid(), 60, "landing t_x")
    with caplog.at_level("INFO", logger="gateway.run"):
        for _ in range(3):
            _kanban_dispatch_allowed()
        hold.release()
        for _ in range(3):
            _kanban_dispatch_allowed()
    held = [r for r in caplog.records if "dispatch held for gateway restart (landing t_x)" in r.getMessage()]
    resumed = [r for r in caplog.records if "dispatch resumed" in r.getMessage()]
    assert len(held) == 1 and len(resumed) == 1
