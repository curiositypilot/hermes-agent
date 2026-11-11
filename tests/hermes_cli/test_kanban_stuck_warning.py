"""The "dispatcher stuck" probes skip cards the quota gate holds with a retry ahead.

Regression for intake t_7933f77e: the gateway logged "kanban dispatcher stuck ...
Last tick held back: quota_hold (1)" every 5 minutes for days while the only ready
card (t_94a52e54) was held on purpose by the quota gate until 2026-11-14 00:08 UTC.
``has_spawnable_ready`` counted that card as work the dispatcher failed to spawn.
Only a ``quota_held`` entry with a future ``retry_at`` is exempt; other holds still warn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_routing as kr
from hermes_cli.kanban_db_boards import create_board
from hermes_cli.kanban_db_dispatch_holds import quota_held_ids

# The two logged warnings (errors.log 13:23:36 and 13:28:39 on 2026-11-11) and the
# hold's retry time from routing_held event 151888 (2026-11-14 00:08:09 UTC).
WARNED_AT = (1794403416, 1794403719)
RETRY_AT = 1794614889
RESETS_AT = "2026-11-14T00:08:09+00:00"

TIERS = {
    "S": [{"model": "claude-s", "provider": "anthropic"}],
}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kbc.connect()
    yield c
    c.close()


@pytest.fixture
def quota_gate_closed(tmp_path, monkeypatch):
    """Anthropic headroom 0%: a P0 card needs P3 and waits until the window resets."""
    path = tmp_path / "headroom.json"
    entry = {"ok": True, "binding": {"name": "session", "headroom": 0.0, "resets_at": RESETS_AT},
             "windows": []}
    path.write_text(json.dumps({"schema": 2, "written_at_epoch": time.time() - 60,
                                "providers": {"anthropic": entry}}))
    cfg = kr.load_routing_config({"routing": {
        "enabled": True, "tiers": TIERS, "escalate": False,
        "review": [{"model": "rev", "provider": "anthropic"}],
        "quota_gate": {"enabled": True, "headroom_file": str(path)},
    }})
    monkeypatch.setattr(kr, "load_routing_config", lambda kanban_cfg=None: cfg)
    monkeypatch.setattr(kr, "pool_availability",
                        lambda provider, model, *, now=None: kr.Availability(True, "pool ok"))


def _spawn(task, workspace, board=None):
    raise AssertionError(f"quota-held card {task.id} was spawned")


def _held_card(conn, *, status="ready") -> str:
    tid = kb.create_task(conn, title="held", assignee="w", complexity="S", priority=0)
    if status != "ready":
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
        conn.commit()
    return tid


# --- replay of t_94a52e54 ---------------------------------------------------------


def test_quota_held_only_card_is_not_stuck(conn, quota_gate_closed, all_assignees_spawnable):
    held = _held_card(conn)
    res = kbd.dispatch_once(conn, spawn_fn=_spawn)

    assert res.quota_held and res.quota_held[0][0] == held
    assert res.quota_held[0][2] == RETRY_AT
    assert res.spawned == []
    # Old logic: a pending spawnable card and nothing spawned -> a bad tick.
    assert kbd.has_spawnable_ready(conn) is True
    for now in WARNED_AT:
        assert kbd.has_spawnable_ready(conn, skip=quota_held_ids([res], now)) is False


def test_second_ready_card_not_held_is_still_pending(conn, quota_gate_closed, all_assignees_spawnable):
    _held_card(conn)
    res = kbd.dispatch_once(conn, spawn_fn=_spawn)
    held = quota_held_ids([res], WARNED_AT[0])
    other = kb.create_task(conn, title="other", assignee="w", complexity="S", priority=0)

    assert other not in held
    assert kbd.has_spawnable_ready(conn, skip=held) is True


def test_review_lane_skip(conn, quota_gate_closed, all_assignees_spawnable):
    rev = _held_card(conn, status="review")
    res = kbd.dispatch_once(conn, spawn_fn=_spawn)
    held = quota_held_ids([res], WARNED_AT[0])

    assert rev in held
    assert kbd.has_spawnable_review(conn, skip=held) is False
    assert kbd.has_spawnable_review(conn, skip=frozenset()) is True
    # A review card the tick did not hold is still pending.
    other = kb.create_task(conn, title="r2", assignee="w", priority=0)
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (other,))
    conn.commit()
    assert kbd.has_spawnable_review(conn, skip=held) is True


# --- what quota_held_ids must not exempt ----------------------------------------------


@pytest.mark.parametrize("retry_at", [None, WARNED_AT[0], WARNED_AT[0] - 1])
def test_hold_without_future_retry_is_not_exempt(retry_at):
    res = kbd.DispatchResult(quota_held=[("t_a", "headroom 0%", retry_at)],
                             respawn_guarded=[("t_a", "quota_hold")])
    assert quota_held_ids([res], WARNED_AT[0]) == frozenset()


@pytest.mark.parametrize("reason", ["tier_exhausted", "blocker_auth", "rate_limit_cooldown"])
def test_other_respawn_guards_are_not_exempt(reason):
    res = kbd.DispatchResult(respawn_guarded=[("t_b", reason)])
    assert quota_held_ids([res], WARNED_AT[0]) == frozenset()


def test_none_results_are_skipped():
    res = kbd.DispatchResult(quota_held=[("t_c", "headroom 0%", RETRY_AT)])
    assert quota_held_ids([None, res, None], WARNED_AT[0]) == frozenset({"t_c"})
    assert quota_held_ids([None], WARNED_AT[0]) == frozenset()


# --- the gateway's probe ------------------------------------------------------------


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    for name in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                 "HERMES_KANBAN_TASK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    create_board("b")
    monkeypatch.setattr(kwd, "_board_slugs", lambda _kb: ["b"])
    return tmp_path


def test_gateway_ready_nonempty_honours_skip(board_home, all_assignees_spawnable):
    with kbc.connect_closing(board="b") as c:
        tid = kb.create_task(c, title="held", assignee="w", priority=0)
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    dispatcher = kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)

    assert dispatcher.ready_nonempty(frozenset()) is True
    assert dispatcher.ready_nonempty(frozenset({tid})) is False
