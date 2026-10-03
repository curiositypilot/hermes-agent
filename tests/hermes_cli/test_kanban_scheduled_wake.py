"""Dated ``scheduled`` cards: the dispatcher wakes them on their date (start | ask).

Behaviour contracts:
* start  -> ``ready`` (``todo`` while a parent is open) and the card dispatches.
* ask    -> sticky ``blocked`` with ``woke_from: scheduled``; waits while a parent is open.
* legacy -> no column value: ``until YYYY-MM-DD`` in the latest scheduled reason is the date.
* undated -> ask after ``kanban.schedule_recheck_days``.
* re-dating a scheduled card works in one call; create can park a card dated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_schedule as ks


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


PAST = "2020-01-01"
FUTURE = "2099-01-01"


def _last_event(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind][-1]


def test_start_mode_wakes_to_ready(conn):
    t = kb.create_task(conn, title="start me", assignee="nobody")
    assert kb.schedule_task(conn, t, reason="later", until=PAST, then="start")
    assert ks.wake_due_scheduled(conn) == [(t, "ready")]
    task = kb.get_task(conn, t)
    assert task.status == "ready"
    assert task.scheduled_until is None and task.scheduled_then is None
    assert _last_event(conn, t, "unblocked").payload["woke_from"] == "scheduled"
    # Released, not sticky: a later recompute leaves it ready.
    kb.recompute_ready(conn)
    assert kb.get_task(conn, t).status == "ready"


def test_ask_mode_wakes_to_sticky_blocked(conn):
    t = kb.create_task(conn, title="ask me", assignee="nobody")
    assert kb.schedule_task(conn, t, reason="decide on wake", until=PAST, then="ask")
    assert ks.wake_due_scheduled(conn) == [(t, "blocked")]
    assert kb.get_task(conn, t).status == "blocked"
    ev = _last_event(conn, t, "blocked")
    assert ev.payload["woke_from"] == "scheduled"
    assert "decide on wake" in ev.payload["reason"]
    kb.recompute_ready(conn)  # sticky: never auto-promoted
    assert kb.get_task(conn, t).status == "blocked"


def test_future_date_does_not_wake(conn):
    t = kb.create_task(conn, title="not yet", assignee="nobody")
    kb.schedule_task(conn, t, until=FUTURE, then="start")
    assert ks.wake_due_scheduled(conn) == []
    assert kb.get_task(conn, t).status == "scheduled"


def test_parents_open_start_goes_todo_ask_waits(conn):
    parent = kb.create_task(conn, title="parent", assignee="nobody")
    start = kb.create_task(conn, title="start child", assignee="nobody", parents=[parent])
    ask = kb.create_task(conn, title="ask child", assignee="nobody", parents=[parent])
    kb.schedule_task(conn, start, until=PAST, then="start")
    kb.schedule_task(conn, ask, until=PAST, then="ask")
    assert dict(ks.wake_due_scheduled(conn)) == {start: "todo"}
    assert kb.get_task(conn, start).status == "todo"
    assert kb.get_task(conn, ask).status == "scheduled"
    # Parent finishes: start child promotes through the normal path, ask child wakes.
    kb.complete_task(conn, parent, result="ok")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, start).status == "ready"
    assert ks.wake_due_scheduled(conn) == [(ask, "blocked")]


def test_legacy_reason_date_is_honoured(conn):
    t = kb.create_task(conn, title="legacy", assignee="nobody")
    assert kb.schedule_task(conn, t, reason=f"until {PAST}: check X")
    task = kb.get_task(conn, t)
    # The reason date is lifted into the column at schedule time.
    assert task.scheduled_until == ks.parse_until(PAST) and task.scheduled_then == "ask"
    # A card parked by an older build has no column value: still read from the event.
    conn.execute("UPDATE tasks SET scheduled_until = NULL, scheduled_then = NULL WHERE id = ?", (t,))
    s = ks.schedule_for(conn, t)
    assert (s.source, s.then, s.wake_at) == ("reason", "ask", ks.parse_until(PAST))
    assert ks.wake_due_scheduled(conn) == [(t, "blocked")]


def test_legacy_future_reason_waits(conn):
    t = kb.create_task(conn, title="legacy future", assignee="nobody")
    kb.schedule_task(conn, t, reason=f"until {FUTURE}: later")
    conn.execute("UPDATE tasks SET scheduled_until = NULL, scheduled_then = NULL WHERE id = ?", (t,))
    assert ks.wake_due_scheduled(conn) == []


def test_undated_wakes_ask_after_recheck_days(conn):
    t = kb.create_task(conn, title="undated", assignee="nobody")
    kb.schedule_task(conn, t, reason="some day")
    now = time.time()
    assert ks.wake_due_scheduled(conn, now=now, recheck_days=7) == []
    assert ks.wake_due_scheduled(conn, now=now + 8 * 86400, recheck_days=7) == [(t, "blocked")]


def test_recheck_zero_disables_undated_wake(conn):
    t = kb.create_task(conn, title="undated off", assignee="nobody")
    kb.schedule_task(conn, t, reason="some day")
    assert ks.wake_due_scheduled(conn, now=time.time() + 400 * 86400, recheck_days=0) == []
    assert kb.get_task(conn, t).status == "scheduled"
    # A dated card still wakes with the recheck off.
    d = kb.create_task(conn, title="dated", assignee="nobody")
    kb.schedule_task(conn, d, until=PAST, then="start")
    assert ks.wake_due_scheduled(conn, recheck_days=0) == [(d, "ready")]


def test_recheck_days_from_config(kanban_home, conn):
    (kanban_home / "config.yaml").write_text("kanban:\n  schedule_recheck_days: 2\n")
    t = kb.create_task(conn, title="undated cfg", assignee="nobody")
    kb.schedule_task(conn, t, reason="some day")
    assert ks.configured_recheck_days() == 2
    assert ks.wake_due_scheduled(conn, now=time.time() + 3 * 86400) == [(t, "blocked")]


def test_reschedule_scheduled_card_in_place(conn):
    t = kb.create_task(conn, title="move me", assignee="nobody")
    kb.schedule_task(conn, t, until=FUTURE, then="ask")
    assert kb.schedule_task(conn, t, reason="sooner", until=PAST, then="start")
    task = kb.get_task(conn, t)
    assert task.status == "scheduled" and task.scheduled_then == "start"
    assert _last_event(conn, t, "scheduled").payload.get("rescheduled") is True
    assert ks.wake_due_scheduled(conn) == [(t, "ready")]


def test_start_requires_date(conn):
    t = kb.create_task(conn, title="no date", assignee="nobody")
    with pytest.raises(ValueError):
        kb.schedule_task(conn, t, then="start")
    with pytest.raises(ValueError):
        kb.schedule_task(conn, t, until="next tuesday")
    assert kb.get_task(conn, t).status == "ready"


def test_create_born_scheduled(conn):
    t = kb.create_task(conn, title="dated", assignee="nobody", scheduled_until=PAST, scheduled_then="start")
    assert kb.get_task(conn, t).status == "scheduled"
    assert ks.wake_due_scheduled(conn) == [(t, "ready")]
    with pytest.raises(ValueError):
        kb.create_task(conn, title="bad", scheduled_until=PAST, initial_status="blocked")


def test_unblock_clears_schedule(conn):
    t = kb.create_task(conn, title="release early", assignee="nobody")
    kb.schedule_task(conn, t, until=FUTURE, then="start")
    assert kb.unblock_task(conn, t)
    task = kb.get_task(conn, t)
    assert task.status == "ready" and task.scheduled_until is None


def test_datetime_until_wakes_after_the_minute(conn):
    soon = time.strftime("%Y-%m-%dT%H:%M", time.localtime(time.time() + 180))
    t = kb.create_task(conn, title="in 3 min", assignee="nobody")
    kb.schedule_task(conn, t, until=soon, then="start")
    assert ks.wake_due_scheduled(conn) == []
    assert ks.wake_due_scheduled(conn, now=time.time() + 240) == [(t, "ready")]


def test_dispatcher_tick_wakes_and_dry_run_writes_nothing(conn):
    start = kb.create_task(conn, title="tick start", assignee="nobody")
    ask = kb.create_task(conn, title="tick ask", assignee="nobody")
    kb.schedule_task(conn, start, until=PAST, then="start")
    kb.schedule_task(conn, ask, until=PAST, then="ask")

    dry = kbd.dispatch_once(conn, dry_run=True, spawn_fn=lambda *a, **k: None)
    assert dict(dry.woke_scheduled) == {start: "ready", ask: "blocked"}
    assert kb.get_task(conn, start).status == "scheduled"
    assert kb.get_task(conn, ask).status == "scheduled"

    spawned = []
    res = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, board=None: spawned.append(task.id) or None)
    assert dict(res.woke_scheduled) == {start: "ready", ask: "blocked"}
    assert kb.get_task(conn, ask).status == "blocked"
    # The woken start card is claimable on the same tick.
    assert start in spawned or kb.get_task(conn, start).status in ("ready", "running")


def _cli(*argv: str) -> int:
    import argparse
    from hermes_cli import kanban as kcli

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kcli.build_parser(parser.add_subparsers(dest="command"))
    return kcli.kanban_command(parser.parse_args(["kanban", *argv]))


def test_cli_create_schedule_list_show(kanban_home, capsys, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _cli("create", "cli dated", "--assignee", "nobody", "--until", "2099-01-01",
                "--then", "start", "--json") == 0
    tid = json.loads(capsys.readouterr().out)["id"]
    assert _cli("list") == 0
    assert "⏱ → 2099-01-01 start" in capsys.readouterr().out
    # Move the date directly on a scheduled card.
    assert _cli("schedule", tid, "--until", "2099-02-03T09:30", "--then", "ask", "moved") == 0
    assert "2099-02-03 09:30 ask): moved" in capsys.readouterr().out
    # Note-first order parses too.
    assert _cli("schedule", tid, "moved", "again", "--until", "2099-02-03T09:30", "--then", "ask") == 0
    assert "ask): moved again" in capsys.readouterr().out
    assert _cli("show", tid, "--json") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["schedule"] == "⏱ → 2099-02-03 09:30 ask"
    assert shown["task"]["scheduled_then"] == "ask"
    # Bad input is a usage error, not a traceback.
    assert _cli("schedule", tid, "--then", "start") == 2
    assert _cli("create", "x", "--until", "tomorrow") == 2
