"""HERMES_KANBAN_HOME vs a kanban env pin outside it fails closed (intake t_29de271a).

A worker's dispatcher-pinned HERMES_KANBAN_DB outranks HERMES_KANBAN_HOME. A probe
inside a worker that set HERMES_KANBAN_HOME=<tmp> to isolate its board therefore
wrote "root triage" junk cards to the live board. A pin outside an explicitly
named kanban home now raises KanbanPathConflict instead of resolving silently.
"""
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture(autouse=True)
def _clean_kanban_env(monkeypatch):
    for name in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_ATTACHMENTS_ROOT",
        "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME", "HERMES_KANBAN_TASK",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def roots(tmp_path):
    a, b = tmp_path / "live", tmp_path / "probe"
    a.mkdir()
    b.mkdir()
    return a, b


def test_home_outside_pin_raises(monkeypatch, roots):
    a, b = roots
    monkeypatch.setenv("HERMES_KANBAN_DB", str(a / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(b))
    with pytest.raises(kb.KanbanPathConflict) as exc:
        kb.kanban_db_path()
    msg = str(exc.value)
    assert str(a / "kanban.db") in msg
    assert str(b.resolve()) in msg
    assert "env -u" in msg


def test_home_outside_workspaces_pin_raises(monkeypatch, roots):
    a, b = roots
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(a / "kanban" / "workspaces"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(b))
    with pytest.raises(kb.KanbanPathConflict, match="HERMES_KANBAN_WORKSPACES_ROOT"):
        kb.workspaces_root()


def test_pin_inside_home_ok(monkeypatch, roots):
    _, b = roots
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(b))
    for pin in (b / "kanban.db", b / "kanban" / "boards" / "x" / "kanban.db"):
        monkeypatch.setenv("HERMES_KANBAN_DB", str(pin))
        assert kb.kanban_db_path() == pin


def test_pin_without_home_unchanged(monkeypatch, roots):
    a, _ = roots
    monkeypatch.setenv("HERMES_KANBAN_DB", str(a / "kanban.db"))
    assert kb.kanban_db_path() == a / "kanban.db"


def test_home_without_pin_unchanged(monkeypatch, roots):
    _, b = roots
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(b))
    assert kb.kanban_db_path() == b / "kanban.db"


def test_explicit_unfenced_board_skips_check(monkeypatch, roots):
    a, b = roots
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(b))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(a / "kanban.db"))
    assert kb.kanban_db_path(board="other") == b / "kanban" / "boards" / "other" / "kanban.db"


def _task_count(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("select count(*) from tasks").fetchone()[0]


def test_incident_t_29de271a_probe_replay(monkeypatch, roots):
    a, b = roots
    live_db = a / "kanban.db"
    # The dispatcher-pinned worker env the probe inherited.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(a / "kanban" / "workspaces"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0000adjd")
    kb.init_db()
    assert _task_count(live_db) == 0

    # The probe: "isolate" the board, then connect + create.
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(b))
    with pytest.raises(kb.KanbanPathConflict):
        conn = kbc.connect()
        kb.create_task(conn, title="root triage", triage=True)
    assert _task_count(live_db) == 0
    assert not (b / "kanban.db").exists()
