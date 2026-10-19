"""A fenced worker naming another board gets an error, never the pinned board's data.

Regression for intake t_20d642ad: inside a dispatched worker (HERMES_KANBAN_DB
pinned), `hermes kanban --board yuliia show t_x` answered "no such task" from the
pinned default DB, and `boards list` showed the pinned board's counts under every
slug. The fence itself (5ec6baa) stays; it now refuses instead of substituting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_KANBAN_ENV = (
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME", "HERMES_KANBAN_TASK",
    "HERMES_DELEGATED_CHILD_CONTEXT",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var in _KANBAN_ENV:
        monkeypatch.delenv(var, raising=False)
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("worker-own")
    kb.create_board("target-board")
    return h


def _fence_worker(monkeypatch, *, board_env: str | None = "worker-own") -> Path:
    """The env a dispatched worker carries, pinned to board ``worker-own``."""
    pinned = kb.kanban_db_path(board="worker-own")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(kb.workspaces_root(board="worker-own")))
    if board_env:
        monkeypatch.setenv("HERMES_KANBAN_BOARD", board_env)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fence")
    return pinned


def _cli(*argv: str) -> int:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    return kc.kanban_command(parser.parse_args(["kanban", *argv]))


def test_default_board_matches_its_natural_path_without_board_env(monkeypatch, tmp_path):
    """Pin = the default board's own DB and HERMES_KANBAN_BOARD unset: board="default"
    is the pinned board (natural-path match) and must not raise."""
    for var in _KANBAN_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pinned = kb.kanban_home() / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fence")
    assert kb.kanban_db_path(board="default") == pinned
    with pytest.raises(kb.KanbanBoardFenced):
        kb.kanban_db_path(board="other-board")


def test_pin_first_machine_flow_wins_over_the_worker_fence(monkeypatch, tmp_path):
    for var in _KANBAN_ENV:
        monkeypatch.delenv(var, raising=False)
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fence")
    with kb.pin_first_board_resolution():
        assert kb.kanban_db_path(board="other-board") == pinned


def test_attachments_root_is_fenced_without_its_own_pin(home, monkeypatch):
    """The dispatcher never pins HERMES_KANBAN_ATTACHMENTS_ROOT; the DB pin still fences it."""
    _fence_worker(monkeypatch)
    with pytest.raises(kb.KanbanBoardFenced):
        kb.attachments_root(board="target-board")
    assert kb.attachments_root(board="worker-own") == kb.board_dir("worker-own") / "attachments"


def test_cli_board_show_and_create_fail_at_resolution(home, monkeypatch, capsys):
    pinned = _fence_worker(monkeypatch)
    assert _cli("--board", "target-board", "show", "t_x") == 1
    err = capsys.readouterr().err
    assert "outside this worker's kanban fence" in err
    assert "no such task" not in err

    assert _cli("--board", "target-board", "create", "x") == 1
    assert "outside this worker's kanban fence" in capsys.readouterr().err
    with kbc.connect(db_path=pinned) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE title = 'x'").fetchone()[0] == 0
    with kbc.connect(db_path=kb.board_dir("target-board") / "kanban.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE title = 'x'").fetchone()[0] == 0


def test_cli_own_board_still_works_in_a_fenced_worker(home, monkeypatch):
    pinned = _fence_worker(monkeypatch)
    assert _cli("--board", "worker-own", "create", "mine") == 0
    with kbc.connect(db_path=pinned) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE title = 'mine'").fetchone()[0] == 1


def test_boards_create_in_a_fenced_worker_refuses_before_writing(home, monkeypatch, capsys):
    _fence_worker(monkeypatch)
    assert _cli("boards", "create", "new-board") == 1
    assert "outside this worker's kanban fence" in capsys.readouterr().err
    assert not kb.board_dir("new-board").exists()


def test_boards_list_json_marks_fenced_boards(home, monkeypatch, capsys):
    pinned = _fence_worker(monkeypatch)
    with kbc.connect(db_path=pinned) as conn:
        kb.create_task(conn, title="own task", assignee="worker")
    assert _cli("boards", "list", "--json") == 0
    rows = {b["slug"]: b for b in json.loads(capsys.readouterr().out)}
    own, other = rows["worker-own"], rows["target-board"]
    assert own["fenced"] is False and own["counts"] and own["total"] == 1
    assert other["fenced"] is True and other["counts"] is None and other["total"] is None
    assert other["db_path"] == str(kb.board_dir("target-board") / "kanban.db")


def test_boards_list_and_show_render_fenced_boards(home, monkeypatch, capsys):
    """Human output labels fenced boards instead of "(empty)"; `boards show` on a
    current board outside the fence does not crash."""
    _fence_worker(monkeypatch, board_env=None)
    kb.current_board_path().write_text("target-board\n", encoding="utf-8")
    assert _cli("boards", "list") == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "target-board" in ln)
    assert "(outside worker fence)" in line
    assert _cli("boards", "show") == 0
    assert "Tasks:        (outside worker fence)" in capsys.readouterr().out
