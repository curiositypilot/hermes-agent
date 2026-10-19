"""Auto-decompose pins each board for its own context, never in os.environ.

Regression for intake t_936a2bc2 (upstream #107718 K10): the tick wrote the
process-global ``os.environ["HERMES_KANBAN_BOARD"]`` while iterating boards. It
runs in a gateway thread beside chat turns, so a terminal command spawned in
that window copied the pin and ``hermes kanban create`` without ``--board``
landed on another board.
"""

from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_boards import create_board


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    for name in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                 "HERMES_KANBAN_TASK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    create_board("b")
    return tmp_path


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def _other_thread_board() -> str:
    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("board", kb.get_current_board()))
    t.start()
    t.join()
    return seen["board"]


def _observe(seen: list, where: str) -> None:
    seen.append({
        "where": where,
        "env": os.environ.get("HERMES_KANBAN_BOARD"),
        "board": kb.get_current_board(),
        "other_thread": _other_thread_board(),
    })


def test_tick_pins_board_for_its_own_context_only(board_home, monkeypatch):
    import hermes_cli

    monkeypatch.setattr(kwd, "_board_slugs", lambda _kb: ["b"])
    seen: list = []

    def list_triage_ids(**_kw):
        _observe(seen, "list_triage_ids")
        return ["t1"]

    def decompose_task(task_id, author=None):
        _observe(seen, "decompose_task")
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=list_triage_ids, decompose_task=decompose_task)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    assert _dispatcher().auto_decompose_tick(5) == 1

    assert [s["where"] for s in seen] == ["list_triage_ids", "decompose_task"]
    for s in seen:
        assert s["env"] is None, f"{s['where']} saw the board pinned in os.environ: {s['env']!r}"
        assert s["board"] == "b", f"{s['where']} resolved board {s['board']!r}, expected 'b'"
        assert s["other_thread"] == "default", (
            f"a concurrent thread saw board {s['other_thread']!r} during {s['where']}")
    assert "HERMES_KANBAN_BOARD" not in os.environ
    assert kb.get_current_board() == "default"
