"""Tests for the worker-context card header: completion contract and children.

The card is the worker's first message.  ``_ctx_header`` must render the
completion contract (when set) and child ids (when the task has children)
so the worker sees them without a ``kanban_show`` call.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Fresh kanban DB in a temp HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    yield kb, conn
    conn.close()


def _header_text(kb, conn, task_id: str) -> str:
    """Render just the header lines for *task_id*."""
    from hermes_cli.kanban_db_context import _ctx_header
    task = kb.get_task(conn, task_id)
    lines: list[str] = []
    _ctx_header(lines, task, conn)
    return "\n".join(lines)


def test_header_shows_contract_and_children(db):
    kb, conn = db
    parent = kb.create_task(conn, title="parent", assignee="w", completion_contract="test:true")
    c1 = kb.create_task(conn, title="child-one", assignee="w", parents=[parent])
    c2 = kb.create_task(conn, title="child-two", assignee="w", parents=[parent])
    text = _header_text(kb, conn, parent)
    assert "Completion contract: test:true" in text
    assert c1 in text
    assert c2 in text
    # status and title appear next to each id
    assert f"{c1} (" in text
    assert f"{c2} (" in text


def test_header_no_children_omits_children_line(db):
    kb, conn = db
    tid = kb.create_task(conn, title="solo", assignee="w")
    text = _header_text(kb, conn, tid)
    assert "Children:" not in text


def test_header_25_children_shows_20_and_plus_5(db):
    kb, conn = db
    parent = kb.create_task(conn, title="big-parent", assignee="w")
    child_ids = []
    for i in range(25):
        cid = kb.create_task(conn, title=f"child-{i:02d}", assignee="w", parents=[parent])
        child_ids.append(cid)
    text = _header_text(kb, conn, parent)
    assert "Children:" in text
    assert "(+5 more)" in text
    # children are listed in id order: the first 20 appear, the other 5 do not
    ordered = sorted(child_ids)
    for cid in ordered[:20]:
        assert cid in text
    for cid in ordered[20:]:
        assert cid not in text
