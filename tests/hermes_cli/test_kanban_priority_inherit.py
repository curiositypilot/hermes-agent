"""Priority defaults inherit through the Kanban create surfaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _created_event(conn, task_id):
    return next(event for event in kb.list_events(conn, task_id) if event.kind == "created")


def test_create_inherits_single_parent_priority_and_records_source(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent", priority=2)
        child_id = kb.create_task(conn, title="child", parents=[parent_id])

        child = kb.get_task(conn, child_id)
        payload = _created_event(conn, child_id).payload

    assert child.priority == 2
    assert payload["priority"] == 2
    assert payload["priority_source"] == "parents"


def test_create_inherits_maximum_priority_from_multiple_parents(kanban_home):
    with kbc.connect_closing() as conn:
        low_id = kb.create_task(conn, title="low", priority=1)
        high_id = kb.create_task(conn, title="high", priority=3)
        child_id = kb.create_task(conn, title="child", parents=[low_id, high_id])

        assert kb.get_task(conn, child_id).priority == 3


@pytest.mark.parametrize(("creator_priority", "expected"), [(2, 1), (0, 0)])
def test_worker_followup_inherits_one_below_creator(kanban_home, creator_priority, expected):
    with kbc.connect_closing() as conn:
        creator_id = kb.create_task(conn, title="creator", priority=creator_priority)
        child_id = kb.create_task(conn, title="follow-up", creator_task_id=creator_id)

        child = kb.get_task(conn, child_id)
        payload = _created_event(conn, child_id).payload

    assert child.priority == expected
    assert payload["priority"] == expected
    assert payload["priority_source"] == "creator"


def test_explicit_zero_overrides_parent_priority(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent", priority=2)
        child_id = kb.create_task(conn, title="child", parents=[parent_id], priority=0)

        child = kb.get_task(conn, child_id)
        payload = _created_event(conn, child_id).payload

    assert child.priority == 0
    assert payload["priority"] == 0
    assert payload["priority_source"] == "explicit"


def test_create_without_parent_or_creator_defaults_to_zero(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="background task")
        task = kb.get_task(conn, task_id)
        payload = _created_event(conn, task_id).payload

    assert task.priority == 0
    assert payload["priority_source"] == "default"


def test_cli_create_without_priority_inherits_from_parent(kanban_home, capsys, monkeypatch):
    from hermes_cli.kanban import kanban_command
    from hermes_cli.kanban_parser import build_parser

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent", priority=2)

    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "create", "CLI child", "--parent", parent_id,
        "--assignee", "peer", "--json",
    ])
    assert args.priority is None
    assert kanban_command(args) == 0
    created = json.loads(capsys.readouterr().out)

    assert created["priority"] == 2


def test_tool_create_without_priority_inherits_from_parent(kanban_home, monkeypatch):
    import tools.kanban_tools  # noqa: F401 — register the kanban tool
    from tools.registry import registry

    with kbc.connect_closing() as conn:
        creator_id = kb.create_task(conn, title="worker", priority=2)
        parent_id = kb.create_task(conn, title="parent", priority=3)
    monkeypatch.setenv("HERMES_KANBAN_TASK", creator_id)

    result = json.loads(registry.dispatch("kanban_create", {
        "title": "tool child", "assignee": "peer", "parents": [parent_id],
    }))

    assert result["ok"] is True
    with kbc.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
    assert child.priority == 3


def test_swarm_worker_inherits_root_priority_but_explicit_zero_wins(kanban_home):
    with kbc.connect_closing() as conn:
        created = create_swarm(
            conn,
            goal="Check swarm priority inheritance.",
            workers=[
                SwarmWorkerSpec(profile="worker", title="Inherited", body=""),
                SwarmWorkerSpec(profile="worker", title="Explicit zero", body="", priority=0),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            priority=2,
        )
        workers = [kb.get_task(conn, task_id) for task_id in created.worker_ids]

    assert all(task is not None for task in workers)
    workers = [task for task in workers if task is not None]
    assert [task.priority for task in workers] == [2, 0]
