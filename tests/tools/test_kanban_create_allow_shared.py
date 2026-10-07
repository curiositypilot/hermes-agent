"""``kanban_create`` tool: a ``dir`` workspace on a git repo becomes a worktree
unless the caller passes ``allow_shared`` (the tool twin of ``--allow-shared``)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    (path / ".git").mkdir(parents=True)
    return path


def test_schema_exposes_allow_shared():
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA
    props = KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert props["allow_shared"]["type"] == "boolean"


@pytest.mark.parametrize("extra, expected", [
    ({}, "worktree"),
    ({"allow_shared": False}, "worktree"),
    ({"allow_shared": True}, "dir"),
    ({"allow_shared": "true"}, "dir"),
])
def test_create_dir_on_repo(worker_env, repo, extra, expected):
    import tools.kanban_tools  # noqa: F401  (registers the tools)
    from tools.registry import registry
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out = registry.dispatch("kanban_create", {
        "title": "child", "assignee": "peer",
        "workspace_kind": "dir", "workspace_path": str(repo), **extra,
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d["workspace_kind"] == expected
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, d["task_id"])
    assert task.workspace_kind == expected
    assert task.workspace_path == str(repo)
