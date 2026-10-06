"""A ``dir`` workspace on a git repo defaults to worktree isolation so parallel
workers never share a checkout; ``--allow-shared`` keeps the shared ``dir``."""

from __future__ import annotations

import argparse
import os

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    c = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    (path / ".git").mkdir(parents=True)
    return path


def test_dir_on_repo_becomes_worktree(kanban_conn, repo):
    tid = kb.create_task(kanban_conn, title="x", workspace_kind="dir", workspace_path=str(repo))
    task = kb.get_task(kanban_conn, tid)
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)


def test_allow_shared_keeps_dir_on_repo(kanban_conn, repo):
    tid = kb.create_task(
        kanban_conn, title="x", workspace_kind="dir", workspace_path=str(repo), allow_shared=True,
    )
    task = kb.get_task(kanban_conn, tid)
    assert task.workspace_kind == "dir"
    assert task.workspace_path == str(repo)


def test_dir_on_non_repo_stays_dir(kanban_conn, tmp_path):
    plain = tmp_path / "notes"
    plain.mkdir()
    tid = kb.create_task(kanban_conn, title="x", workspace_kind="dir", workspace_path=str(plain))
    assert kb.get_task(kanban_conn, tid).workspace_kind == "dir"


def test_default_stays_scratch(kanban_conn):
    tid = kb.create_task(kanban_conn, title="x")
    assert kb.get_task(kanban_conn, tid).workspace_kind == "scratch"


def test_project_dir_on_repo_becomes_project_worktree(kanban_conn, repo):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name="Repo App", folders=[str(repo)])
        proj = pdb.get_project(pc, pid)
    tid = kb.create_task(kanban_conn, title="x", workspace_kind="dir", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    kb.init_db()
    return home


@pytest.mark.parametrize("extra, expected", [([], "worktree"), (["--allow-shared"], "dir")])
def test_cli_create_dir_repo(kanban_home, repo, extra, expected):
    """Real ``hermes kanban create --workspace dir:<repo>`` parser path."""
    root = argparse.ArgumentParser(prog="hermes")
    kanban_cli.build_parser(root.add_subparsers())
    args = root.parse_args(["kanban", "create", "x", "--workspace", f"dir:{repo}", *extra])
    assert kanban_cli._cmd_create(args) == 0
    with kbc.connect_closing() as conn:
        task = kb.list_tasks(conn)[-1]
    assert task.workspace_kind == expected
    assert task.workspace_path == str(repo)
