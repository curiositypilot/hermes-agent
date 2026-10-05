"""Kanban task data-class CLI and JSON fields."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  data_policies:\n    default: internal\n"
        "    tenant_defaults: {work: confidential}\n"
        "    classes:\n      internal: {providers: any, fallback: true, auxiliary: any}\n"
        "      confidential: {providers: [anthropic], fallback: false, auxiliary: same_provider_or_fail}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _parser():
    parser = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(parser.add_subparsers(dest="command"))
    return parser


def test_create_data_class_and_show_json_resolved_class(kanban_home, capsys):
    parser = _parser()
    create_args = parser.parse_args([
        "kanban", "create", "explicit class", "--data-class", "confidential", "--json",
    ])
    assert kc._cmd_create(create_args) == 0
    explicit = json.loads(capsys.readouterr().out)
    assert explicit["data_class"] == "confidential"
    assert explicit["data_class_resolved"] == "confidential"

    with kbc.connect_closing() as conn:
        inherited_id = kb.create_task(conn, title="tenant default", tenant="work")
    show_args = parser.parse_args(["kanban", "show", inherited_id, "--json"])
    assert kc._cmd_show(show_args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["task"]["data_class"] is None
    assert shown["task"]["data_class_resolved"] == "confidential"


def test_child_task_persists_parent_resolved_class(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="explicit parent", data_class="confidential", tenant="other")
        child_id = kb.create_task(conn, title="child", tenant="work", parents=(parent_id,))
        child = kb.get_task(conn, child_id)
    assert child.data_class == "confidential"


def test_child_inheritance_never_loosens_class(kanban_home):
    from agent.provider_policy import ProviderDenied

    with kbc.connect_closing() as conn:
        internal_parent = kb.create_task(conn, title="internal parent", tenant="other")
        confidential_parent = kb.create_task(conn, title="conf parent", data_class="confidential")
        # Tenant default `work` -> confidential survives an internal parent.
        tightened = kb.get_task(conn, kb.create_task(
            conn, title="work child", tenant="work", parents=(internal_parent,),
        ))
        # Unlabeled child of an internal parent is persisted explicitly.
        plain = kb.get_task(conn, kb.create_task(
            conn, title="plain child", tenant="other", parents=(internal_parent,),
        ))
        with pytest.raises(ProviderDenied, match="loosen"):
            kb.create_task(
                conn, title="downgrade", data_class="internal", parents=(confidential_parent,),
            )
    assert tightened.data_class == "confidential"
    assert plain.data_class == "internal"


def test_tasks_schema_contains_additive_data_class_column(kanban_home):
    with kbc.connect_closing() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "data_class" in columns


def test_existing_kanban_schema_migrates_data_class_without_losing_tasks(kanban_home):
    legacy_db = kanban_home / "legacy.db"
    conn = sqlite3.connect(legacy_db)
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "status TEXT NOT NULL, created_at INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO tasks VALUES ('legacy-task', 'keep me', 'done', 1)")
    conn.commit()
    conn.close()

    kb.init_db(legacy_db)
    with kbc.connect_closing(legacy_db) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        migrated = conn.execute("SELECT id, title, status, data_class FROM tasks").fetchone()
    assert "data_class" in columns
    assert tuple(migrated) == ("legacy-task", "keep me", "done", None)


def test_dispatch_exports_resolved_data_class(kanban_home, monkeypatch, tmp_path):
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles
    from tools.environments import local
    from tools import process_registry
    from agent import secret_scope

    monkeypatch.setattr(profiles, "normalize_profile_name", lambda name: name)
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda _name: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(local, "build_subprocess_env", lambda **_kwargs: {})
    monkeypatch.setattr(local, "_is_routed_home", lambda _home: False)
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: False)
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda env: env)
    monkeypatch.setattr(kbd, "_worker_argv", lambda *_args: ["hermes"])
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda _task, command: command)
    monkeypatch.setattr(kbd, "_open_worker_log", lambda *_args: object())
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kbd._kb, "kanban_db_path", lambda **_kwargs: kanban_home / "kanban.db")
    monkeypatch.setattr(kbd._kb, "workspaces_root", lambda **_kwargs: str(kanban_home / "workspaces"))
    monkeypatch.setattr(kbd._kb, "get_current_board", lambda: "default")
    monkeypatch.setattr(kbd._kb, "_normalize_board_slug", lambda _board: "default")
    captured = {}
    monkeypatch.setattr(
        kbd.subprocess, "Popen",
        lambda *_args, **kwargs: (captured.update(kwargs) or SimpleNamespace(pid=712)),
    )
    task = SimpleNamespace(
        id="task-x", assignee="worker", tenant="work", data_class=None,
        branch_name=None, current_run_id=None, claim_lock=None,
        goal_mode=False, max_runtime_seconds=None,
    )

    assert kbd._default_spawn(task, str(tmp_path)) == 712
    assert captured["env"]["HERMES_DATA_CLASS"] == "confidential"

