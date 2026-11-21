"""A sandboxed Kanban worker imports the live tree, never the card workspace (t_aeedfe09).

The sandbox ``--chdir``s into the card workspace and ``python -m`` puts the cwd ahead of
PYTHONPATH, so a fork worktree's own ``agent/`` shadowed the pinned checkout and the worker
died at boot ("managed worker lost: EOFError", runs 10504/10506).
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from gateway import session_managed_worker as managed

REPO_ROOT = Path(managed.__file__).resolve().parents[1]
WORKER_ARGV = [sys.executable, '-m', 'agent.managed_worker']


def _captured_sandbox_command(monkeypatch, tmp_path, command):
    from hermes_cli import kanban_sandbox
    captured = {}

    def plan(context, command, *, worker_env, profile_home, allow_control_socket):
        captured.update(command=list(command), env=worker_env)
        return None, b''

    monkeypatch.setattr(kanban_sandbox, 'plan_for_managed_worker', plan)
    monkeypatch.setattr(managed, '_owner_control_socket', lambda: None)
    authority = SimpleNamespace(profile_id=str(tmp_path))
    policy = SimpleNamespace(kanban_json=json.dumps(
        {'task_id': 't_x', 'run_id': 1, 'tenant': 'sandboxed', 'workspace': str(tmp_path)}))
    asyncio.run(managed._kanban_sandbox(authority, policy, command, dict(os.environ), REPO_ROOT))
    return captured


def test_sandbox_command_runs_python_with_safe_path(monkeypatch, tmp_path):
    command = list(WORKER_ARGV)
    captured = _captured_sandbox_command(monkeypatch, tmp_path, command)
    assert captured['command'] == [sys.executable, '-P', '-m', 'agent.managed_worker']
    # The plain spawn (no sandbox plan) keeps the caller's argv: no -P leaks into it.
    assert command == WORKER_ARGV


def _shadow_workspace(tmp_path):
    workspace = tmp_path / 'workspace'
    (workspace / 'agent').mkdir(parents=True)
    (workspace / 'agent' / '__init__.py').write_text('')
    (workspace / 'agent' / 'managed_worker.py').write_text("print('SHADOW', flush=True)\nraise SystemExit(3)\n")
    return workspace


def _run(command, env, cwd):
    return subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60, check=False)


def test_workspace_package_cannot_shadow_the_worker(monkeypatch, tmp_path):
    captured = _captured_sandbox_command(monkeypatch, tmp_path, list(WORKER_ARGV))
    workspace = _shadow_workspace(tmp_path)

    fixed = _run(captured['command'], captured['env'], workspace)
    assert 'SHADOW' not in fixed.stdout
    assert fixed.returncode != 3

    old = _run(WORKER_ARGV, captured['env'], workspace)
    assert 'SHADOW' in old.stdout and old.returncode == 3
