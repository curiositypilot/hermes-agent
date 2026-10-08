"""A module-form kanban worker must import the running install, not its cwd.

The dispatcher spawns ``sys.executable -m hermes_cli.main`` with ``cwd`` = the
task workspace. ``-m`` puts the cwd first on ``sys.path``, ahead of the
``PYTHONPATH`` pin, so a workspace that is a hermes-agent worktree (or holds any
``hermes_cli/``, ``pm/``, ``yaml.py``) shadowed the install and every worker on
such a card died at boot (t_8eb8551f). ``-P`` keeps the cwd off ``sys.path``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_dispatch_argv as kda

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="-P needs Python >= 3.11")

ROOT = str(Path(kbd.__file__).resolve().parents[1])


def _module_argv(monkeypatch) -> list[str]:
    monkeypatch.delenv("HERMES_BIN", raising=False)
    return kbd._resolve_hermes_argv()


def _pinned_env(cmd: list[str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    kbd._propagate_module_import_root(cmd, env)
    return env


def test_module_argv_keeps_cwd_off_sys_path(monkeypatch):
    argv = _module_argv(monkeypatch)
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-P", "-m", "hermes_cli.main"]


def test_import_root_is_pinned_for_the_safe_path_form(monkeypatch):
    env = _pinned_env(_module_argv(monkeypatch))
    assert env["PYTHONPATH"].split(os.pathsep)[0] == ROOT


def test_legacy_module_form_is_still_recognised_and_pinned():
    legacy = [sys.executable, "-m", "hermes_cli.main"]
    assert kda._is_module_argv(legacy)
    assert kda._is_module_argv([sys.executable, "-P", "-m", "hermes_cli.main"])
    assert not kda._is_module_argv(["/opt/hermes/bin/hermes"])
    assert _pinned_env(legacy)["PYTHONPATH"].split(os.pathsep)[0] == ROOT


def test_fallback_module_argv_actually_runs(monkeypatch):
    """The module fallback must stay runnable: with -P the PYTHONPATH pin is its only
    route to the package, so spawn it exactly as _default_spawn does."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    argv = _module_argv(monkeypatch)
    r = subprocess.run([*argv, "--version"], env=_pinned_env(argv),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"`{' '.join(argv)} --version` rc={r.returncode}: {r.stderr[-300:]}"


def test_worker_in_a_shadowing_workspace_runs_the_install(tmp_path, monkeypatch):
    shadow = tmp_path / "hermes_cli"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise SystemExit(97)\n")
    argv = _module_argv(monkeypatch)
    env = _pinned_env(argv)

    # The fixture really shadows: the pre-fix argv (no -P) imports the cwd's package.
    old = subprocess.run([sys.executable, "-m", "hermes_cli.main", "--version"], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=60)
    assert old.returncode == 97

    r = subprocess.run([*argv, "--version"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
    assert ROOT in r.stdout
