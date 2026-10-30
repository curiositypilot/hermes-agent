"""Terminal-subshell PATH completion in ``tools/environments/local.py``.

A backend started by a non-interactive SSH session, systemd or a GUI launcher
inherits a PATH without ``~/.local/bin`` (only the login shell adds it), so CLIs
installed there were ``command not found`` from the terminal tool (#111778).
"""

import os
import sys

import pytest

from tools.environments import local as local_mod
from tools.environments.local import _append_missing_sane_path_entries, _make_run_env

pytestmark = pytest.mark.platforms("posix")  # POSIX PATH completion only


def test_existing_user_local_bin_appended_after_inherited_entries(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", list)
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", list)
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    entries = _make_run_env({})["PATH"].split(os.pathsep)

    assert entries[:2] == ["/usr/bin", "/bin"]
    assert entries.count(str(local_bin)) == 1
    # Already on PATH: position kept, no duplicate appended.
    already = _append_missing_sane_path_entries(f"{local_bin}:/usr/bin").split(":")
    assert already[0] == str(local_bin) and already.count(str(local_bin)) == 1


def test_missing_user_local_bin_not_appended(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", list)

    assert ".local" not in _append_missing_sane_path_entries("/usr/bin:/bin")


def test_background_spawn_env_completes_path_like_foreground(monkeypatch, tmp_path):
    """``ProcessRegistry._spawn_env`` resolves the same managed-runtime and
    ``~/.local/bin`` entries as a foreground ``_make_run_env`` run: a background
    job could not find a tool that lives only in ``$HERMES_HOME/bin`` while the
    same command ran in the foreground (#124820)."""
    from tools.process_registry import ProcessRegistry

    managed_bin = tmp_path / "hermes-home" / "bin"
    managed_bin.mkdir(parents=True)
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", list)
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(managed_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    fg = _make_run_env({})["PATH"].split(":")
    bg = ProcessRegistry._spawn_env({})["PATH"].split(":")

    for entries in (fg, bg):
        # Inherited user entries keep precedence, ahead of the appended dirs.
        assert entries[:2] == ["/usr/bin", "/bin"], entries
        for entry in (str(managed_bin), str(local_bin)):
            assert entries.count(entry) == 1, entries
            assert entries.index(entry) > entries.index("/bin"), entries


def test_background_spawn_env_keeps_existing_user_entry_position(monkeypatch, tmp_path):
    """A managed dir already on the user's PATH is neither duplicated nor moved
    behind the appended entries in a background spawn (#124820)."""
    from tools.process_registry import ProcessRegistry

    managed_bin = tmp_path / "hermes-home" / "bin"
    managed_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{managed_bin}:/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", list)
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(managed_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    bg = ProcessRegistry._spawn_env({})["PATH"].split(":")

    assert bg[:3] == [str(managed_bin), "/usr/bin", "/bin"], bg
    assert bg.count(str(managed_bin)) == 1, bg
