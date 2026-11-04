"""A checkpoint's per-project index lock never disables checkpoints of a directory for good.

Regression for intake t_011aaf97: a timed-out ``git add -A`` was SIGKILLed by
``subprocess.run(timeout=)``, which leaves git's ``<index>.lock`` behind (git removes it only from
its SIGTERM/SIGINT/SIGHUP handlers); every later checkpoint of that directory then failed with
"File exists" until the lock was deleted by hand.
"""
from __future__ import annotations

import os
import time

import pytest

import tools.checkpoint_manager as checkpoints
from hermes_cli.gitlock import STALE_LOCK_MIN_AGE_SECONDS


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    workdir = tmp_path / "project"
    workdir.mkdir()
    (workdir / "a.txt").write_text("one", encoding="utf-8")
    manager = checkpoints.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(workdir), "first")
    store = checkpoints._store_path()
    index = checkpoints._index_path(store, checkpoints._project_hash(str(workdir)))
    (workdir / "a.txt").write_text("two", encoding="utf-8")
    manager.new_turn()
    return manager, workdir, index.with_name(index.name + ".lock")


def _age(lock, seconds):
    lock.write_text("", encoding="utf-8")
    then = time.time() - seconds
    os.utime(lock, (then, then))


def _snapshot_count(manager, workdir):
    return len(manager.list_checkpoints(str(workdir)))


@pytest.mark.platforms("posix")
def test_timeout_releases_index_lock(tmp_path):
    workdir = tmp_path / "slow"
    workdir.mkdir()
    (workdir / ".gitattributes").write_text("* filter=slow\n", encoding="utf-8")
    (workdir / "file.txt").write_text("content\n", encoding="utf-8")
    store = tmp_path / "checkpoints" / "store"  # its own base: _init_store migrates foreign dirs in the base
    assert checkpoints._init_store(store, str(workdir)) is None
    index = tmp_path / "index"
    lock = index.with_name(index.name + ".lock")

    start = time.monotonic()
    ok, _out, err = checkpoints._run_git(
        ["-c", "filter.slow.clean=sleep 30", "add", "-A"], store, str(workdir),
        timeout=2, index_file=index)

    assert not ok
    assert "timed out" in err
    assert not lock.exists()
    assert time.monotonic() - start < 15


def test_stale_unheld_lock_removed(project):
    manager, workdir, lock = project
    _age(lock, 3600)

    assert manager.ensure_checkpoint(str(workdir), "second")
    assert not lock.exists()
    assert _snapshot_count(manager, workdir) == 2


def test_young_lock_kept(project):
    manager, workdir, lock = project
    _age(lock, 60)

    assert not manager.ensure_checkpoint(str(workdir), "second")
    assert lock.exists()
    assert _snapshot_count(manager, workdir) == 1


@pytest.mark.platforms("posix")
def test_held_lock_kept(project):
    manager, workdir, lock = project
    _age(lock, STALE_LOCK_MIN_AGE_SECONDS + 600)

    with open(lock, "rb"):
        assert not manager.ensure_checkpoint(str(workdir), "second")
        assert lock.exists()
    assert _snapshot_count(manager, workdir) == 1


@pytest.mark.platforms("posix")
def test_unknowable_holder_keeps_lock(project, monkeypatch):
    import hermes_cli._early_recovery as early_recovery

    manager, workdir, lock = project
    _age(lock, 3600)
    monkeypatch.setattr(early_recovery, "_held_open", lambda *a, **k: None)

    assert not manager.ensure_checkpoint(str(workdir), "second")
    assert lock.exists()
