"""Oversize files are excluded before ``git add`` hashes them.

Regression for intake t_298481e1: ``_take`` staged with a plain ``add -A`` and
only then dropped files over ``max_file_size_mb``, so git hashed and compressed
every byte first. A 67 GB tree timed out mid-add and left a 1.4 GB tmp pack in
the checkpoint store. The contract: no blob of an oversize file ever reaches
the store, from a checkpoint, a diff, or a safe-restore plan.
"""

import os
import subprocess
from pathlib import Path

import pytest

import tools.checkpoint_manager as checkpoints
from tools.checkpoint_manager import CheckpointManager, _store_path

MIB = 1024 * 1024


@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "small.txt").write_text("small\n")
    return d


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    return CheckpointManager(enabled=True, max_snapshots=50, max_file_size_mb=1)


def _blob_sha(path: Path) -> str:
    """The id git would give *path*'s content, computed without writing it."""
    return subprocess.run(
        ["git", "hash-object", "--no-filters", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _in_store(sha: str) -> bool:
    store = _store_path()
    return subprocess.run(
        ["git", "--git-dir", str(store), "cat-file", "-e", sha],
        capture_output=True, check=False,
    ).returncode == 0


def _tree_paths(work_dir: Path, mgr: CheckpointManager) -> set[str]:
    commit = mgr.list_checkpoints(str(work_dir))[0]["hash"]
    out = subprocess.run(
        ["git", "--git-dir", str(_store_path()), "ls-tree", "-r", "-z", "--name-only", commit],
        capture_output=True, check=True,
    ).stdout
    return {os.fsdecode(p) for p in out.split(b"\0") if p}


def _checkpoint(mgr: CheckpointManager, work_dir: Path) -> str:
    assert mgr.ensure_checkpoint(str(work_dir), "cp") is True
    mgr.new_turn()
    return mgr.list_checkpoints(str(work_dir))[0]["hash"]


def test_take_never_writes_an_oversize_blob(mgr, work_dir):
    big = work_dir / "big.bin"
    big.write_bytes(os.urandom(2 * MIB))

    _checkpoint(mgr, work_dir)

    assert not _in_store(_blob_sha(big))
    assert _in_store(_blob_sha(work_dir / "small.txt"))
    assert _tree_paths(work_dir, mgr) == {"small.txt"}


def test_file_grown_past_cap_is_dropped_and_not_hashed(mgr, work_dir):
    grew = work_dir / "grew.txt"
    grew.write_text("tiny\n")
    _checkpoint(mgr, work_dir)
    assert "grew.txt" in _tree_paths(work_dir, mgr)

    grew.write_bytes(os.urandom(2 * MIB))
    (work_dir / "small.txt").write_text("changed\n")
    _checkpoint(mgr, work_dir)

    assert "grew.txt" not in _tree_paths(work_dir, mgr)
    assert not _in_store(_blob_sha(grew))


def test_diff_reports_oversize_without_hashing(mgr, work_dir):
    base = _checkpoint(mgr, work_dir)
    big = work_dir / "big.bin"
    big.write_bytes(os.urandom(2 * MIB))
    (work_dir / "small.txt").write_text("edited\n")

    result = mgr.diff(str(work_dir), base)

    assert result["success"] is True
    assert result["skipped_oversize"] == ["big.bin"]
    assert "not compared" in result["stat"]
    assert "small.txt" in result["diff"]
    assert "big.bin" not in result["diff"]
    assert not _in_store(_blob_sha(big))


def test_safe_restore_plan_never_hashes_oversize(mgr, work_dir):
    corpus = work_dir / "corpus.jsonl"
    corpus.write_bytes(b"original\n" + b"x" * (2 * MIB))
    base = _checkpoint(mgr, work_dir)

    corpus.write_bytes(b"agent appended\n" + os.urandom(2 * MIB))
    mgr.record_agent_write(str(corpus))

    result = mgr.restore(str(work_dir), base, safe=True)

    assert result["success"] is True
    assert corpus.read_bytes().startswith(b"agent appended")
    assert "corpus.jsonl" in result.get("skipped_oversize", [])
    assert not _in_store(_blob_sha(corpus))


def test_pathspec_magic_names_are_matched_literally(mgr, work_dir):
    odd = work_dir / ":odd name*.bin"
    odd.write_bytes(os.urandom(2 * MIB))
    (work_dir / "a b[1].txt").write_text("brackets\n")
    (work_dir / "*.bin").write_text("star\n")

    _checkpoint(mgr, work_dir)

    assert not _in_store(_blob_sha(odd))
    assert {"small.txt", "a b[1].txt", "*.bin"} <= _tree_paths(work_dir, mgr)
    assert ":odd name*.bin" not in _tree_paths(work_dir, mgr)


def test_inherited_literal_pathspecs_does_not_defeat_the_exclusion(mgr, work_dir, monkeypatch):
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    big = work_dir / "big.bin"
    big.write_bytes(os.urandom(2 * MIB))

    _checkpoint(mgr, work_dir)

    assert _tree_paths(work_dir, mgr) == {"small.txt"}
    assert not _in_store(_blob_sha(big))


def test_no_oversize_keeps_plain_add(mgr, work_dir, monkeypatch):
    calls = []
    original = checkpoints._run_git

    def spy(args, *rest, **kwargs):
        calls.append(list(args))
        return original(args, *rest, **kwargs)

    monkeypatch.setattr(checkpoints, "_run_git", spy)
    _checkpoint(mgr, work_dir)

    adds = [c for c in calls if c and c[0] == "add"]
    assert adds == [["add", "-A"]]
