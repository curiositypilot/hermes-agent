"""Nested ``<repo>/.worktrees/`` stays out of the main checkout's ``git status``.

Kanban and desktop worktrees live under ``<repo>/.worktrees/<name>``; without an ignore
entry the main checkout reports ``?? .worktrees/`` while any card runs, breaking every
clean-status check. ``ensure_worktrees_excluded`` writes the repo-local ``info/exclude``
(untracked, so it cannot dirty the checkout itself).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import web_git
from hermes_cli.worktree_ops import ensure_worktrees_excluded


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _exclude(repo: Path) -> Path:
    return repo / ".git" / "info" / "exclude"


def _exclude_count(repo: Path) -> int:
    path = _exclude(repo)
    if not path.exists():
        return 0
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()].count(".worktrees/")


def test_anchored_worktree_excludes_dir(tmp_path):
    repo = _make_repo(tmp_path)

    target, _ = kbw._anchored_worktree(repo, "t_x", "wt/t_x")

    assert target.is_dir()
    assert _git(repo, "status", "--porcelain") == ""
    assert _exclude_count(repo) == 1


def test_exclude_idempotent(tmp_path):
    repo = _make_repo(tmp_path)

    ensure_worktrees_excluded(repo)
    ensure_worktrees_excluded(repo)

    assert _exclude_count(repo) == 1


def test_exclude_not_duplicated_when_gitignore_negates(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text("!.worktrees/\n", encoding="utf-8")

    ensure_worktrees_excluded(repo)
    ensure_worktrees_excluded(repo)

    assert _exclude_count(repo) == 1


def test_already_gitignored_untouched(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text(".worktrees/\n", encoding="utf-8")
    exclude = _exclude(repo)
    before = exclude.read_bytes() if exclude.exists() else None

    ensure_worktrees_excluded(repo)

    assert (exclude.read_bytes() if exclude.exists() else None) == before


def test_linked_worktree_root_writes_shared_exclude(tmp_path):
    repo = _make_repo(tmp_path)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", str(linked), "-b", "side", "HEAD")

    ensure_worktrees_excluded(linked)

    assert _exclude_count(repo) == 1
    (linked / ".worktrees").mkdir()
    (linked / ".worktrees" / "f").write_text("x", encoding="utf-8")
    assert _git(linked, "status", "--porcelain") == ""


def test_helper_fail_soft(tmp_path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    ensure_worktrees_excluded(plain)
    ensure_worktrees_excluded(tmp_path / "missing")

    assert not (plain / ".git").exists()


def test_web_git_worktree_add_excludes_dir(tmp_path):
    repo = _make_repo(tmp_path)

    result = web_git.worktree_add(str(repo), {"name": "feature"})

    assert Path(result["path"]).is_dir()
    assert _git(repo, "status", "--porcelain") == ""
    assert _exclude_count(repo) == 1
