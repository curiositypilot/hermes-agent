"""Foreign-writer git guard: refuse `git add` / `git commit` of files a still-running delegated
worker (not the caller) is writing, in any checkout.

Regression for intake t_7cbec833: a second session ran `git add scripts/buy-usdc.sh
scripts/onboard.sh && git commit ... && git push` in ~/src/caravel while delegated worker
deleg_ebcedfcc was still writing both files, and pushed untested code. Real temp git repos and a
real live manifest under the test HERMES_HOME; only the live-manifest content is fabricated.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from tools.delegation_live_log import live_transcript_root
from tools.terminal_tool_guards import foreign_writer_block

WORKER = "subagent-0-ebcedfcc"
CALLER = "other-session-task"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "caravel"
    (root / "scripts").mkdir(parents=True)
    (root / "app").mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "app" / "README.md").write_text("app\n", encoding="utf-8")
    (root / "scripts" / "deploy.sh").write_text("echo deploy\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root.resolve()


def _write_manifest(paths, *, deleg_id="deleg_ebcedfcc", task_id=WORKER, completed=None):
    from gateway.status import get_process_start_time
    d = live_transcript_root() / deleg_id
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "delegation_id": deleg_id, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(), "pid_start": get_process_start_time(os.getpid()),
        "tasks": [{"index": 0, "log": str(d / "task-0.log"), "status": "running", "task_id": task_id,
                   "files_written": [str(p) for p in paths]}]}
    if completed:
        manifest["completed"] = completed
        manifest["tasks"][0]["status"] = "completed"
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _blocked(command, cwd, task_id: "str | None" = CALLER):
    return foreign_writer_block(command=command, cwd=str(cwd), workdir=None, session_key="", task_id=task_id)


@pytest.fixture
def untracked_foreign(repo):
    """The incident: the worker wrote onboard.sh and buy-usdc.sh, both still untracked."""
    onboard, buy = repo / "scripts" / "onboard.sh", repo / "scripts" / "buy-usdc.sh"
    onboard.write_text("#!/bin/sh\necho onboard\n", encoding="utf-8")
    buy.write_text("#!/bin/sh\necho buy\n", encoding="utf-8")
    _write_manifest([onboard, buy])
    return onboard, buy


class TestBlocks:
    @pytest.mark.parametrize("command", [
        "git add scripts/onboard.sh",
        "git add .",
        "git add -A",
        "git add --all",
        "git add scripts",
        "git add 'scripts/*.sh'",
        "git add :/",
    ])
    def test_add_covering_a_foreign_untracked_file(self, repo, untracked_foreign, command):
        result = _blocked(command, repo)
        assert result is not None
        payload = json.loads(result)
        assert payload["status"] == "blocked"
        assert "scripts/onboard.sh" in payload["error"] and "deleg_ebcedfcc" in payload["error"]

    def test_add_from_another_dir_via_cd(self, repo, untracked_foreign, tmp_path):
        assert _blocked(f"cd {repo} && git add scripts/onboard.sh", tmp_path)

    def test_add_via_dash_c(self, repo, untracked_foreign, tmp_path):
        assert _blocked(f"git -C {repo} add -A", tmp_path)

    def test_commit_with_foreign_file_staged(self, repo, untracked_foreign):
        _git(repo, "add", "scripts/onboard.sh")
        assert _blocked("git commit -m x", repo)

    def test_commit_all_with_foreign_tracked_modification(self, repo):
        deploy = repo / "scripts" / "deploy.sh"
        deploy.write_text("echo half-done\n", encoding="utf-8")
        _write_manifest([deploy])
        assert _blocked("git commit -am x", repo)
        assert _blocked("git add -u", repo)

    def test_commit_pathspec_naming_foreign_tracked_file(self, repo):
        deploy = repo / "scripts" / "deploy.sh"
        deploy.write_text("echo half-done\n", encoding="utf-8")
        _write_manifest([deploy])
        assert _blocked("git commit -m x scripts/deploy.sh", repo)

    def test_parent_session_is_blocked_too(self, repo, untracked_foreign):
        """The delegating parent runs under its own task_id, never the child's."""
        assert _blocked("git add .", repo, task_id="parent-turn-task")
        assert _blocked("git add .", repo, task_id=None)

    def test_incident_replay_at_15_30(self, repo, untracked_foreign):
        command = ("git add scripts/buy-usdc.sh scripts/onboard.sh && git commit -m 'Add onboarding "
                   "scripts' && git push -u origin simon/app-wallet-login")
        payload = json.loads(_blocked(command, repo))
        assert "buy-usdc.sh" in payload["error"] and "onboard.sh" in payload["error"]


class TestAllows:
    def test_add_non_foreign_file_while_foreign_is_untracked(self, repo, untracked_foreign):
        (repo / "app" / "README.md").write_text("changed\n", encoding="utf-8")
        assert _blocked("git add app/README.md", repo) is None

    def test_commit_with_only_non_foreign_files_staged(self, repo, untracked_foreign):
        (repo / "app" / "README.md").write_text("changed\n", encoding="utf-8")
        _git(repo, "add", "app/README.md")
        assert _blocked("git commit -m x", repo) is None

    def test_writer_own_task_id(self, repo, untracked_foreign):
        assert _blocked("git add scripts/onboard.sh && git commit -m x", repo, task_id=WORKER) is None

    def test_after_manifest_completed(self, repo):
        onboard = repo / "scripts" / "onboard.sh"
        onboard.write_text("#!/bin/sh\n", encoding="utf-8")
        _write_manifest([onboard], completed="2026-10-25 15:40:33")
        assert _blocked("git add -A && git commit -m x", repo) is None

    def test_foreign_path_in_a_different_checkout(self, repo, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        _git(other, "init", "-q")
        stray = other / "x.sh"
        stray.write_text("x\n", encoding="utf-8")
        _write_manifest([stray])
        (repo / "app" / "new.md").write_text("n\n", encoding="utf-8")
        assert _blocked("git add -A", repo) is None

    def test_foreign_file_already_committed_by_the_worker(self, repo):
        """Nothing for git to take: the clean file is not in the way of `git add .`."""
        _write_manifest([repo / "scripts" / "deploy.sh"])
        (repo / "app" / "new.md").write_text("n\n", encoding="utf-8")
        assert _blocked("git add . && git commit -m x", repo) is None

    @pytest.mark.parametrize("command", [
        "git status", "git diff", "git diff --cached", "git log -1", "git add -n .",
        "git commit --dry-run -a", "ls scripts", "git push"])
    def test_read_only_and_unrelated_commands(self, repo, untracked_foreign, command):
        assert _blocked(command, repo) is None

    def test_unreadable_manifest_fails_open(self, repo):
        (repo / "scripts" / "onboard.sh").write_text("x\n", encoding="utf-8")
        d = live_transcript_root() / "deleg_bad"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("{", encoding="utf-8")
        assert _blocked("git add -A", repo) is None

    def test_git_failure_fails_open(self, repo, untracked_foreign, monkeypatch):
        import tools.self_repo_guard as srg

        def boom(*_a, **_k):
            raise subprocess.CalledProcessError(128, "git")
        monkeypatch.setattr(srg, "_foreign_state", boom)
        assert _blocked("git add -A", repo) is None


def test_terminal_pre_exec_chain_runs_the_guard(repo, untracked_foreign):
    """Wiring: the terminal's local pre-exec chain refuses the command before it runs."""
    from tools.terminal_tool import _pre_exec_block, _Rejected

    with pytest.raises(_Rejected):
        _pre_exec_block("git add -A", env=None, env_type="local", cwd=str(repo), workdir=None,
                        session_key="", task_id=CALLER)
    _pre_exec_block("git add -A", env=None, env_type="local", cwd=str(repo), workdir=None,
                    session_key="", task_id=WORKER)
    assert not Path(repo / ".git" / "index.lock").exists()
