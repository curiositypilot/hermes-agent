"""Shared-checkout git guard: refuse git commands that destroy other agents' uncommitted work in a
protected checkout (default HERMES_HOME), before approvals, so mode=off / yolo cannot bypass it."""

import json
import subprocess
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from tools.self_repo_guard import detect_shared_checkout_git_mutation, protected_checkout_roots


@pytest.fixture
def shared(tmp_path):
    """A protected root repo with a nested repo and a plain kanban workspace dir inside it."""
    root = tmp_path / "home"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    nested = root / "hermes-agent"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    workspace = root / "kanban" / "workspaces" / "t_x"
    workspace.mkdir(parents=True)
    (root / "scripts").mkdir()
    return root.resolve(), nested.resolve(), workspace.resolve()


def _blocked(command, cwd, roots):
    hit, _ = detect_shared_checkout_git_mutation(command, str(cwd), roots)
    return hit


class TestBlocks:
    @pytest.mark.parametrize("command", [
        "git checkout e72ff3082e315b71a48e80f7bc26b754761a1b17 -- scripts/ && git apply --check "
        "/x/kanban/workspaces/t_f0b2e3ed/impl.diff",
        "git checkout HEAD -- scripts/",
        "git checkout -- .",
        "git restore scripts/",
        "git stash",
        "git stash push -u",
        "git stash drop",
        "git stash clear",
        "git reset --hard",
        "git clean -fd",
        "git switch -f main",
    ])
    def test_mutations_at_root(self, shared, command):
        root, _, _ = shared
        assert _blocked(command, root, [root])

    def test_dash_c_root_stash(self, shared, tmp_path):
        root, _, _ = shared
        assert _blocked(f"git -C {root} stash", tmp_path, [root])

    def test_dash_c_root_from_nested_repo(self, shared):
        root, nested, _ = shared
        assert _blocked(f"git -C {root} stash", nested, [root])

    def test_cd_root_then_restore(self, shared, tmp_path):
        root, _, _ = shared
        assert _blocked(f"cd {root} && git restore .", tmp_path, [root])

    def test_from_workspace_without_git(self, shared):
        root, _, workspace = shared
        assert _blocked("git checkout HEAD -- scripts/", workspace, [root])

    def test_message_names_root_and_alternatives(self, shared):
        root, _, _ = shared
        hit, msg = detect_shared_checkout_git_mutation("git stash", str(root), [root])
        assert hit and msg is not None
        assert str(root) in msg and "uncommitted" in msg
        assert "git show <rev>:<path>" in msg and "GIT_INDEX_FILE" in msg
        assert "git worktree add --detach" in msg


class TestAllows:
    @pytest.mark.parametrize("command", [
        "git stash list",
        "git stash show -p",
        "git restore --staged f",
        "git status",
        "git diff",
        "git show HEAD:f",
        "GIT_INDEX_FILE=x git read-tree HEAD",
        "git apply --cached --check d",
        "git add -A",
        "git rm --cached f",
        "git reset --soft origin/hermes",
        "git commit -m backup",
        "git push origin hermes",
    ])
    def test_read_only_and_backup_ops_at_root(self, shared, command):
        root, _, _ = shared
        assert not _blocked(command, root, [root])

    @pytest.mark.parametrize("command", [
        "git checkout HEAD -- .", "git stash", "git stash drop", "git reset --hard", "git restore x"])
    def test_any_mutation_inside_nested_repo(self, shared, command):
        root, nested, _ = shared
        assert not _blocked(command, nested, [root])

    def test_empty_protected_roots(self, shared):
        root, _, _ = shared
        assert not _blocked("git checkout HEAD -- scripts/", root, [])

    def test_repo_outside_root(self, shared, tmp_path):
        root, _, _ = shared
        other = tmp_path / "other"
        other.mkdir()
        subprocess.run(["git", "init", "-q", str(other)], check=True)
        assert not _blocked("git stash", other, [root])


def test_incident_t_ab53aec9_commands_blocked(shared):
    """The two commands critic t_a43f4d4f ran in ~/.hermes (state.db msgs 185968/185978)."""
    root, _, _ = shared
    for command in (
        "git checkout e72ff3082e315b71a48e80f7bc26b754761a1b17 -- scripts/ && git apply --check "
        "/home/humansimulacrum/.hermes/kanban/workspaces/t_f0b2e3ed/impl.diff",
        "git checkout HEAD -- scripts/",
    ):
        assert _blocked(command, root, [root]), command


class TestProtectedRootsConfig:
    def test_unset_defaults_to_git_home(self, shared):
        root, _, _ = shared
        assert protected_checkout_roots(None, root) == [root]

    def test_unset_without_git_is_empty(self, tmp_path):
        assert protected_checkout_roots(None, tmp_path) == []

    def test_explicit_empty_disables(self, shared):
        root, _, _ = shared
        assert protected_checkout_roots([], root) == []

    def test_explicit_list_expands_tilde(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "repo").mkdir()
        assert protected_checkout_roots(["~/repo"], tmp_path / "x") == [(tmp_path / "repo").resolve()]


def _write_config(home, body):
    (home / "config.yaml").write_text(body, encoding="utf-8")


class TestConfigThroughLoader:
    """E2E via a temp HERMES_HOME and the real config loader."""

    def test_unset_key_protects_git_home(self, tmp_path, monkeypatch):
        from tools.terminal_tool_guards import _protected_checkouts

        home = tmp_path / "hermes"
        home.mkdir()
        subprocess.run(["git", "init", "-q", str(home)], check=True)
        _write_config(home, "approvals:\n  mode: 'off'\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert _protected_checkouts() == [home.resolve()]

    def test_explicit_empty_list_disables(self, tmp_path, monkeypatch):
        from tools.terminal_tool_guards import _protected_checkouts

        home = tmp_path / "hermes"
        home.mkdir()
        subprocess.run(["git", "init", "-q", str(home)], check=True)
        _write_config(home, "approvals:\n  protected_checkouts: []\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert _protected_checkouts() == []

    def test_unreadable_config_fails_closed_loudly(self, tmp_path, monkeypatch, caplog):
        import hermes_cli.config as config_mod
        from tools.terminal_tool_guards import _protected_checkouts

        home = tmp_path / "hermes"
        home.mkdir()
        subprocess.run(["git", "init", "-q", str(home)], check=True)
        monkeypatch.setenv("HERMES_HOME", str(home))

        def _broken():
            raise config_mod.InvalidUserConfigError("bad yaml")

        monkeypatch.setattr(config_mod, "load_config_readonly", _broken)
        with caplog.at_level("WARNING", logger="tools.terminal_tool_guards"):
            assert _protected_checkouts() == [home.resolve()]
        assert "config unreadable (InvalidUserConfigError: bad yaml)" in caplog.text

    def test_unexpected_loader_error_propagates(self, tmp_path, monkeypatch):
        import hermes_cli.config as config_mod
        from tools.terminal_tool_guards import _protected_checkouts

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def _bug():
            raise KeyError("loader bug")

        monkeypatch.setattr(config_mod, "load_config_readonly", _bug)
        with pytest.raises(KeyError):
            _protected_checkouts()


def _run_terminal(command, cwd, **kwargs):
    from tools.terminal_tool import terminal_tool

    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "ok", "returncode": 0}
    mock_env.cwd = str(cwd)
    config = {"env_type": "local", "timeout": 180, "cwd": str(cwd), "host_cwd": None,
              "modal_mode": "auto", "docker_image": "", "singularity_image": "",
              "modal_image": "", "daytona_image": ""}
    with ExitStack() as stack:
        stack.enter_context(patch("tools.terminal_tool._get_env_config", return_value=config))
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(patch("tools.terminal_tool._active_environments", {"default": mock_env}))
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(patch("tools.terminal_tool._session_cwd", {}))
        # Approvals approve everything (mode off / yolo): the guard must still block.
        stack.enter_context(patch("tools.terminal_tool._check_all_guards",
                                  return_value={"approved": True}))
        result = json.loads(terminal_tool(command=command, **kwargs))
    return result, mock_env


class TestTerminalWiring:
    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        home.mkdir()
        subprocess.run(["git", "init", "-q", str(home)], check=True)
        _write_config(home, "approvals:\n  mode: 'off'\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home.resolve()

    def test_blocked_despite_approvals_off_and_force(self, home):
        result, env = _run_terminal("git checkout HEAD -- scripts/", home, force=True)
        assert result["status"] == "blocked"
        assert str(home) in result["error"]
        env.execute.assert_not_called()

    def test_workspace_under_home_is_blocked(self, home):
        workspace = home / "kanban" / "workspaces" / "t_x"
        workspace.mkdir(parents=True)
        result, env = _run_terminal("git stash", workspace)
        assert result["status"] == "blocked"
        env.execute.assert_not_called()

    def test_read_only_passes(self, home):
        result, env = _run_terminal("git status", home)
        assert result.get("status") != "blocked"
        env.execute.assert_called_once()
