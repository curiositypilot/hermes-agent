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


class TestEnvVarTargets:
    """``$VAR`` in cd / ``git -C`` targets resolves like the shell would (intake t_cdc954c3): a
    nested repo under ``$GUARDTMP`` is not the cwd's checkout, and ``$ROOTVAR`` names the root."""

    @pytest.fixture
    def env(self, shared, tmp_path, monkeypatch):
        root, _, workspace = shared
        scratch = root / "cache" / "scratch"
        scratch.mkdir(parents=True)
        repo = scratch / "repo"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "--allow-empty", "-m", "init"], check=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach",
                        str(scratch / "wt")], check=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("GUARDTMP", str(scratch))
        monkeypatch.setenv("ROOTVAR", str(root))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("NOPE_UNSET", raising=False)
        return root, workspace, outside

    @pytest.mark.parametrize("command", [
        "cd $GUARDTMP/wt && git checkout -q -- .",
        "cd ${GUARDTMP}/wt && git checkout -- .",
        "git -C $GUARDTMP/wt restore --worktree -- x",
        'cd "$GUARDTMP/wt" && git reset --hard',
    ])
    def test_nested_linked_worktree_not_blocked(self, env, command):
        root, workspace, _ = env
        for cwd in (root, workspace):
            assert not _blocked(command, cwd, [root]), (command, cwd)

    @pytest.mark.parametrize("command", [
        "cd $ROOTVAR && git checkout -- .",
        "git -C $ROOTVAR checkout -- .",
        "cd ${ROOTVAR}/scripts && git restore .",
        'git -C "$ROOTVAR" reset --hard',
        "cd $HOME/home && git checkout -- .",
    ])
    def test_root_named_by_var_blocked_from_outside(self, env, command):
        root, _, outside = env
        assert _blocked(command, outside, [root]), command

    def test_unknown_var_fails_closed_for_mutating_git(self, env):
        """An unset name cannot be resolved, so a mutating git there is refused from any cwd."""
        root, _, outside = env
        command = "cd $NOPE_UNSET/x && git checkout -- ."
        assert _blocked(command, root, [root])
        assert _blocked(command, outside, [root])

    def test_read_only_via_var_not_blocked(self, env):
        root, _, outside = env
        assert not _blocked("git -C $ROOTVAR status", outside, [root])


class TestShellVariables:
    """Variables the shell knows but the guard process does not (intake t_b4c67044): assigned in
    the same command, or exported in an earlier call (the session snapshot, ``base_vars``)."""

    @pytest.fixture
    def env(self, shared, tmp_path, monkeypatch):
        root, _, _ = shared
        scratch = root / "cache" / "scratch"
        scratch.mkdir(parents=True)
        repo = scratch / "repo"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "--allow-empty", "-m", "init"], check=True)
        worktree = (scratch / "wt").resolve()
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach",
                        str(worktree)], check=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))  # $HOME/home and ~/home name the root
        for name in ("R", "WT", "NOPE_UNSET"):
            monkeypatch.delenv(name, raising=False)
        return root, worktree, outside

    @pytest.mark.parametrize("command", [
        "R=$HOME/home; cd $R && git checkout -- .",
        "R=~/home && git -C $R reset --hard",
        'export R=$HOME/home; cd "$R" && git checkout -- .',
        'declare -x R={root}; git -C "$R" restore .',
        "R={root}; (cd $R && git checkout -- .)",
        "A=$HOME B=$A/home; cd $B && git stash",
        "cd $R && git checkout -- .",  # R exported in an earlier call, no snapshot: unknown
        "cd $(git rev-parse --show-toplevel) && git reset --hard",
        'R="$(pwd)"; git -C "$R" checkout -- .',
    ])
    def test_blocked_from_outside(self, env, command):
        root, _, outside = env
        assert _blocked(command.format(root=root), outside, [root]), command

    def test_snapshot_export_names_root(self, env):
        root, _, outside = env
        hit, msg = detect_shared_checkout_git_mutation(
            "cd $R && git checkout -- .", str(outside), [root], base_vars={"R": str(root)})
        assert hit and msg is not None and "cannot resolve" not in msg

    def test_own_assignment_beats_snapshot(self, env):
        root, worktree, _ = env
        command = f"R={worktree}; cd $R && git checkout -q -- ."
        hit, _ = detect_shared_checkout_git_mutation(
            command, str(root), [root], base_vars={"R": str(root)})
        assert not hit

    @pytest.mark.parametrize("command, cwd", [
        ("WT={wt}; cd $WT && git checkout -q -- .", "root"),
        ("( R={root} ); cd $R && git status", "outside"),
        ("R={wt}; ( R={root} ); cd $R && git checkout -q -- .", "root"),
        ("R={root} true; git -C $R status", "outside"),
        ("R={wt}; R={root} | true; cd $R && git checkout -q -- .", "root"),  # pipeline subshell
        ("cd $NOPE_UNSET/x && git log", "outside"),
        ("R={root} git -C {wt} checkout -- .", "outside"),
        ("cd $(git rev-parse --show-toplevel) && git diff", "outside"),
    ])
    def test_not_blocked(self, env, command, cwd):
        root, worktree, outside = env
        where = {"root": root, "outside": outside}[cwd]
        assert not _blocked(command.format(root=root, wt=worktree), where, [root]), command

    def test_unknown_target_message_names_the_word(self, env):
        root, _, outside = env
        hit, msg = detect_shared_checkout_git_mutation(
            "cd $NOPE_UNSET && git checkout -- .", str(outside), [root])
        assert hit and msg is not None and "$NOPE_UNSET" in msg and "literal absolute path" in msg

    def test_unset_drops_snapshot_value(self, env):
        root, worktree, _ = env
        hit, msg = detect_shared_checkout_git_mutation(
            "unset R; cd $R && git checkout -- .", str(worktree), [root],
            base_vars={"R": str(worktree)})
        assert hit and msg is not None and "$R" in msg

    @pytest.mark.parametrize("command", [
        "cd $NOPE_UNSET && git -C {outside} status",
        "cd $NOPE_UNSET && cd {outside} && git checkout -- .",
        "cd $NOPE_UNSET && cd {wt} && git checkout -q -- .",
    ])
    def test_absolute_target_escapes_unresolved_cwd(self, env, command):
        """An absolute cd / -C word after an unresolved cwd resolves from ``/`` (no TypeError)."""
        root, worktree, outside = env
        command = command.format(outside=outside, wt=worktree)
        assert not _blocked(command, outside, [root]), command

    def test_absolute_root_after_unresolved_cwd_blocked(self, env):
        root, _, outside = env
        assert _blocked(f"cd $NOPE_UNSET && git -C {root} checkout -- .", outside, [root])


class TestSnapshotWiring:
    """``shared_checkout_block`` reads exports from the session snapshot (``env._snapshot_path``)."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        home.mkdir()
        subprocess.run(["git", "init", "-q", str(home)], check=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("R", raising=False)
        outside = tmp_path / "outside"
        outside.mkdir()
        return home.resolve(), outside

    @staticmethod
    def _block(command, outside, snapshot):
        from types import SimpleNamespace

        from tools.terminal_tool_guards import shared_checkout_block

        env = SimpleNamespace(_snapshot_ready=True, _snapshot_path=str(snapshot))
        return shared_checkout_block(command=command, cwd=str(outside), workdir=str(outside),
                                     session_key="snapshot-test", env=env)

    def test_snapshot_export_blocks(self, home, tmp_path):
        root, outside = home
        snapshot = tmp_path / "hermes-snap-x.sh"
        snapshot.write_text(f'declare -x OTHER="a\\$b"\ndeclare -x R="{root}"\ndeclare -x EMPTY\n')
        blocked = self._block("cd $R && git checkout -- .", outside, snapshot)
        assert blocked and json.loads(blocked)["status"] == "blocked"
        assert "cannot resolve" not in json.loads(blocked)["error"]

    def test_missing_snapshot_still_fails_closed(self, home, tmp_path):
        _, outside = home
        missing = tmp_path / "absent.sh"
        blocked = self._block("cd $R && git checkout -- .", outside, missing)
        assert blocked and "$R" in json.loads(blocked)["error"]
        assert self._block("cd $R && git status", outside, missing) is None

    def test_snapshot_parser_unescapes_values(self, tmp_path):
        from types import SimpleNamespace

        from tools.terminal_tool_guards import _snapshot_exports

        snapshot = tmp_path / "snap.sh"
        snapshot.write_text('declare -x A="x\\$y\\"z"\ndeclare -x B="multi\nline"\n'
                            'declare -rx C="/c"\nexport D="/d"\nnot a line\n')
        exports = _snapshot_exports(SimpleNamespace(_snapshot_ready=True, _snapshot_path=str(snapshot)))
        assert exports == {"A": 'x$y"z', "C": "/c", "D": "/d"}
        assert _snapshot_exports(SimpleNamespace(_snapshot_ready=False, _snapshot_path=str(snapshot))) is None


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
