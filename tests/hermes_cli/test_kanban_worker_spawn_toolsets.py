from __future__ import annotations

import subprocess


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """Manual profile assignment should keep that profile's CLI tools.

    Regression guard for dispatcher-spawned workers that boot with
    HERMES_KANBAN_TASK: the worker must not collapse to only kanban lifecycle
    tools when the assigned profile's top-level ``toolsets`` is the default
    composite. The spawned CLI gets an explicit --toolsets pin resolved from
    platform_toolsets.cli; model_tools appends task-scoped kanban tools later.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - clarify
    - code_execution
    - delegation
    - file
    - memory
    - session_search
    - skills
    - terminal
    - web
toolsets:
  - hermes-cli
agent:
  disabled_toolsets: []
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kbd._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert "--toolsets" in captured["cmd"]
    pinned = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    for required in ("terminal", "web", "file", "skills", "code_execution", "delegation"):
        assert required in pinned


def test_default_spawn_model_override_survives_real_cli_parse(monkeypatch, tmp_path):
    """The dispatcher's pre-``chat`` model flag must reach ``args.model``.

    This is an integration contract between Kanban's worker argv builder and
    the real CLI parser. A parser default once erased the explicit override,
    silently sending the worker to its profile default or fallback instead.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="elias")
    task.model_override = "gpt-5.6-sol"
    kbd._default_spawn(task, str(workspace))

    parser, _subparsers, _chat_parser = build_top_level_parser()
    # Profile selection is attached by the outer CLI bootstrap rather than
    # build_top_level_parser(); remove that already-validated prefix and parse
    # the worker flags/subcommand through the real shared parser.
    assert captured["cmd"][1:3] == ["-p", "elias"]
    args = parser.parse_args(captured["cmd"][3:])

    assert args.command == "chat"
    assert args.model == "gpt-5.6-sol"
    assert args.query == "work kanban task t_spawn_tools"


def test_default_spawn_resolves_env_passthrough_under_multiplex(monkeypatch, tmp_path):
    """Under multiplex a worker spawn with ``terminal.env_passthrough`` configured must
    forward the ASSIGNEE profile's own value, never crash on an unscoped read or leak the
    dispatcher's ambient os.environ (#109494).
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "terminal:\n  env_passthrough:\n    - MY_PASSTHROUGH_VAR\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    profile.joinpath(".env").write_text("MY_PASSTHROUGH_VAR=elias-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("MY_PASSTHROUGH_VAR", "dispatcher-value")

    from agent.secret_scope import set_multiplex_active
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4243

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    set_multiplex_active(True)
    try:
        pid = kbd._default_spawn(_make_task(kb, assignee="elias"), str(workspace))
    finally:
        set_multiplex_active(False)

    assert pid == 4243
    # The assignee's own scoped value, not the dispatcher's ambient os.environ one.
    assert captured["env"].get("MY_PASSTHROUGH_VAR") == "elias-value"


def test_resolve_worker_cli_toolsets_uses_profile_home_not_parent_config(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("platform_toolsets:\n  cli:\n    - kanban\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - terminal
    - web
toolsets:
  - hermes-cli
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    resolved = kbd._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert "terminal" in resolved
    assert "web" in resolved
    # Opt-in is no longer inferred for ordinary chats. The dispatcher-owned
    # worker gets lifecycle tools at schema assembly, independently of the
    # assignee's saved chat selection.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_spawn_tools")
    from model_tools import get_tool_definitions
    names = {t["function"]["name"] for t in get_tool_definitions(resolved, quiet_mode=True, skip_tool_search_assembly=True)}
    assert "kanban_complete" in names
    assert "kanban_list" not in names
    assert resolved != ["kanban"]


def _write_profile(tmp_path, cli_toolsets: list[str], kanban_yaml: str = "") -> tuple:
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    body = "platform_toolsets:\n  cli:\n" + "".join(f"    - {t}\n" for t in cli_toolsets)
    profile.joinpath("config.yaml").write_text(body + kanban_yaml, encoding="utf-8")
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    return root, profile


def test_worker_pin_drops_headless_toolsets_by_default(monkeypatch, tmp_path):
    """A dispatcher-spawned worker is headless: ``clarify`` (nobody answers) and an MCP server
    named in ``kanban.worker_disabled_toolsets`` leave the ``--toolsets`` pin, so the worker
    neither sees the schema nor spawns the server. Everything else the profile enables stays."""
    root, profile = _write_profile(
        tmp_path, ["browser", "clarify", "terminal", "web", "granola"],
        "kanban:\n  worker_disabled_toolsets: [clarify, browser_vault, granola]\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db_dispatch as kbd

    resolved = kbd._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert {"browser", "terminal", "web"} <= set(resolved)
    assert "clarify" not in resolved
    assert "granola" not in resolved


def test_worker_pin_default_trim_without_config_key(monkeypatch, tmp_path):
    root, profile = _write_profile(tmp_path, ["clarify", "terminal", "web"])
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db_dispatch as kbd

    resolved = kbd._resolve_worker_cli_toolsets(str(profile))

    assert "clarify" not in resolved and {"terminal", "web"} <= set(resolved)


def test_worker_pin_explicit_empty_list_keeps_everything(monkeypatch, tmp_path):
    root, profile = _write_profile(tmp_path, ["clarify", "terminal"], "kanban:\n  worker_disabled_toolsets: []\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db_dispatch as kbd

    assert "clarify" in kbd._resolve_worker_cli_toolsets(str(profile))


def test_owned_worker_schema_strips_vault_tools_but_keeps_browser(monkeypatch, tmp_path):
    """The vault tools ride inside ``browser`` so the pin cannot drop them; schema assembly
    subtracts ``browser_vault`` for the dispatcher-owned worker only. A delegated child of the
    worker (same env, not the owner) and an ordinary CLI session keep today's set."""
    root, _profile = _write_profile(tmp_path, ["browser", "terminal"])
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_spawn_tools")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from model_tools import _select_tool_names
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

    owned = _select_tool_names(["browser", "terminal", "clarify"], None, quiet_mode=True)
    assert "browser_navigate" in owned and "terminal" in owned
    assert not {t for t in owned if t.startswith("browser_vault_")}
    assert "clarify" not in owned

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(root / "kanban"))
    child = _select_tool_names(["browser", "terminal", "clarify"], None, quiet_mode=True)
    assert "browser_vault_list" in child and "clarify" in child

    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    plain = _select_tool_names(["browser", "terminal", "clarify"], None, quiet_mode=True)
    assert "browser_vault_list" in plain and "clarify" in plain


def test_browser_vault_toolset_is_a_subtraction_handle_only():
    """``browser_vault`` names exactly the vault members of ``browser``: disabling it must not
    touch the rest of the browser surface, and it is not a per-platform checklist entry."""
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    from toolsets import resolve_toolset

    vault = set(resolve_toolset("browser_vault", include_registry=False))
    browser = set(resolve_toolset("browser", include_registry=False))
    assert vault and vault < browser
    assert all(t.startswith("browser_vault_") for t in vault)
    assert {t for t in browser if t.startswith("browser_vault_")} == vault
    assert "browser_vault" not in {key for key, _, _ in CONFIGURABLE_TOOLSETS}
