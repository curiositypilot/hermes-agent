from __future__ import annotations

import pytest
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
    """The transport client reaches the assigned profile with its exact run claim."""
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
    assert captured["cmd"][-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    assert captured["env"]["HERMES_KANBAN_RUN_ID"] == "7"
    assert captured["env"]["HERMES_KANBAN_CLAIM_LOCK"] == "lock"


def test_default_spawn_does_not_pass_mutable_launch_overrides(monkeypatch, tmp_path):
    """Owner policy reads overrides from the claimed card, not executable argv."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

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

    # Mutable model strings are not launch authority; the owner reads the claimed card.
    assert captured["cmd"][-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    assert task.model_override not in captured["cmd"]

def test_pinned_spawn_env(monkeypatch, tmp_path):
    """A pinned route reaches the worker as ``HERMES_KANBAN_PINNED=1``; a tier route does
    not, even when the dispatcher's own environment carries a stale value."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_PINNED", "1")

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_routing as kr

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    envs = []

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        envs.append(dict(kwargs.get("env") or {}))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    pinned = _make_task(kb, assignee="elias")
    pinned.model_override, pinned.provider_override = "gemini-x", "antigravity"
    kr.apply_route(pinned, kr.RouteDecision("pinned", candidate=kr.TierCandidate("gemini-x", "antigravity")))
    kbd._default_spawn(pinned, str(workspace))

    tiered = _make_task(kb, assignee="elias")
    kr.apply_route(tiered, kr.RouteDecision("tier", requested_tier="S", tier="S",
                                            candidate=kr.TierCandidate("cheap-1", "prov-a")))
    kbd._default_spawn(tiered, str(workspace))

    assert envs[0].get("HERMES_KANBAN_PINNED") == "1"
    assert "HERMES_KANBAN_PINNED" not in envs[1]


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


def test_worker_toolsets_come_from_the_assignee_profile_not_the_parent_config(tmp_path, monkeypatch):
    """The worker's toolsets are frozen by the assignee profile's own gateway (its config), never
    by the dispatcher's root/active profile: a root config that only lists ``kanban`` must not
    reach a worker whose profile grants ``terminal`` and ``web``."""
    import json
    from contextlib import closing
    from types import SimpleNamespace

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy

    profile = tmp_path / "profiles" / "elias"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: str(profile))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("owned")  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board="owned")) as conn:
        task_id = kb.create_task(conn, title="t", body="b", assignee="elias",
                                 workspace_kind="dir", workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal("owner", str(profile), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(profile)), actor=actor, native_owner=True)
    params = dict(board="owned", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)

    assignee_config = {"model": {"default": "m", "provider": "custom"},
                       "platform_toolsets": {"cli": ["terminal", "web"]}, "toolsets": ["hermes-cli"]}
    policy, _ = build_kanban_policy(connection, params, assignee_config)
    assert {"terminal", "web", "kanban"} <= set(policy.toolsets)

    # Headless trim (kanban.worker_disabled_toolsets, default list when the key is absent):
    # clarify and the granola MCP server never reach the owner-run worker's policy.
    headless = {**assignee_config, "platform_toolsets": {"cli": ["terminal", "clarify", "granola"]}}
    trimmed, _ = build_kanban_policy(connection, params, headless)
    assert "terminal" in trimmed.toolsets and "kanban" in trimmed.toolsets
    assert not {"clarify", "granola"} & set(trimmed.toolsets)

    root_only_kanban = {"model": {"default": "m", "provider": "custom"}, "platform_toolsets": {"cli": ["kanban"]}}
    other_profile = tmp_path
    with pytest.raises(Exception):
        # A dispatcher acting for another profile cannot mint this worker's policy at all.
        build_kanban_policy(SimpleNamespace(authority=SimpleNamespace(profile_id=str(other_profile)),
                                            actor=actor, native_owner=True), params, root_only_kanban)
    assert json.loads(policy.kanban_json)["profile"] == "elias"


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
    """No ``kanban.worker_disabled_toolsets`` key anywhere: the built-in default still drops
    ``clarify`` AND the ``granola`` MCP server from the pin when the profile enables them, so
    a stock install never spawns the meeting-notes server inside a headless worker."""
    root, profile = _write_profile(tmp_path, ["clarify", "granola", "terminal", "web"])
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import kanban_db_dispatch as kbd

    resolved = kbd._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert {"terminal", "web"} <= set(resolved)
    assert "clarify" not in resolved
    assert "granola" not in resolved


def test_worker_disabled_defaults_agree_between_code_and_config():
    """The in-code fallback (key absent) and the registered DEFAULT_CONFIG value are the same
    list, so a profile that never wrote the key and one that merged defaults trim identically."""
    from hermes_cli.config import DEFAULT_CONFIG
    from tools.kanban_toolset_context import DEFAULT_WORKER_DISABLED_TOOLSETS, worker_disabled_toolsets

    registered = DEFAULT_CONFIG["kanban"]["worker_disabled_toolsets"]
    assert list(registered) == list(DEFAULT_WORKER_DISABLED_TOOLSETS)
    assert worker_disabled_toolsets({}) == list(DEFAULT_WORKER_DISABLED_TOOLSETS)
    assert worker_disabled_toolsets(DEFAULT_CONFIG) == list(DEFAULT_WORKER_DISABLED_TOOLSETS)
    assert {"clarify", "browser_vault", "granola"} <= set(DEFAULT_WORKER_DISABLED_TOOLSETS)


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
    assert "browser_exec" in owned and "terminal" in owned
    assert not {t for t in owned if t.startswith("browser_vault")}
    assert "clarify" not in owned

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(root / "kanban"))
    child = _select_tool_names(["browser", "terminal", "clarify"], None, quiet_mode=True)
    assert "browser_vault" in child and "clarify" in child

    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    plain = _select_tool_names(["browser", "terminal", "clarify"], None, quiet_mode=True)
    assert "browser_vault" in plain and "clarify" in plain


def test_browser_vault_toolset_is_a_subtraction_handle_only():
    """``browser_vault`` names exactly the vault members of ``browser``: disabling it must not
    touch the rest of the browser surface, and it is not a per-platform checklist entry."""
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    from toolsets import resolve_toolset

    vault = set(resolve_toolset("browser_vault", include_registry=False))
    browser = set(resolve_toolset("browser", include_registry=False))
    assert vault and vault < browser
    assert all(t.startswith("browser_vault") for t in vault)
    assert {t for t in browser if t.startswith("browser_vault")} == vault
    assert "browser_vault" not in {key for key, _, _ in CONFIGURABLE_TOOLSETS}
