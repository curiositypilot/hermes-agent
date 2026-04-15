import sys
from types import SimpleNamespace


def test_top_level_skills_flag_defaults_to_chat(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_chat(args):
        captured["skills"] = args.skills
        captured["command"] = args.command

    monkeypatch.setattr(main_mod, "cmd_chat", fake_cmd_chat)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "-s", "hermes-agent-dev,github-auth"],
    )

    main_mod.main()

    assert captured == {
        "skills": ["hermes-agent-dev,github-auth"],
        "command": None,
    }


def test_chat_subcommand_accepts_skills_flag(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_chat(args):
        captured["skills"] = args.skills
        captured["query"] = args.query

    monkeypatch.setattr(main_mod, "cmd_chat", fake_cmd_chat)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "chat", "-s", "github-auth", "-q", "hello"],
    )

    main_mod.main()

    assert captured == {
        "skills": ["github-auth"],
        "query": "hello",
    }


def test_continue_worktree_and_skills_flags_work_together(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_chat(args):
        captured["continue_last"] = args.continue_last
        captured["worktree"] = args.worktree
        captured["skills"] = args.skills
        captured["command"] = args.command

    monkeypatch.setattr(main_mod, "cmd_chat", fake_cmd_chat)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "-c", "-w", "-s", "hermes-agent-dev"],
    )

    main_mod.main()

    assert captured == {
        "continue_last": True,
        "worktree": True,
        "skills": ["hermes-agent-dev"],
        "command": "chat",
    }


def test_main_skips_bundled_skill_auto_sync_by_default(monkeypatch):
    import hermes_cli.main as main_mod

    sync_calls = []
    cli_calls = []

    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(
        "tools.skills_sync.maybe_auto_sync_bundled_skills",
        lambda quiet=True, config=None: sync_calls.append((quiet, config)),
    )
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(main=lambda **kwargs: cli_calls.append(kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "chat", "-q", "hello"],
    )

    main_mod.main()

    assert sync_calls == [(True, None)]
    assert cli_calls == [{"verbose": False, "quiet": False, "query": "hello", "worktree": False, "checkpoints": False, "pass_session_id": False}]
