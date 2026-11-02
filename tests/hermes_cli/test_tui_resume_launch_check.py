"""``hermes --tui --resume/-c`` refuses typed creation flags the session was not launched with (t_942f47e1).

Classic chat refuses them (``check_resume_policy``); the TUI used to attach on the frozen route and drop
them without a word. The launcher computes the check's inputs (``resume_check_request``), the Ink client
compares them with the attach snapshot and writes the differing keys back, and the launcher prints the
classic refusal with exit 2.
"""

from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest

from hermes_cli import gateway_chat, main_tui_launch


def _args(**overrides):
    base = {"model": None, "provider": None, "toolsets": None, "skills": None, "yolo": False,
            "no_fallback": False, "in_dir": None, "resume": "sid-1"}
    base.update(overrides)
    return Namespace(**base)


def test_no_typed_flag_means_no_check():
    assert gateway_chat.resume_check_request(_args()) is None
    assert main_tui_launch._tui_resume_check(_args(resume=None, yolo=True)) is None


def test_model_carries_the_id_the_owner_would_freeze(monkeypatch):
    monkeypatch.setattr(gateway_chat, "_resolved_launch_model", lambda requested: "resolved/" + requested["model"])
    assert gateway_chat.resume_check_request(_args(model="x")) == {
        "requested": {"model": "x"}, "model_resolved": "resolved/x"}


def test_unresolvable_model_omits_model_resolved(monkeypatch):
    def broken(requested):
        raise ValueError("no such alias")

    monkeypatch.setattr(gateway_chat, "_resolved_launch_model", broken)
    assert gateway_chat.resume_check_request(_args(model="x")) == {"requested": {"model": "x"}}


def test_creation_flags_appear_normalized(tmp_path):
    check = gateway_chat.resume_check_request(
        _args(no_fallback=True, yolo=True, skills=["a,b", "a"], in_dir=str(tmp_path)))
    assert check == {"requested": {"yolo": True, "skills": ["a", "b"], "no_fallback": True,
                                   "cwd": str(tmp_path.resolve())}}


def test_relative_in_dir_is_the_directory_the_launch_entered(monkeypatch, tmp_path):
    entered = tmp_path / "project"
    entered.mkdir()
    monkeypatch.chdir(entered)  # _apply_in_dir has already chdir'd into --in project
    check = main_tui_launch._tui_resume_check(_args(in_dir="project"))
    assert check == {"requested": {"cwd": str(entered.resolve())}}


def _launch(monkeypatch, written):
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.setenv("HERMES_TUI_RESUME_CHECK", '{"stale": true}')
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", lambda tui_dir, tui_dev: (["node", "entry.js"], Path(".")))
    monkeypatch.setattr("hermes_cli.shared_session_attach.configure_tui_attachment", lambda env, sid: None)
    summaries = []
    monkeypatch.setattr(main_tui_launch, "_print_tui_exit_summary", lambda *a: summaries.append(a))
    seen = {}

    def fake_call(argv, cwd=None, env=None):
        assert env is not None
        handoff = json.loads(env["HERMES_TUI_RESUME_CHECK"])
        seen.update(handoff=handoff, env=env)
        Path(handoff["refusal_file"]).write_text(written, encoding="utf-8")
        return 2 if written.strip("[] ") else 0

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)
    with pytest.raises(SystemExit) as exc:
        main_mod._launch_tui("sid-1", resume_check={"requested": {"model": "x"}, "model_resolved": "x"})
    return exc.value.code, seen, summaries


def test_refused_resume_prints_the_classic_refusal_and_exits_2(monkeypatch, capsys):
    code, seen, summaries = _launch(monkeypatch, '["model"]')
    assert seen["handoff"]["requested"] == {"model": "x"} and seen["handoff"]["model_resolved"] == "x"
    assert seen["env"]["HERMES_TUI_RESUME"] == "sid-1"
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("Error:")
    assert "-m/--model" in err and "/model" in err
    assert summaries == []
    assert not Path(seen["handoff"]["refusal_file"]).exists()


def test_accepted_resume_takes_the_normal_exit_path(monkeypatch, capsys):
    code, seen, summaries = _launch(monkeypatch, "")
    assert code == 0
    assert "Error:" not in capsys.readouterr().err
    assert len(summaries) == 1
    assert not Path(seen["handoff"]["refusal_file"]).exists()


def test_fresh_launch_drops_an_inherited_check(monkeypatch):
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.setenv("HERMES_TUI_RESUME_CHECK", '{"requested": {"model": "x"}, "refusal_file": "/tmp/x"}')
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", lambda tui_dir, tui_dev: (["node", "entry.js"], Path(".")))
    captured = {}
    monkeypatch.setattr(main_mod.subprocess, "call", lambda argv, cwd=None, env=None: captured.update(env=env) or 1)
    with pytest.raises(SystemExit):
        main_mod._launch_tui(None, resume_check={"requested": {"model": "x"}})
    assert "HERMES_TUI_RESUME_CHECK" not in captured["env"]
