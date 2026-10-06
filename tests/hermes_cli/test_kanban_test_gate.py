"""Test-command completion gate: ``completion_contract = "test:<cmd>"``.

Real SQLite + real subprocesses in a temp workspace; the contract is that a
card's terminal transition (done, review) happens only when the declared
command exits 0 in that card's workspace, and the receipt rides the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_pr_acceptance import validate_contract
from hermes_cli.kanban_test_gate import TAIL_LINES, resolve_test_command


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    repo = tmp_path / "repo"
    repo.mkdir()
    with connect() as conn:
        yield conn, repo


def _claimed(conn, repo, contract):
    tid = kb.create_task(conn, title="gated", assignee="default", workspace_kind="dir",
                         workspace_path=str(repo), completion_contract=contract)
    return tid, kb.claim_task(conn, tid).current_run_id


def _run_metadata(conn, run_id):
    row = conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()
    return json.loads(row["metadata"]) if row["metadata"] else {}


@pytest.mark.parametrize("value, expected", [
    (None, "local-only"),
    ("local-only", "local-only"),
    ("test:true", "test:true"),
    ("test:  make test  ", "test:make test"),
    ("acme/repo", "acme/repo"),
    ("https://github.com/acme/repo/pull/7", "https://github.com/acme/repo/pull/7"),
])
def test_validate_contract_accepts(value, expected):
    assert validate_contract(value) == expected


@pytest.mark.parametrize("value", ["test:", "test:   ", "test:a\nb", "tests:true", "nonsense"])
def test_validate_contract_rejects(value):
    with pytest.raises(ValueError):
        validate_contract(value)


@pytest.mark.linux_only
def test_failing_command_refuses_completion_with_tail_and_keeps_run_open(board):
    conn, repo = board
    script = "; ".join(f"echo line{i}" for i in range(60)) + "; echo boom >&2; exit 3"
    tid, run_id = _claimed(conn, repo, "test:" + script)

    assert kb.complete_task(conn, tid, summary="claims done", expected_run_id=run_id) is False

    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.current_run_id == run_id
    assert "exited 3" in task.last_failure_error and "boom" in task.last_failure_error
    receipt = _run_metadata(conn, run_id)["tests"]
    assert receipt["exit_code"] == 3 and receipt["ok"] is False
    assert receipt["cwd"] == str(repo)
    tail = receipt["tail"].splitlines()
    assert len(tail) == TAIL_LINES and tail[-1] == "boom" and "line0" not in tail
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "test_gate" in kinds and "completed" not in kinds


@pytest.mark.linux_only
def test_passing_command_runs_in_workspace_and_lands_receipt_on_closed_run(board):
    conn, repo = board
    (repo / "marker").write_text("x")
    tid, run_id = _claimed(conn, repo, "test:test -f marker && echo green")

    assert kb.complete_task(conn, tid, summary="done", metadata={"k": 1}, expected_run_id=run_id)

    task = kb.get_task(conn, tid)
    assert task.status == "done"
    meta = _run_metadata(conn, run_id)
    assert meta["k"] == 1
    assert meta["tests"]["exit_code"] == 0 and meta["tests"]["tail"] == "green"


@pytest.mark.linux_only
def test_request_review_is_gated_by_the_same_command(board):
    conn, repo = board
    tid, run_id = _claimed(conn, repo, "test:echo red; exit 1")
    ok, reason = kb.request_review(conn, tid, summary="ready", expected_run_id=run_id, with_reason=True)
    assert ok is False and "exited 1" in reason and "red" in reason
    assert kb.get_task(conn, tid).status == "running"

    tid2, run2 = _claimed(conn, repo, "test:true")
    ok, reason = kb.request_review(conn, tid2, summary="ready", expected_run_id=run2, with_reason=True)
    assert ok is True, reason
    assert kb.get_task(conn, tid2).status == "review"
    assert _run_metadata(conn, run2)["tests"]["exit_code"] == 0


@pytest.mark.linux_only
def test_kanban_complete_tool_surfaces_the_refusal_tail(board, monkeypatch):
    from tools import kanban_tools

    conn, repo = board
    tid, run_id = _claimed(conn, repo, "test:echo failing-assert; false")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setattr(kanban_tools, "_worker_run_id", lambda _tid: run_id)
    out = json.loads(kanban_tools._handle_complete({"summary": "done"}))
    assert "failing-assert" in json.dumps(out) and "Test gate refused" in json.dumps(out)
    assert kb.get_task(conn, tid).status == "running"


@pytest.mark.linux_only
def test_timed_out_command_is_killed_and_refused(tmp_path, monkeypatch):
    from hermes_cli import kanban_test_gate

    monkeypatch.setattr(kanban_test_gate, "TIMEOUT_SECONDS", 1)
    receipt = kanban_test_gate.run_test_contract("test:echo started; sleep 30", str(tmp_path))
    assert receipt["ok"] is False and receipt["exit_code"] is None
    assert receipt["duration_s"] < 15
    assert "started" in receipt["tail"] and "timed out" in receipt["tail"]


def test_resolve_test_command_order(tmp_path):
    assert resolve_test_command(tmp_path) is None
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert resolve_test_command(tmp_path) is None
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "vitest run"}}))
    assert resolve_test_command(tmp_path) == "npm test"
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    assert resolve_test_command(tmp_path).endswith("-m pytest -q")
    (tmp_path / "Makefile").write_text("build:\n\ttrue\ntest: build\n\ttrue\n")
    assert resolve_test_command(tmp_path) == "make test"
    (tmp_path / ".hermes-test").write_text("# comment\n\nscripts/run_tests.sh tests/x\n")
    assert resolve_test_command(tmp_path) == "scripts/run_tests.sh tests/x"


def test_repo_workspace_defaults_to_resolved_test_contract(board, tmp_path):
    conn, repo = board
    (repo / ".hermes-test").write_text("true\n")
    default = kb.create_task(conn, title="d", workspace_kind="dir", workspace_path=str(repo))
    explicit = kb.create_task(conn, title="e", workspace_kind="dir", workspace_path=str(repo),
                              completion_contract="local-only")
    scratch = kb.create_task(conn, title="s")
    bare = tmp_path / "bare"
    bare.mkdir()
    unresolved = kb.create_task(conn, title="u", workspace_kind="dir", workspace_path=str(bare))
    assert kb.get_task(conn, default).completion_contract == "test:true"
    assert kb.get_task(conn, explicit).completion_contract == "local-only"
    assert kb.get_task(conn, scratch).completion_contract == "local-only"
    assert kb.get_task(conn, unresolved).completion_contract == "local-only"
