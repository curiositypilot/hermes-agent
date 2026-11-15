"""Canonical Kanban receipts retain the shared one-shot outcome semantics."""
import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


def _bound_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title="private fixture", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="private-host:owner")
        assert task is not None
        context = {"db": str(path), "task_id": task_id,
                   "run_id": task.current_run_id, "claim_lock": task.claim_lock}
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "worker_bound",
                             {"pid": os.getpid(), "claim_lock": task.claim_lock},
                             run_id=task.current_run_id)
    return path, context


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"completed": True}, 0),
        ({"failed": True}, 1),
        ({"partial": True}, 1),
        ({"completed": False}, 1),
        ({"interrupted": True}, 130),
        (None, 1),
        *[({"failed": True, "failure_reason": reason}, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
          for reason in ("rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout")],
        *[({"failed": True, "failure_reason": reason}, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE)
          for reason in ("auth", "auth_permanent", "model_not_found", "ssl_cert_verification", "upstream_blocked")],
    ],
)
def test_managed_result_persists_exact_worker_exit(tmp_path, monkeypatch, result, expected):
    path, context = _bound_claim(tmp_path, monkeypatch)
    monkeypatch.setattr(session_kanban, "_run_task_turns", lambda *_args: result)
    frame = {"policy": {"kanban_json": json.dumps(context)}}

    assert session_kanban.run_worker_turns(SimpleNamespace(), frame, []) is result
    assert session_kanban.worker_exit_code(path, context) == expected
    assert session_kanban.worker_exit_code(path, context | {"claim_lock": "foreign"}) == 1


def _credential_failures():
    from hermes_cli.auth import AuthError
    from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE
    wrapped = RuntimeError("agent init failed")
    wrapped.__cause__ = AuthError("refresh revoked", provider="nous", relogin_required=True)
    return [
        (AuthError("quota", provider="openai-codex", code=CODEX_RATE_LIMITED_CODE), kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        (AuthError("relogin", provider="nous", relogin_required=True), kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE),
        (wrapped, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE),
        (RuntimeError("unknown startup failure"), 1),
    ]


@pytest.mark.parametrize("error, expected", _credential_failures())
def test_managed_start_failure_books_the_one_shot_credential_exit(tmp_path, monkeypatch, error, expected):
    """Credentials that fail before any turn (agent construction) keep the one-shot CLI's mapping:
    a quota wall is EX_TEMPFAIL, an explicit re-login is EX_CONFIG, anything else stays 1."""
    path, context = _bound_claim(tmp_path, monkeypatch)
    frame = {"policy": {"kanban_json": json.dumps(context)}}

    session_kanban.record_start_failure(frame, error)
    assert session_kanban.worker_exit_code(path, context) == expected


def test_start_failure_books_redacted_cause_line(tmp_path, monkeypatch):
    """Incident replay (t_40dcc86f): agent construction raised AttributeError and the card saw only
    'Kanban execution unknown'. The attempt now books the exception type and message, exit still 1."""
    path, context = _bound_claim(tmp_path, monkeypatch)
    frame = {"policy": {"kanban_json": json.dumps(context)}, "api_key": None}
    error = AttributeError("'AIAgent' object has no attribute '_anthropic_api_key'. Did you mean: '_anthropic_client'?")

    session_kanban.record_start_failure(frame, error)
    result = session_kanban.worker_result(path, context)
    assert result["last_output"].startswith("managed worker start failed: AttributeError:")
    assert "_anthropic_api_key" in result["last_output"]
    assert result["exit_code"] == 1


@pytest.mark.parametrize("error, expected", _credential_failures()[:2])
def test_start_failure_cause_line_keeps_credential_exit(tmp_path, monkeypatch, error, expected):
    path, context = _bound_claim(tmp_path, monkeypatch)
    session_kanban.record_start_failure({"policy": {"kanban_json": json.dumps(context)}}, error)
    result = session_kanban.worker_result(path, context)
    assert result["exit_code"] == expected
    assert result["last_output"].startswith("managed worker start failed: AuthError:")


def test_start_failure_line_redacts_credentials(tmp_path, monkeypatch):
    path, context = _bound_claim(tmp_path, monkeypatch)
    launch_key = "launch-key-7f3a9c2e1d0b4a68"  # no vendor shape: only exact-value replacement catches it
    vendor_key = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    frame = {"policy": {"kanban_json": json.dumps(context)}, "api_key": launch_key}

    session_kanban.record_start_failure(frame, RuntimeError(f"auth with {launch_key} and\n{vendor_key} failed"))
    line = session_kanban.worker_result(path, context)["last_output"]
    assert launch_key not in line and vendor_key not in line
    assert "\n" not in line and line.startswith("managed worker start failed: RuntimeError: auth with ***")

    session_kanban.record_start_failure(frame, RuntimeError("unknown startup failure"))
    assert (session_kanban.worker_result(path, context)["last_output"]
            == "managed worker start failed: RuntimeError: unknown startup failure")

    session_kanban.record_start_failure(frame, RuntimeError("x" * 2000))
    assert len(session_kanban.worker_result(path, context)["last_output"]) == 240


def test_turn_exception_books_cause_line(tmp_path, monkeypatch):
    path, context = _bound_claim(tmp_path, monkeypatch)

    def boom(*_args):
        raise RuntimeError("boom")
    monkeypatch.setattr(session_kanban, "_run_task_turns", boom)
    frame = {"policy": {"kanban_json": json.dumps(context)}, "api_key": None}

    with pytest.raises(RuntimeError, match="boom"):
        session_kanban.run_worker_turns(SimpleNamespace(), frame, [])
    result = session_kanban.worker_result(path, context)
    assert result["last_output"] == "managed worker turn failed: RuntimeError: boom"
    assert result["exit_code"] == 1


def _execute_frame(context, api_key=None):
    return {"policy": {"kanban_json": json.dumps(context)}, "api_key": api_key}


def test_execute_failure_books_cause_line_for_missing_attachment(tmp_path, monkeypatch):
    path, context = _bound_claim(tmp_path, monkeypatch)
    session_kanban.record_execute_failure(_execute_frame(context), ValueError("managed_attachment_unavailable"))
    result = session_kanban.worker_result(path, context)
    assert result["last_output"] == "managed worker execute failed: ValueError: managed_attachment_unavailable"
    assert result["exit_code"] == 1


def test_execute_failure_replaces_booked_exit_zero_reply(tmp_path, monkeypatch):
    """Finish timeout: the turn booked its reply (exit 0) and the owner never got ``finished``."""
    path, context = _bound_claim(tmp_path, monkeypatch)
    session_kanban._record_worker_result(context, 0, "done")
    session_kanban.record_execute_failure(_execute_frame(context), ValueError("managed_finish_timeout"))
    result = session_kanban.worker_result(path, context)
    assert result["last_output"] == "managed worker execute failed: ValueError: managed_finish_timeout"
    assert result["exit_code"] == 1


def test_execute_failure_keeps_earlier_start_and_turn_cause_lines(tmp_path, monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE
    path, context = _bound_claim(tmp_path, monkeypatch)
    frame = _execute_frame(context)
    session_kanban.record_start_failure(frame, AuthError("quota", provider="openai-codex", code=CODEX_RATE_LIMITED_CODE))
    session_kanban.record_execute_failure(frame, ValueError("managed_attachment_unavailable"))
    result = session_kanban.worker_result(path, context)
    assert result["exit_code"] == kb.KANBAN_RATE_LIMIT_EXIT_CODE
    assert result["last_output"].startswith("managed worker start failed:")

    session_kanban._record_worker_result(context, 1, "managed worker turn failed: RuntimeError: boom")
    session_kanban.record_execute_failure(frame, ValueError("managed_finish_timeout"))
    result = session_kanban.worker_result(path, context)
    assert result["last_output"] == "managed worker turn failed: RuntimeError: boom"
    assert result["exit_code"] == 1


def test_execute_failure_without_kanban_context_books_nothing(tmp_path, monkeypatch):
    path, context = _bound_claim(tmp_path, monkeypatch)
    session_kanban.record_execute_failure({"policy": {}}, ValueError("managed_attachment_unavailable"))
    assert session_kanban.worker_result(path, context) == {}


def test_execute_failure_line_redacts_the_launch_key(tmp_path, monkeypatch):
    path, context = _bound_claim(tmp_path, monkeypatch)
    launch_key = "launch-key-7f3a9c2e1d0b4a68"
    session_kanban.record_execute_failure(_execute_frame(context, launch_key), RuntimeError(f"bad {launch_key}"))
    line = session_kanban.worker_result(path, context)["last_output"]
    assert launch_key not in line and line.startswith("managed worker execute failed:")


@pytest.mark.parametrize("error", [sqlite3.OperationalError("database is locked"), RuntimeError("boom"),
                                   ImportError("no module")])
def test_execute_failure_helper_never_raises(tmp_path, monkeypatch, error):
    _path, context = _bound_claim(tmp_path, monkeypatch)

    def broken(*_args):
        raise error
    monkeypatch.setattr(session_kanban, "_record_worker_result", broken)
    session_kanban.record_execute_failure(_execute_frame(context), ValueError("managed_finish_timeout"))


@pytest.mark.parametrize("policy", [None, "kanban", ["kanban_json"]])
def test_execute_failure_helper_ignores_a_non_mapping_policy(policy):
    session_kanban.record_execute_failure({"policy": policy}, ValueError("managed_attachment_unavailable"))


def test_execute_books_failure_then_reraises_the_original(monkeypatch):
    from agent import managed_worker
    seen = []

    def no_attachment(_frame):
        raise ValueError("managed_attachment_unavailable")
    monkeypatch.setattr(managed_worker, "bind_worker_policy", no_attachment)
    monkeypatch.setattr(session_kanban, "record_execute_failure", lambda frame, exc: seen.append((frame, exc)))
    frame = {"policy": {}}

    with pytest.raises(ValueError, match="managed_attachment_unavailable") as raised:
        managed_worker.execute(frame, None)
    assert seen == [(frame, raised.value)]
