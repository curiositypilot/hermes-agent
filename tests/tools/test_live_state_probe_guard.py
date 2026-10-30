"""Kanban-worker probes of live cron writers are refused before they run (t_90c6fa5e).

Regression for t_90c6fa5e / t_93b0fd0a: workers ran ``hermes-python -c "create_job(...)"``
against the live HERMES_HOME and the gateway ticker fired the junk jobs. The replayed
incident commands go through the terminal's real pre-exec chain (``_pre_exec_block``) and
``execute_code``, so a guard that is not wired in fails these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.code_execution_tool as cet
from tools.terminal_tool import _pre_exec_block, _Rejected

INCIDENTS = json.loads(
    (Path(__file__).parent / "fixtures" / "live_state_probe_incidents.json").read_text(encoding="utf-8")
)


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")


def _refusal(command: str, cwd: Path) -> str | None:
    """The terminal pre-exec chain's refusal text for *command*, or None when it may run."""
    try:
        _pre_exec_block(command, env=None, env_type="local", cwd=str(cwd), workdir=None,
                        session_key="", task_id=None)
    except _Rejected as rejected:
        return json.loads(rejected.args[0])["error"]
    return None


@pytest.mark.parametrize("incident", INCIDENTS, ids=lambda i: str(i["message_id"]))
def test_incident_commands_are_blocked_in_a_worker(incident, worker, tmp_path):
    error = _refusal(incident["command"], tmp_path)
    assert error is not None and "home-probe" in error


@pytest.mark.parametrize("incident", INCIDENTS, ids=lambda i: str(i["message_id"]))
def test_incident_commands_pass_outside_a_worker(incident, monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _refusal(incident["command"], tmp_path) is None


MUST_BLOCK = [
    "python3 -c \"import cron.jobs as j; j.create_job(prompt='x', schedule='every 1h')\"",
    "~/.hermes/scripts/hermes-python -c \"from tools.cronjob_tools import cronjob\"",
    "python3 - <<'EOF'\nfrom hermes_cli.cron import cron_run\nEOF",
    "python3 <<< 'import os, cron.scheduler'",
    "python3 -c 'from cron import jobs; jobs.create_job(prompt=\"x\", schedule=\"every 1h\")'",
    "python3 -c 'from tools import cronjob_tools as ct'",
    "python3 -c 'from hermes_cli import cron'",
    "python3 -c 'from hermes_cli import (\n    cli_commands_mixin,\n)'",
    "python3 -c 'import importlib; importlib.import_module(\"cron.jobs\")'",
    "env FOO=1 timeout 30 python3 -Ic 'import cron.jobs'",
    "uv run --with rich python -c 'import cron.jobs'",
    "bash -c \"python3 -c 'import cron.jobs'\"",
    # Chained after a probe helper: only the helper's own segment is exempt.
    "python3 -c \"from cron.jobs import create_job\" && ~/.hermes/scripts/home-probe -c 'print(1)'",
    "~/.hermes/scripts/home-probe -c 'print(1)'; python3 -c 'import cron.jobs'",
    # Code piped into a stdin-reading interpreter from an earlier pipeline segment.
    "echo 'import cron.jobs' | python3 -",
    "echo import cron.jobs | python3",
    "printf 'import os\\nfrom cron.jobs import create_job\\n' | python3 -",
    "cat <<'EOF' | python3 -\nimport cron.jobs\nEOF",
    "(echo 'import os'; echo 'import cron.scheduler') | python3 -",
    # The cron package re-exports the cron.jobs writers and serves tick lazily.
    "python3 -c \"from cron import create_job; create_job(prompt='x', schedule='every 1h')\"",
    "python3 -c \"from cron import (get_job, remove_job as rm)\"",
    "python3 -c \"import cron; cron.create_job(prompt='x', schedule='every 1h')\"",
    "python3 -c 'import os, cron as c'",
    "python3 -c 'from cron import *'",
    "python3 -c 'from cron import tick; tick()'",
    "python3 -c 'import importlib; importlib.import_module(\"cron\").create_job()'",
    # Shell -c inside a flag cluster.
    "bash -lc \"python3 -c 'import cron.jobs'\"",
    "sh -ec 'python3 -c \"from cron import create_job\"'",
    "bash -o pipefail -c \"python3 -c 'import cron.jobs'\"",
]


@pytest.mark.parametrize("command", MUST_BLOCK)
def test_must_block(command, worker, tmp_path):
    assert _refusal(command, tmp_path) is not None


def test_script_argument_is_read(worker, tmp_path):
    (tmp_path / "probe.py").write_text("from cron.jobs import create_job\ncreate_job(prompt='x')\n")
    assert _refusal("python3 probe.py", tmp_path) is not None
    assert _refusal(f"cd {tmp_path} && python3 probe.py --flag", tmp_path.parent) is not None
    assert _refusal("python3 < probe.py", tmp_path) is not None
    assert _refusal("cat probe.py | python3 -", tmp_path) is not None
    (tmp_path / "ok.py").write_text("print(1)\n")
    assert _refusal("python3 ok.py", tmp_path) is None


MUST_PASS = [
    "~/.hermes/scripts/home-probe -c 'from cron.jobs import create_job; create_job(prompt=\"x\")'",
    "~/.hermes/scripts/kanban-probe -c 'import cron.jobs'",
    "grep -n \"from cron.jobs import create_job\" cron/jobs.py",
    "rg 'import cron.jobs'",
    "python -m pytest tests/cron/test_jobs.py",
    "python3 -c \"import cronjob_helpers\"",
    "python3 -c \"from cron.lifecycle_guard import x\"",
    "python3 -c 'from cron import lifecycle_guard'",
    "python3 -c 'import cron.lifecycle_guard'",
    "python3 -c 'from cron import list_jobs; print(list_jobs())'",
    "bash -lc 'python3 -c \"print(1)\"'",
    "cat ~/.hermes/cron/jobs.json",
    "hermes cron list",
    "echo 'python3 -c \"import cron.jobs\"'",
    # Piped text that is data, not code: a -c/-m interpreter, a non-interpreter consumer,
    # or a pipeline the separator already ended.
    "grep -rn 'import cron.jobs' . | python3 -c 'import sys; print(len(sys.stdin.read()))'",
    "echo 'import cron.jobs' | grep cron",
    "echo 'import cron.jobs' && python3 -c 'print(1)'",
    "cat ~/.hermes/cron/jobs.json | python3 -m json.tool",
]


@pytest.mark.parametrize("command", MUST_PASS)
def test_must_pass(command, worker, tmp_path):
    assert _refusal(command, tmp_path) is None


def test_execute_code_refuses_a_cron_writer_import_without_spawning(worker, monkeypatch):
    from tools import code_kernel

    ran: list[str] = []
    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(code_kernel, "execute_in_session_kernel",
                        lambda code, **_kw: ran.append(code) or json.dumps({"status": "success"}))
    import tools.terminal_tool as terminal_module
    monkeypatch.setattr(terminal_module, "_get_env_config", lambda: {"env_type": "local"})

    result = json.loads(cet.execute_code("from cron.jobs import create_job"))
    assert "home-probe" in result["error"]
    assert ran == []

    assert "error" not in json.loads(cet.execute_code("print(1)"))
    assert ran == ["print(1)"]
