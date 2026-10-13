"""``kanban.worker_memory_max_mb`` reaches the Kanban worker's systemd scope (intake t_058789f6).

The default 4 GiB scope cap OOM-killed workers running npm/vitest for apps/desktop on a host with
>100 GB free; the board-wide knob lifts it for Kanban workers only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_run_scopes import _restart_safe_worker_argv

GiB = 1024 * 1024 * 1024


def _task(workspace: Path, run_id: int | None = 23) -> kb.Task:
    return kb.Task(
        id="t_memory_max", title="desktop typecheck", body=None, assignee="coder",
        status="running", priority=0, created_by="test", created_at=1, started_at=1,
        completed_at=None, workspace_kind="dir", workspace_path=str(workspace),
        claim_lock="host:dispatcher", claim_expires=999, tenant=None, current_run_id=run_id,
    )


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / ".hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "profiles" / "coder" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root


def _write_config(home: Path, value: str | None) -> None:
    body = "{}\n" if value is None else f"kanban:\n  worker_memory_max_mb: {value}\n"
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("12288", 12884901888), (None, None), ('"abc"', None), ("0", None), ("-5", None)],
)
def test_kanban_worker_memory_max_from_config(
    home: Path, monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int | None,
) -> None:
    _write_config(home, value)
    calls: list[dict] = []

    def fake_child_argv(command, **kwargs):
        calls.append(kwargs)
        from tools.process_registry import GatewayChildDispatch

        return GatewayChildDispatch("in_process", command)

    monkeypatch.setattr("tools.process_registry.restart_safe_gateway_child_argv", fake_child_argv)

    _restart_safe_worker_argv(_task(home), ["hermes"])
    _restart_safe_worker_argv(_task(home, run_id=None), ["hermes"])

    assert [call["memory_max_bytes"] for call in calls] == [expected, expected]


@pytest.mark.platforms("linux")
def test_managed_gateway_worker_scope_carries_configured_memory_max(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the dispatcher spawn: 12 GiB on a 64 GiB host, past the 4 GiB default."""
    _write_config(home, "12288")
    workspace = home.parent / "ws"
    workspace.mkdir()
    captured: list[str] = []

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **_kw: captured.extend(cmd) or FakeProc())
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("tools.process_registry_memory.gateway_cgroup_memory_max", lambda _floor: None)
    monkeypatch.setattr(
        "tools.process_registry_memory.physical_memory_bytes", lambda: 64 * GiB,
    )
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

    assert kbd._default_spawn(_task(workspace), str(workspace)) == 4242
    assert f"MemoryMax={12 * GiB}" in captured
