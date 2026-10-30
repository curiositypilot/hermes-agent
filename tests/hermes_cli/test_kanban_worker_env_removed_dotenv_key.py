"""A key dropped from the launch ``.env`` does not reach the next spawned Kanban worker.

Regression for t_8d6c7d5a: the dispatcher builds every worker env from the gateway's ``os.environ``.
A key deleted from ``.env`` stayed there until a restart, so a dead ``ANTHROPIC_API_KEY`` reached every
worker and beat the OAuth pool. The spawn now sweeps keys the launch ``.env`` no longer defines.
"""

from __future__ import annotations

import os

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.env_loader import load_hermes_dotenv


def _spawn_env(monkeypatch, tmp_path) -> dict:
    captured: dict = {}

    class _Proc:
        pid = 4321

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")
    task = kb.Task(
        id="t_8d6c7d5a", title="replay", body=None, assignee="default", status="in_progress", priority=0,
        created_by=None, created_at=0, started_at=None, completed_at=None, workspace_kind="scratch",
        workspace_path=None, claim_lock=None, claim_expires=None, tenant=None,
    )
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    kbd._default_spawn(task, str(workspace))
    return captured["env"]


def test_worker_env_lacks_key_removed_from_launch_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / "hermes" / ".env"
    env_file.parent.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(env_file.parent))
    monkeypatch.delenv("DROPPED_SETTING_VALUE", raising=False)
    env_file.write_text("DROPPED_SETTING_VALUE=stale\n", encoding="utf-8")
    load_hermes_dotenv(load_external_secrets=False)
    assert _spawn_env(monkeypatch, tmp_path).get("DROPPED_SETTING_VALUE") == "stale"

    env_file.write_text("\n", encoding="utf-8")  # no reload between the edit and the next spawn

    assert "DROPPED_SETTING_VALUE" not in _spawn_env(monkeypatch, tmp_path)
    assert "DROPPED_SETTING_VALUE" not in os.environ
