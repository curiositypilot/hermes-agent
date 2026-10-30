"""Tail readers of the append-mode worker log only see the current run (#88456).

Regression for intake t_1f853b01: t_b87258ac's 2026-10-30 crashes were booked with the
2026-10-27 run's text ("Claude Opus 4.6 is no longer available") because
``_worker_final_output`` cut at the newest exit summary in a tail spanning several runs;
the real cause ("Kanban owner refused") was dropped. ``_worker_log_exit_code`` had the
same flaw for the exit trailer.
"""

import contextlib
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_worker_log import current_run_section, spawn_marker_line

TASK = "t_b87258ac"

# One 2026-10-27 run as it sits in the incident log (lines 57-84): exit summary + rc=0.
_RULE = "\u2500" * 40
_OLD_RUN = (
    "Query: work kanban task t_b87258ac\n"
    "Initializing agent...\r\n"
    f"{_RULE}\r\n\r\n"
    f"\u256d\u2500 \u2624 Hermes {_RULE}\u256e\r\n"
    f"{'Claude Opus 4.6 is no longer available. Please switch to Claude Opus 5.5.' * 3}\r\n"
    f"\u2570{_RULE}\u256f\r\n"
    "\u26a0\ufe0f Kanban worker tried to exit without a terminal board call "
    "(kanban_complete/kanban_request_review/kanban_block) \u2014 nudging to finish\r\n"
    "\nResume this session with:\n"
    "  hermes --resume 20261027_09{n}53_178129\n"
    '  hermes -c "intake/L-review: cron:marketplace-watch fails #{n}"\n\n'
    "Session:        20261027_09{n}53_178129\n"
    "Duration:       11s\n"
    "Messages:       6 (3 user, 0 tool calls)\n\n"
    "[kanban-worker-exit] rc=0\n"
)
# The 2026-10-30 runs (lines 85-88 and 89-92).
_NEW_RUN = (
    "Session: local-{sid}\n"
    "Kanban owner refused: Kanban execution unknown; retained by owner\n\n"
    "[kanban-worker-exit] rc=1\n"
)


def _old_runs() -> str:
    return "".join(_OLD_RUN.replace("{n}", str(n)) for n in range(4, 12))


@pytest.fixture
def logs(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: log_dir)

    def write(text: str) -> None:
        (log_dir / f"{TASK}.log").write_text(text, encoding="utf-8")

    return write


def _marker(run_id: int) -> str:
    return spawn_marker_line(run_id).decode()


def test_current_run_section_without_marker_is_unchanged():
    assert current_run_section("a\nb\n") == "a\nb\n"


def test_current_run_section_keeps_only_text_after_last_marker():
    raw = "old\n" + _marker(1) + "mid\n" + _marker(2) + "new\nlast\n"
    assert current_run_section(raw) == "new\nlast\n"


def test_current_run_section_marker_at_end_is_empty():
    assert current_run_section("old\n" + _marker(3)) == ""
    assert current_run_section("old\n" + _marker(3).rstrip("\n")) == ""


def test_incident_replay_reports_current_run_output(logs):
    """Run 7216 died before its exit summary; its own lines must be the output."""
    logs(_old_runs() + _marker(7216) + _NEW_RUN.format(sid="3cfbc800").replace("\n[kanban-worker-exit] rc=1\n", ""))
    out = kbd._worker_final_output(TASK)
    assert "Kanban owner refused" in out
    assert "no longer available" not in out


def test_incident_replay_second_run(logs):
    logs(
        _old_runs()
        + _marker(7216) + _NEW_RUN.format(sid="3cfbc800")
        + _marker(7217) + _NEW_RUN.format(sid="3518aa43")
    )
    out = kbd._worker_final_output(TASK)
    assert "Kanban owner refused" in out
    assert "3518aa43" in out and "3cfbc800" not in out
    assert "no longer available" not in out


def test_legacy_log_without_markers_keeps_old_behaviour(logs):
    """No marker anywhere (logs written before this change): output as before, the
    newest exit summary still cuts."""
    logs(_old_runs() + _NEW_RUN.format(sid="3cfbc800").replace("\n[kanban-worker-exit] rc=1\n", ""))
    out = kbd._worker_final_output(TASK)
    assert "no longer available" in out
    assert "Kanban owner refused" not in out


def test_exit_trailer_of_older_run_is_ignored(logs):
    logs(_old_runs() + _marker(7216) + "Session: local-3cfbc800\n")
    assert kbd._worker_log_exit_code(TASK) is None


def test_exit_trailer_of_current_run_counts(logs):
    logs(_old_runs() + _marker(7216) + _NEW_RUN.format(sid="3cfbc800"))
    assert kbd._worker_log_exit_code(TASK) == 1


def test_exit_trailer_legacy_log_takes_last(logs):
    logs(_old_runs())
    assert kbd._worker_log_exit_code(TASK) == 0


def test_default_spawn_writes_marker_before_sandbox_prefix(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: log_dir)
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)

    @contextlib.contextmanager
    def _sandbox(task, cmd, env, workspace, profile_home):
        yield list(cmd), env, (), b"[kanban-sandbox] bwrap tenant=x\n"

    monkeypatch.setattr(kbd, "worker_sandbox", _sandbox)

    class _Proc:
        pid = 4321

    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: _Proc())
    (log_dir).mkdir()
    (log_dir / "t_spawn01.log").write_bytes(b"previous run output, no newline")
    task = kb.Task(
        id="t_spawn01", title="ship it", body=None, assignee="default", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None, current_run_id=7263,
    )
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    kbd._default_spawn(task, workspace)

    text = (log_dir / "t_spawn01.log").read_text()
    assert "\n[kanban-worker-spawn] run=7263\n" in text
    assert text.index("[kanban-worker-spawn] run=7263") < text.index("[kanban-sandbox]")
    assert current_run_section(text) == "[kanban-sandbox] bwrap tenant=x\n"
