"""Completion artifacts are copied into the task's attachments whatever the
workspace kind; a declared path that cannot be copied refuses the transition.

Regression for intake t_3083f018: a worktree card declared PNGs under the
global ``cache/scratch`` dir and kanban_complete returned ``attachments: []``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _complete_with_global_scratch_png(kind: str, tmp_path: Path) -> None:
    """A ``kind`` task declares a PNG under a global ``cache/scratch`` dir (the
    t_e24c1035 shape); the copy must land on the card and outlive the source."""
    workdir = tmp_path / f"{kind}-workdir"
    workdir.mkdir()
    source = tmp_path / "cache" / "scratch" / "smoke" / "main.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"png-bytes")
    with kbc.connect() as conn:
        t = kb.create_task(conn, title=f"{kind} smoke", workspace_kind=kind, workspace_path=str(workdir))
        assert kb.complete_task(conn, t, result="ok", metadata={"artifacts": [str(source)]})
        completed = [e for e in kb.list_events(conn, t) if e.kind == "completed"][-1]
        attachments = kb.list_attachments(conn, t)
    copy = Path(completed.payload["artifacts"][0])
    assert copy.parent == kb.task_attachments_dir(t).resolve()
    assert [(a.filename, a.stored_path) for a in attachments] == [("main.png", str(copy))]
    source.unlink()
    assert copy.read_bytes() == b"png-bytes"


def test_worktree_task_copies_artifact_from_global_scratch(kanban_home, tmp_path):
    _complete_with_global_scratch_png("worktree", tmp_path)


def test_dir_task_copies_artifact_outside_workspace(kanban_home, tmp_path):
    _complete_with_global_scratch_png("dir", tmp_path)


def _assert_artifact_refused(tmp_path: Path, artifact: Path, reason: str) -> None:
    good = tmp_path / "good.txt"
    good.write_text("ok")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="refuse", workspace_kind="dir", workspace_path=str(workdir))
        kb.claim_task(conn, t)
        run_id = kb.get_task(conn, t).current_run_id
        with pytest.raises(kb.ArtifactPreservationError, match=reason):
            kb.complete_task(
                conn, t, result="ok", expected_run_id=run_id,
                metadata={"artifacts": [str(good), str(artifact)]},
            )
        assert kb.get_task(conn, t).status == "running"
        assert kb.list_attachments(conn, t) == []
    attachment_dir = kb.task_attachments_dir(t)
    assert not attachment_dir.exists() or not any(attachment_dir.iterdir())


def test_missing_artifact_outside_scratch_refuses_completion(kanban_home, tmp_path):
    _assert_artifact_refused(tmp_path, tmp_path / "gone.png", "not found")


def test_directory_artifact_refuses_completion(kanban_home, tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    _assert_artifact_refused(tmp_path, shots, "is a directory")


def test_oversize_artifact_refuses_completion(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "KANBAN_ATTACHMENT_MAX_BYTES", 8)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 9)
    _assert_artifact_refused(tmp_path, big, "exceeds 8 bytes")


def test_already_staged_artifact_not_recopied(kanban_home, tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf")
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="staged", workspace_kind="dir", workspace_path=str(workdir))
        kb.claim_task(conn, t)
        run_id = kb.get_task(conn, t).current_run_id
        assert kb.request_review(
            conn, t, summary="ready", metadata={"artifacts": [str(source)]}, expected_run_id=run_id)
        staged = [e for e in kb.list_events(conn, t) if e.kind == "review_requested"][-1].payload["artifacts"]
        assert kb.complete_task(conn, t, summary="approved", metadata={"artifacts": staged})
        completed = [e for e in kb.list_events(conn, t) if e.kind == "completed"][-1]
        attachments = kb.list_attachments(conn, t)
    assert completed.payload["artifacts"] == staged
    assert [a.stored_path for a in attachments] == staged
