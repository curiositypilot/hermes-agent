"""A chat subagent that keeps a file outside scratch tells the parent (intake t_e67c62ec).

system.md § Delegation: a delegate_task result dies with the chat, so kept output belongs on a card.
``file_state.written_by`` already knew the child's writes but only the UI event saw them.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

import tools.delegate_tool_child_run as child_run
from tools.delegate_tool_child_run import _ChildRun, _kept_paths, _scratch_roots
from tools.file_tools import clear_file_ops_cache, write_file_tool

HOME = os.path.expanduser("~")
INCIDENT_PATHS = [
    f"{HOME}/workspace/research/swarm-control-landscape.md",
    f"{HOME}/workspace/projects/movis/ads01-niche-alternatives.md",
    f"{HOME}/.hermes/workspace/projects/curiosity-pilot/style-ref-2026-11-22/stregum_page.py",
]


class _Child:
    pass


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A child run whose scratch root is ``tmp_path/scratch``, outside any Kanban worker."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(child_run, "_scratch_roots", lambda: [os.path.realpath(tmp_path / "scratch")])
    child_id = f"sa-kept-{tmp_path.name}"
    r = _ChildRun(_Child(), object(), 0, "goal", child_id, None)
    r.child_task_id = child_id
    yield r
    clear_file_ops_cache(child_id)


def _write(run: _ChildRun, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert json.loads(write_file_tool(str(path), "x\n", task_id=run.child_task_id)).get("error") is None
    return os.path.realpath(path)


def test_kept_paths_incident_replay():
    roots = [f"{HOME}/.hermes/cache"]
    scratch = [f"{HOME}/.hermes/cache/scratch/x.md", f"{HOME}/.hermes/cache/terminal/y"]
    assert _kept_paths(INCIDENT_PATHS + scratch, roots) == sorted(INCIDENT_PATHS)


def test_kept_paths_prefix_lookalikes():
    assert _kept_paths(["/tmpfoo/a", "/tmp-old/b", "/tmp/c", "/tmp"], ["/tmp"]) == ["/tmp-old/b", "/tmpfoo/a"]


def test_scratch_roots_include_cache_and_tempdir(tmp_path, monkeypatch):
    home, tmp = tmp_path / "home", tmp_path / "tmpdir"
    home.mkdir(), tmp.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    roots = _scratch_roots()
    assert os.path.realpath(home / "cache") in roots
    assert os.path.realpath(tmp) in roots


def test_note_appended_for_kept_write(run, tmp_path):
    kept = _write(run, tmp_path / "kept" / "a.md")
    _write(run, tmp_path / "scratch" / "b.txt")
    entry = {"summary": "done"}
    run.append_kept_files_note(entry)
    assert entry["kept_files"] == [kept]
    assert entry["summary"].startswith("done\n\n[NOTE: subagent kept files outside the scratch dir: ")
    assert entry["summary"].endswith("or say in your reply why they are scratch.]")
    assert "a.md" in entry["summary"] and "b.txt" not in entry["summary"]


def test_no_note_for_scratch_only_or_no_writes(run, tmp_path):
    entry = {"summary": "done"}
    run.append_kept_files_note(entry)
    assert entry == {"summary": "done"}
    _write(run, tmp_path / "scratch" / "b.txt")
    run.append_kept_files_note(entry)
    assert entry == {"summary": "done"}


def test_no_note_under_kanban_worker(run, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    _write(run, tmp_path / "kept" / "a.md")
    entry = {"summary": "done"}
    run.append_kept_files_note(entry)
    assert entry == {"summary": "done"}


def test_note_kept_for_cron_inside_kanban_worker(run, tmp_path, monkeypatch):
    """The env var alone is not worker identity: a cron run inside a worker inherits it."""
    from agent.delegation_context import non_dispatcher_owned_context
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    kept = _write(run, tmp_path / "kept" / "a.md")
    entry = {"summary": "done"}
    with non_dispatcher_owned_context():
        run.append_kept_files_note(entry)
    assert entry["kept_files"] == [kept]


def test_summary_none_sets_kept_files_only(run, tmp_path):
    kept = _write(run, tmp_path / "kept" / "a.md")
    entry = {"summary": None}
    run.append_kept_files_note(entry)
    assert entry == {"summary": None, "kept_files": [kept]}


def test_worktree_paths_excluded(run, tmp_path):
    run.worktree_info = {"path": str(tmp_path / "wt")}
    _write(run, tmp_path / "wt" / "src.py")
    entry = {"summary": "done"}
    run.append_kept_files_note(entry)
    assert entry == {"summary": "done"}


@pytest.mark.parametrize("status,summary", [("timeout", None), ("error", "boom")])
def test_failed_child_still_reports_kept_files(run, tmp_path, status, summary):
    """Write, then time out or raise: the failure tail (finish_failed) carries the signal too."""
    kept = _write(run, tmp_path / "kept" / "a.md")
    entry = {"task_index": 0, "status": status, "summary": summary, "error": "x", "duration_seconds": 1.0}
    out = run.finish_failed(entry, None, preview="x")
    assert out["kept_files"] == [kept]
    assert (out["summary"] is None) if summary is None else ("[NOTE: subagent kept files" in out["summary"])
