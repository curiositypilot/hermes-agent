"""``subagent.complete`` reports the files the child itself wrote.

Regression for intake t_f1935989: ``emit_complete`` asked ``writes_since`` with an empty path list,
which filters every write out, so ``files_written`` was always ``[]``.
"""
from __future__ import annotations

import json
import os

from tools import file_state
from tools.delegate_tool_child_run import _ChildRun
from tools.file_tools import clear_file_ops_cache, write_file_tool


class _Child:
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_reasoning_tokens = 0


def _complete_event(run: _ChildRun) -> dict:
    events = []
    run.child_progress_cb = lambda event, **kw: events.append((event, kw))
    run.emit_complete({"messages": []}, {"summary": "done", "status": "completed", "api_calls": 1}, 1.0)
    return next(kw for event, kw in events if event == "subagent.complete")


def test_complete_event_lists_the_childs_own_writes_only(tmp_path):
    child_id, sibling_id = "sa-files-written-child", "sa-files-written-sibling"
    mine = [tmp_path / "b.txt", tmp_path / "a.txt"]
    theirs = tmp_path / "sibling.txt"
    try:
        for path in mine:
            assert json.loads(write_file_tool(str(path), "x\n", task_id=child_id)).get("error") is None
        assert json.loads(write_file_tool(str(theirs), "y\n", task_id=sibling_id)).get("error") is None

        run = _ChildRun(_Child(), object(), 0, "goal", child_id, None)
        run.child_task_id = child_id
        written = _complete_event(run)["files_written"]
    finally:
        clear_file_ops_cache(child_id)
        clear_file_ops_cache(sibling_id)

    assert written == sorted(os.path.realpath(p) for p in mine)


def test_complete_event_files_written_empty_without_writes():
    run = _ChildRun(_Child(), object(), 0, "goal", "sa-files-written-idle", None)
    run.child_task_id = "sa-files-written-idle"
    assert _complete_event(run)["files_written"] == []
    assert file_state.written_by("sa-files-written-idle") == []
