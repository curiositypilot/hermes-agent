"""sessions.tool_call_count counts the calls a session made; in-place compaction never lowers it.

Compaction archives the summarised rows (non-destructive), so resetting the counter to the live
set dropped calls analytics reports as made. Rewind keeps its live-transcript meaning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB

SUMMARY = {"role": "user", "content": "[CONTEXT COMPACTION] summary"}


def _call(call_id: str) -> list[dict]:
    return [{"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]


def _turn(db: SessionDB, call_id: str) -> None:
    db.append_message("sid", "user", f"run {call_id}")
    db.append_message("sid", "assistant", "", tool_calls=_call(call_id))
    db.append_message("sid", "tool", f"out {call_id}", tool_call_id=call_id, tool_name="terminal")


@pytest.fixture
def db(tmp_path: Path):
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sid", source="test")
    for call_id in "abc":
        _turn(d, call_id)
    yield d
    d.close()


def _count(db: SessionDB) -> int:
    return db.get_session("sid")["tool_call_count"]


def _carried_c(db: SessionDB) -> list[dict]:
    """Verbatim copies of the live call-c pair (assistant + its tool result)."""
    held = db.get_messages_as_conversation("sid", include_row_ids=True)
    return [m for m in held if m.get("tool_call_id") == "c" or (m.get("tool_calls") and m["tool_calls"][0]["id"] == "c")]


def _active_ids(db: SessionDB) -> list[int]:
    return [r[0] for r in db._conn.execute(
        "SELECT id FROM messages WHERE session_id = 'sid' AND active = 1 ORDER BY id").fetchall()]


def test_watermark_compaction_keeps_count(db):
    assert _count(db) == 3
    carried = _carried_c(db)
    db.archive_and_compact("sid", [SUMMARY, *carried], watermark=db.get_active_message_watermark("sid"),
                           tail_count=len(carried))
    assert _count(db) == 3  # old code reset it to the live set: 1
    assert db.get_session("sid")["message_count"] == 3


def test_named_coverage_compaction_keeps_count(db):
    carried = _carried_c(db)
    db.archive_and_compact("sid", [SUMMARY, *carried], watermark=db.get_active_message_watermark("sid"),
                           tail_count=len(carried), covered_ids=_active_ids(db), unresolved_held=[])
    assert _count(db) == 3


def test_unpersisted_held_call_counted(db):
    held_d = {"role": "assistant", "content": "", "tool_calls": _call("d")}
    db.archive_and_compact("sid", [SUMMARY, *_carried_c(db), held_d],
                           watermark=db.get_active_message_watermark("sid"))
    assert _count(db) == 4


def test_tail_clone_not_double_counted(db):
    watermark = db.get_active_message_watermark("sid")
    db.append_message("sid", "assistant", "", tool_calls=_call("e"))  # lands during the slow summary
    assert _count(db) == 4
    db.archive_and_compact("sid", [SUMMARY], watermark=watermark)
    live_calls = [m["tool_calls"][0]["id"] for m in db.get_messages_as_conversation("sid") if m.get("tool_calls")]
    assert live_calls == ["e"]  # the tail row was cloned, not summarised
    assert _count(db) == 4


def test_repeat_compaction_stable(db):
    for _ in range(2):
        carried = _carried_c(db)
        db.archive_and_compact("sid", [SUMMARY, *carried], watermark=db.get_active_message_watermark("sid"),
                               tail_count=len(carried))
    assert _count(db) == 3


def test_rewind_still_live(db):
    carried = _carried_c(db)
    db.archive_and_compact("sid", [SUMMARY, *carried], watermark=db.get_active_message_watermark("sid"),
                           tail_count=len(carried))
    _turn(db, "f")
    assert _count(db) == 4
    rows = db.get_messages("sid")
    target = next(r for r in rows if r["role"] == "user" and r["content"] == "run f")
    db.rewind_to_message("sid", target["id"])
    # Rewind sets the live count on purpose: the carried call c is the only live call left.
    assert _count(db) == 1
