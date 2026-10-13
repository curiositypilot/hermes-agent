"""Display projection strips the compaction todo snapshot folded into a real user turn.

``_fold_todo_snapshot`` appends the snapshot to the trailing real user row (model-facing); every
display read must show only the user's own words (upstream #130961).
"""

import copy

from agent.compaction_display import project_compaction_message_for_display
from agent.conversation_compression import _PRUNED_SKILL_RELOAD_NOTICE_HEADER
from hermes_cli.web_routers.sessions import _project_for_display
from tools.todo_tool import TODO_INJECTION_HEADER
from tui_gateway import server

SNAPSHOT = f"{TODO_INJECTION_HEADER}\n- [>] x (in_progress)"
MERGED = {"role": "user", "content": f"yes, please resume\n\n{SNAPSHOT}"}


def test_merged_text_row_shows_only_the_user_text_and_input_is_untouched():
    original = copy.deepcopy(MERGED)
    projected = project_compaction_message_for_display(MERGED)
    assert projected["content"] == "yes, please resume"
    assert MERGED == original


def test_pruned_skill_notice_after_the_snapshot_is_stripped_too():
    row = {"role": "user",
           "content": f"go on\n\n{SNAPSHOT}\n\n{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\nreload skill_view(name='x')"}
    assert project_compaction_message_for_display(row)["content"] == "go on"


def test_pure_snapshot_row_is_hidden():
    row = {"role": "user", "content": SNAPSHOT, "_todo_snapshot_synthetic": True}
    assert project_compaction_message_for_display(row) is None


def test_list_content_keeps_non_text_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    row = {"role": "user", "content": [{"type": "text", "text": f"hi\n\n{SNAPSHOT}"}, image]}
    original = copy.deepcopy(row)
    projected = project_compaction_message_for_display(row)
    assert projected["content"] == [{"type": "text", "text": "hi"}, image]
    assert row == original


def test_assistant_row_quoting_the_header_is_unchanged():
    row = {"role": "assistant", "content": f"I saw {SNAPSHOT}"}
    assert project_compaction_message_for_display(row) == row


def test_tui_history_projection_shows_the_user_text():
    messages = server._history_to_messages([dict(MERGED)])
    assert [(m["role"], m["text"]) for m in messages] == [("user", "yes, please resume")]


def test_desktop_rest_projection_carries_stripped_display_content():
    pure = {"role": "user", "content": SNAPSHOT}
    plain = {"role": "user", "content": "hello"}
    merged, hidden, untouched = _project_for_display([dict(MERGED), pure, plain])
    assert merged["content"] == MERGED["content"]  # physical content kept for export
    assert merged["display_content"] == "yes, please resume"
    assert hidden["display_kind"] == "hidden"
    assert "display_content" not in untouched
