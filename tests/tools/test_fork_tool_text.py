"""Fork-owned shorter core tool wording (tools/fork_tool_text.py) applied at schema build.

The overlay edits text inside upstream-owned schemas, so an upstream reword silently stops an edit
from applying. ``test_every_edit_applies`` is the tripwire that turns that into a failing test on the
next sync instead of a quiet token regression.
"""

from __future__ import annotations

import json
import logging

import pytest

TOOLS = ("clarify", "delegate_task", "execute_code", "read_file", "search_files",
         "session_search", "skill_manage", "terminal", "todo_list")

MUST_KEEP = [
    ("execute_code", ("parameters", "properties", "code", "description"), "Import tools with `from hermes_tools import "),
    ("session_search", ("parameters", "properties", "query", "description"), "Omit to browse recent sessions"),
    ("session_search", ("description",), "`session_id` alone = read a whole session"),
    ("delegate_task", ("description",), "END YOUR TURN"),
    ("delegate_task", ("description",), "Never wait or poll"),
    ("clarify", ("description",), "ONLY"),
    ("todo_list", ("description",), "Only ONE item in_progress"),
]


def _built(names=TOOLS) -> dict[str, dict]:
    """Definitions as model_tools builds them up to the overlay (registry + dynamic rewriters)."""
    from model_tools import _apply_dynamic_schemas
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS
    from tools.registry import registry

    defs = _apply_dynamic_schemas(registry.get_definitions(set(names) | set(SANDBOX_ALLOWED_TOOLS), quiet=True))
    return {d["function"]["name"]: d["function"] for d in defs if d["function"]["name"] in names}


def _overlaid(functions: dict[str, dict]) -> dict[str, dict]:
    from tools.fork_tool_text import apply_fork_tool_text

    defs = [{"type": "function", "function": f} for f in functions.values()]
    return {d["function"]["name"]: d["function"] for d in apply_fork_tool_text(defs)}


def _skips(functions: dict[str, dict]) -> dict[str, list[int]]:
    from tools.fork_tool_text import EDITS, apply_edits

    return {name: skipped for name, function in functions.items()
            if (skipped := apply_edits(function, EDITS[name], name)[1])}


def _strip(node):
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if not (k == "description" and isinstance(v, str))}
    if isinstance(node, list):
        return [_strip(x) for x in node]
    return node


def _descriptions(node, path=()):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "description" and isinstance(v, str):
                out[path + (k,)] = v
            else:
                out.update(_descriptions(v, path + (k,)))
    elif isinstance(node, list):
        for i, x in enumerate(node):
            out.update(_descriptions(x, path + (i,)))
    return out


def _at(node, path):
    for key in path:
        node = node[key]
    return node


def test_every_edit_applies():
    """Default config, sandbox tools selected: every edit matches exactly once (upstream-sync tripwire)."""
    built = _built()
    assert set(built) == set(TOOLS)
    assert _skips(built) == {}


def test_only_descriptions_change_and_none_is_empty():
    built = _built()
    after = _overlaid(built)
    for name in TOOLS:
        assert json.dumps(_strip(built[name])) == json.dumps(_strip(after[name])), name
        before_d, after_d = _descriptions(built[name]), _descriptions(after[name])
        assert set(before_d) == set(after_d), name
        assert all(text.strip() for text in after_d.values()), name
        assert after_d != before_d, name  # each tool really gets shorter wording


def test_must_keep_phrases_survive():
    after = _overlaid(_built())
    for name, path, phrase in MUST_KEEP:
        assert phrase in _at(after[name], path), (name, phrase)
    session_id = _at(after["session_search"], ("parameters", "properties", "session_id", "description"))
    assert "Must be paired" not in session_id
    assert "whole session" in session_id


def test_registry_schemas_are_not_mutated():
    built = _built()
    snapshot = json.dumps(built, sort_keys=True)
    _overlaid(built)
    assert json.dumps(built, sort_keys=True) == snapshot
    assert json.dumps(_built(), sort_keys=True) == snapshot


@pytest.mark.parametrize("text", ["nothing to see", "twice old and old"])
def test_absent_or_ambiguous_old_leaves_field_unchanged(text, caplog):
    from tools.fork_tool_text import apply_edits

    function = {"name": "x", "description": text, "parameters": {"type": "object", "properties": {}}}
    edit = (("description",), "old" if "twice" in text else "missing", "new")
    with caplog.at_level(logging.WARNING, logger="tools.fork_tool_text"):
        out, skipped = apply_edits(function, [edit], f"probe-{len(text)}")
    assert out == function
    assert skipped == [0]
    assert "skipped" in caplog.text


def test_configured_caps_render_and_edits_still_apply(monkeypatch):
    import tools.file_tools as ft

    monkeypatch.setattr(ft, "_get_max_read_chars", lambda: 40000)
    monkeypatch.setattr("tools.read_extract.hosted_ocr_available", lambda: False)
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 20000)

    built = _built()
    assert _skips(built) == {}
    after = _overlaid(built)
    read_file = after["read_file"]
    assert "~40K characters" in read_file["description"] and "~100K" not in read_file["description"]
    assert "~40K-character" in read_file["parameters"]["properties"]["limit"]["description"]
    assert "Over 20KB, output is head+tail cut" in after["execute_code"]["description"]


def test_delegate_edits_apply_when_clarify_is_absent():
    from model_tools import _rewrite_delegate_task
    from tools.fork_tool_text import EDITS, apply_edits
    from tools.registry import registry

    raw = registry.get_definitions({"delegate_task"}, quiet=True)[0]
    td = _rewrite_delegate_task(raw, {"delegate_task", "memory"})
    assert td is not None
    description = td["function"]["description"]
    restrictions = [line for line in description.splitlines() if line.startswith("- Children cannot call")]
    assert restrictions and all("clarify" not in line and "memory" in line for line in restrictions)
    out, skipped = apply_edits(td["function"], EDITS["delegate_task"], "delegate_task")
    assert skipped == []
    assert [line for line in out["description"].splitlines() if line.startswith("- Children cannot call")] \
        == restrictions


def test_hook_never_breaks_tool_loading(monkeypatch):
    import model_tools
    import tools.fork_tool_text as ftt
    from tools.clarify_tool import CLARIFY_SCHEMA

    def boom(_defs):
        raise RuntimeError("overlay bug")

    monkeypatch.setattr(ftt, "apply_fork_tool_text", boom)
    defs = model_tools._compute_tool_definitions(enabled_toolsets=["clarify"], quiet_mode=True,
                                                 skip_tool_search_assembly=True)
    clarify = next(d["function"] for d in defs if d["function"]["name"] == "clarify")
    assert clarify["description"] == CLARIFY_SCHEMA["description"]


def test_tool_definitions_carry_the_overlay():
    import model_tools

    defs = model_tools._compute_tool_definitions(enabled_toolsets=["clarify"], quiet_mode=True,
                                                 skip_tool_search_assembly=True)
    clarify = next(d["function"] for d in defs if d["function"]["name"] == "clarify")
    assert clarify["description"].startswith("Ask the user for a decision or feedback.")
