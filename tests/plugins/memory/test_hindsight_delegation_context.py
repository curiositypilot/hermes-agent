"""Hindsight parent-side recall for delegate_task children (t_2354171e)."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


@pytest.fixture
def make_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

    def make(results=None, **config):
        path = tmp_path / "hindsight" / "config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"mode": "local_external", "auto_retain": False, **config}))
        p = HindsightMemoryProvider()
        p.initialize(session_id="deleg-test", hermes_home=str(tmp_path), platform="cli")
        items = [SimpleNamespace(text=t) for t in (results if results is not None else ["fact one"])]
        client = SimpleNamespace(arecall=MagicMock(return_value=SimpleNamespace(results=items)))
        p._run_hindsight_operation = lambda fn, **_kw: fn(client)
        p._client_for_test = client
        return p
    return make


@pytest.mark.parametrize("config", [
    {"memory_mode": "tools"}, {"auto_recall": False}, {"recall_delegate_children": False},
    {"recall_delegate_children": "false"}, {"recall_delegate_max_items": 0},
])
def test_disabled_returns_empty_without_server_call(make_provider, config):
    p = make_provider(**config)
    assert p.delegation_context("Audit auto-recall") == ""
    p._client_for_test.arecall.assert_not_called()


def test_kanban_worker_opt_out_does_not_silence_children(make_provider, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deleg_test")
    p = make_provider(recall_kanban_workers=False, recall_kanban_card_query=True)
    p._kanban_recall_context = MagicMock(return_value=("CARD TITLE TEXT", []))
    out = p.delegation_context("Audit what Hindsight auto-recall injects")
    assert "fact one" in out
    kwargs = p._client_for_test.arecall.call_args.kwargs
    assert kwargs["query"] == "Audit what Hindsight auto-recall injects"


def test_query_truncated_to_recall_max_input_chars(make_provider):
    p = make_provider(recall_max_input_chars=10)
    p.delegation_context("x" * 50)
    assert p._client_for_test.arecall.call_args.kwargs["query"] == "x" * 10


def test_capped_at_max_items_and_prefetch_state_untouched(make_provider):
    p = make_provider(results=[f"fact {i}" for i in range(12)], recall_delegate_max_items=3)
    p._prefetch_result, p._prefetch_count = "parent turn result", 4
    p._last_recall_returned, p._last_recall_count = True, 4
    out = p.delegation_context("goal")
    assert [ln for ln in out.splitlines() if ln.startswith("- ")] == ["- fact 0", "- fact 1", "- fact 2"]
    assert (p._prefetch_result, p._prefetch_count) == ("parent turn result", 4)
    assert (p._last_recall_returned, p._last_recall_count) == (True, 4)


def test_default_cap_is_eight_and_reranker_floor_applies(make_provider):
    p = make_provider(results=[f"fact {i}" for i in range(12)], recall_min_reranker=0.05)
    out = p.delegation_context("goal")
    assert sum(1 for ln in out.splitlines() if ln.startswith("- ")) == 8
    assert p._client_for_test.arecall.call_args.kwargs["min_scores"] == {"reranker": 0.05}


def test_block_does_not_forbid_tool_lookups(make_provider):
    out = make_provider().delegation_context("goal")
    assert "Do not call tools" not in out


def test_recall_error_returns_empty(make_provider):
    p = make_provider()
    p._client_for_test.arecall.side_effect = RuntimeError("down")
    assert p.delegation_context("goal") == ""


def test_schema_declares_new_settings(make_provider):
    keys = {f["key"] for f in make_provider().get_config_schema()}
    assert {"recall_delegate_children", "recall_delegate_max_items"} <= keys
