"""Kanban automatic recall is optional, independently of explicit memory tools."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


@pytest.fixture
def make_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

    def make(**config):
        path = tmp_path / "hindsight" / "config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"mode": "local_external", "auto_retain": False, **config}))
        p = HindsightMemoryProvider()
        p.initialize(session_id="gate-test", hermes_home=str(tmp_path), platform="cli")
        return p
    return make


@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize("disabled", [False, "false"])
def test_worker_skips_sync_and_background(make_provider, monkeypatch, sync, disabled):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_gate_test")
    p = make_provider(recall_kanban_workers=disabled, recall_sync=sync)
    p._do_recall = MagicMock(side_effect=AssertionError("automatic recall reached server"))
    p._join_prefetch = MagicMock(side_effect=AssertionError("disabled worker waited"))
    p._prefetch_result, p._prefetch_count = "stale", 1
    p._last_recall_returned, p._last_recall_count = True, 1
    assert p.prefetch("S2 robot camera detection recorder") == ""
    p.queue_prefetch("S2 robot camera detection recorder")
    p._do_recall.assert_not_called()
    assert p._prefetch_thread is None
    assert p.recall_status() is None


@pytest.mark.parametrize("task", [None, "", "   "])
def test_interactive_recall_unchanged(make_provider, monkeypatch, task):
    if task is not None:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    p = make_provider(recall_kanban_workers=False, recall_sync=True)
    p._do_recall = MagicMock(return_value=("interactive evidence", 1))
    assert "interactive evidence" in p.prefetch("What did we decide about memory latency?")
    p._do_recall.assert_called_once()


@pytest.mark.parametrize("config", [{}, {"recall_kanban_workers": True}])
def test_worker_default_and_opt_in_preserve_recall(make_provider, monkeypatch, config):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_gate_test")
    p = make_provider(recall_sync=True, **config)
    p._do_recall = MagicMock(return_value=("worker evidence", 1))
    assert "worker evidence" in p.prefetch("S2 robot camera detection recorder")


def test_explicit_tools_ignore_automatic_gate(make_provider, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_gate_test")
    p = make_provider(recall_kanban_workers=False)
    client = SimpleNamespace(arecall=MagicMock(return_value=SimpleNamespace(results=["evidence"])),
                             areflect=MagicMock(return_value=SimpleNamespace(text="synthesis")))
    p._run_hindsight_operation = lambda fn: fn(client)
    assert p._recall("targeted search") == ["evidence"]
    assert p._reflect("targeted question") == "synthesis"
    client.arecall.assert_called_once()
    client.areflect.assert_called_once()
