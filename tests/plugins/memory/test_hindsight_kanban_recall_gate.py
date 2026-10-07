"""Kanban automatic recall is optional, independently of explicit memory tools; with
recall_kanban_first_turn a worker runs one card-query recall behind host-wide slots."""
import fcntl
import json
import os
import threading
import time
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


# -- recall_kanban_first_turn (t_b765c677) ----------------------------------------
@pytest.fixture
def make_first_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_first_turn")
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("plugins.memory.hindsight.get_default_hermes_root", lambda: tmp_path)

    def make(title="Gateway restart: load kanban caps", **config):
        path = tmp_path / "hindsight" / "config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"mode": "local_external", "auto_retain": False,
                                    "recall_kanban_workers": False, **config}))
        p = HindsightMemoryProvider()
        p.initialize(session_id="first-turn-test", hermes_home=str(tmp_path), platform="cli")
        # The card read: stub the one DB read so title/query come from a fixed card.
        p._kanban_recall = None

        def _ctx():
            if p._kanban_recall is None:
                p._kanban_card_title = title
                p._kanban_recall = (f"{title}\n\ncard body", [])
            return p._kanban_recall
        p._kanban_recall_context = _ctx
        return p
    return make


def _lock(tmp_path, i=0):
    lock_dir = tmp_path / "hindsight" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_dir / f"kanban-recall-{i}.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@pytest.mark.parametrize("sync", [True, False])
def test_first_turn_only(make_first_turn, sync):
    p = make_first_turn(recall_kanban_first_turn=True, recall_sync=sync)
    p._do_recall = MagicMock(return_value=("dispatcher reads max_in_progress at boot", 1))
    assert "max_in_progress" in p.prefetch("work kanban task t_first_turn")
    p.queue_prefetch("work kanban task t_first_turn")
    assert "" == p.prefetch("turn 2")
    p.queue_prefetch("turn 2")
    p._do_recall.assert_called_once()
    assert p._prefetch_thread is None


@pytest.mark.parametrize("outcome", ["raise", "empty"])
def test_first_turn_counts_failures(make_first_turn, outcome):
    p = make_first_turn(recall_kanban_first_turn=True, recall_sync=True)
    p._recall = MagicMock(side_effect=RuntimeError("server down") if outcome == "raise"
                          else None, return_value=[])
    assert p.prefetch("turn 1") == ""
    assert p.prefetch("turn 2") == ""
    p._recall.assert_called_once()


def test_skip_prefix(make_first_turn):
    p = make_first_turn(title="critique/plan: x", recall_kanban_first_turn=True)
    p._do_recall = MagicMock(side_effect=AssertionError("critic recalled"))
    assert p.prefetch("turn 1") == ""
    p._do_recall.assert_not_called()


def test_slot_exhausted_skips_and_marks_done(make_first_turn, tmp_path):
    p = make_first_turn(recall_kanban_first_turn=True, recall_kanban_slot_wait_s=0.5)
    p._do_recall = MagicMock(side_effect=AssertionError("recalled without a slot"))
    fd = _lock(tmp_path)
    try:
        start = time.monotonic()
        assert p.prefetch("turn 1") == ""
        assert time.monotonic() - start < 1.5
        # A no-slot skip counts as the first attempt: turn 2 does not queue for a slot again.
        start = time.monotonic()
        assert p.prefetch("turn 2") == ""
        assert time.monotonic() - start < 0.2
    finally:
        os.close(fd)
    p._do_recall.assert_not_called()


def test_interactive_takes_no_slot(make_first_turn, tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    p = make_first_turn(recall_kanban_first_turn=True, recall_sync=True, recall_kanban_slot_wait_s=0)
    p._do_recall = MagicMock(return_value=("interactive evidence", 1))
    fd = _lock(tmp_path)
    try:
        assert "interactive evidence" in p.prefetch("q1")
        assert "interactive evidence" in p.prefetch("q2")
    finally:
        os.close(fd)
    assert p._do_recall.call_count == 2


def test_first_turn_recall_bounds_its_call_timeout(make_first_turn):
    """The slot holder's request is capped by recall_kanban_recall_timeout_s, not the 120 s API
    timeout (the manager's join timeout cannot stop the thread), and the cap is per call."""
    p = make_first_turn(recall_kanban_first_turn=True, recall_kanban_recall_timeout_s=4)
    seen = []
    p._run_sync = lambda coro: None
    p._do_recall = lambda q: (seen.append(getattr(p._call_timeout, "value", None)), ("x", 1))[1]
    p.prefetch("turn 1")
    assert seen == [4]
    assert p._call_timeout.value is None


def test_first_turn_reads_card_without_card_query_knob(tmp_path, monkeypatch):
    """First-turn mode reads the card (query + title) even with recall_kanban_card_query off."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_card")
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    task = SimpleNamespace(title="critique/diff: impl.diff", body="body", tenant="")
    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_connect as kdc
    monkeypatch.setattr(kb, "get_task", lambda conn, tid: task)
    monkeypatch.setattr(kdc, "connect_closing", lambda: __import__("contextlib").nullcontext(None))
    (tmp_path / "hindsight").mkdir()
    (tmp_path / "hindsight" / "config.json").write_text(json.dumps(
        {"mode": "local_external", "auto_retain": False, "recall_kanban_workers": False,
         "recall_kanban_first_turn": True}))
    p = HindsightMemoryProvider()
    p.initialize(session_id="card", hermes_home=str(tmp_path), platform="cli")
    query, _ = p._kanban_recall_context()
    assert query.startswith("critique/diff: impl.diff")
    assert p._kanban_card_title == "critique/diff: impl.diff"


@pytest.mark.parametrize("first_turn", [True, None])
def test_burst_replay_serializes(make_first_turn, tmp_path, first_turn):
    """2026-10-04 00:05:21 burst (two cards spawned together) plus one: with one slot the three
    recalls never overlap; with first_turn unset no recall runs at all (the 10-04 behaviour)."""
    cfg = {"recall_kanban_slots": 1, "recall_kanban_slot_wait_s": 5}
    if first_turn:
        cfg["recall_kanban_first_turn"] = True
    intervals, lock = [], threading.Lock()

    def fake_recall(query):
        start = time.monotonic()
        time.sleep(0.3)
        with lock:
            intervals.append((start, time.monotonic()))
        return "fact", 1

    providers = [make_first_turn(**cfg) for _ in range(3)]
    for p in providers:
        p._do_recall = fake_recall
    threads = [threading.Thread(target=p.prefetch, args=("work kanban task",)) for p in providers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    if not first_turn:
        assert intervals == []
        return
    assert len(intervals) == 3
    intervals.sort()
    for (_, end), (start, _) in zip(intervals, intervals[1:]):
        assert start >= end


def test_kanban_prefetch_timeout_override(monkeypatch):
    from agent.agent_init import _external_prefetch_timeout

    cfg = {"external_prefetch_timeout_seconds": 8, "kanban_external_prefetch_timeout_seconds": 30}
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _external_prefetch_timeout(cfg) == 8
    monkeypatch.setenv("HERMES_KANBAN_TASK", "  ")
    assert _external_prefetch_timeout(cfg) == 8
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    assert _external_prefetch_timeout(cfg) == 30
    assert _external_prefetch_timeout({**cfg, "kanban_external_prefetch_timeout_seconds": 0}) == 8
    assert _external_prefetch_timeout({"external_prefetch_timeout_seconds": 8}) == 8
