"""Recalled-id log: one JSON line per surfaced memory id, fail-open (feeds the dream age review)."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider, recall_log


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_line_per_id_with_query_sha_not_query(tmp_path):
    path = tmp_path / "data" / "memory-system" / "recalled.jsonl"
    results = [SimpleNamespace(id="m-1", text="a"), SimpleNamespace(id="m-2", text="b")]
    assert recall_log.log_recalled(results, "what is the dream job cap?", path=str(path)) == 2
    rows = _lines(path)
    sha = hashlib.sha256(b"what is the dream job cap?").hexdigest()[:16]
    assert [(r["memory_id"], r["query_sha"]) for r in rows] == [("m-1", sha), ("m-2", sha)]
    assert all(set(r) == {"ts", "memory_id", "query_sha"} and r["ts"].endswith("Z") for r in rows)
    assert "dream job cap" not in path.read_text()
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_appends_across_calls_and_dedupes_within_one_recall(tmp_path):
    path = tmp_path / "recalled.jsonl"
    recall_log.log_recalled([SimpleNamespace(id="o-1"), SimpleNamespace(id="o-1"), SimpleNamespace(id="f-2")],
                            "q1", path=str(path))
    recall_log.log_recalled([SimpleNamespace(id="f-1")], "q2", path=str(path))
    assert [r["memory_id"] for r in _lines(path)] == ["o-1", "f-2", "f-1"]


def test_empty_or_malformed_results_write_nothing(tmp_path):
    path = tmp_path / "recalled.jsonl"
    junk = [SimpleNamespace(text="no id"), SimpleNamespace(id=None), SimpleNamespace(id=7), MagicMock()]
    assert recall_log.log_recalled([], "q", path=str(path)) == 0
    assert recall_log.log_recalled(junk, "q", path=str(path)) == 0
    assert not path.exists()


def test_unwritable_target_is_swallowed(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert recall_log.log_recalled([SimpleNamespace(id="m")], "q", path=str(blocker / "sub" / "r.jsonl")) == 0


def test_default_path_is_under_the_default_root(tmp_path, monkeypatch):
    monkeypatch.setattr(recall_log, "get_default_hermes_root", lambda: tmp_path)
    assert recall_log.log_path() == str(tmp_path / "data" / "memory-system" / "recalled.jsonl")


def test_logging_is_fast(tmp_path):
    import time
    path = str(tmp_path / "recalled.jsonl")
    results = [SimpleNamespace(id=f"m-{i}") for i in range(40)]
    recall_log.log_recalled(results, "warm", path=path)
    start = time.perf_counter()
    for _ in range(50):
        recall_log.log_recalled(results, "q", path=path)
    per_call_ms = (time.perf_counter() - start) / 50 * 1000
    assert per_call_ms < 1.0, f"{per_call_ms:.3f} ms per 40-id recall"


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(recall_log, "get_default_hermes_root", lambda: tmp_path)
    cfg = tmp_path / "hindsight" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"mode": "cloud", "apiKey": "k", "api_url": "http://localhost:9999",
                               "bank_id": "b", "auto_retain": False}))
    p = HindsightMemoryProvider()
    p.initialize(session_id="s", hermes_home=str(tmp_path), platform="cli")
    p._client = MagicMock()
    p._client.arecall = AsyncMock(return_value=SimpleNamespace(
        results=[SimpleNamespace(id="m-1", text="one"), SimpleNamespace(id="m-2", text="two")]))
    return p


def test_provider_recall_logs_ids_for_auto_and_tool_paths(provider, tmp_path):
    path = tmp_path / "data" / "memory-system" / "recalled.jsonl"
    assert [r.text for r in provider._recall("auto query", auto=True)] == ["one", "two"]
    provider.handle_tool_call("hindsight_recall", {"query": "tool query"})
    rows = _lines(path)
    assert [r["memory_id"] for r in rows] == ["m-1", "m-2", "m-1", "m-2"]
    assert rows[0]["query_sha"] != rows[2]["query_sha"]


def test_provider_recall_survives_a_failing_log(provider, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(recall_log, "get_default_hermes_root", boom)
    assert [r.text for r in provider._recall("q")] == ["one", "two"]
