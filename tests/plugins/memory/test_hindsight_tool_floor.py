"""hindsight_recall tool: opt-in relative floor (``recall_tool_relative_floor``) and dated output.

Regression for the 2026-10-20 EDTH incident (intake t_2065ee46): the tool printed every row
undated and unfloored, so a cancelled plan read as live. The replay fixture keeps only role,
reranker score and dates of the stored rows; no bank text."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider
from plugins.memory.hindsight.settings import _DATED_RECALL_HEADER

_REPLAY = json.loads((Path(__file__).parent / "fixtures" / "hindsight_edth_replay.json").read_text())
_NOTE = re.compile(r"^\((\d+) lower-relevance results dropped; refine the query to see them\)$")


def _row(text, score=None, mentioned_at=None, occurred_start=None):
    scores = None if score is None else SimpleNamespace(reranker=score)
    return SimpleNamespace(text=text, scores=scores, mentioned_at=mentioned_at, occurred_start=occurred_start)


def _replay_rows(query):
    return [_row(role, score, m, o) for role, score, m, o in _REPLAY[query]["rows"]]


@pytest.fixture
def tool_recall(tmp_path, monkeypatch):
    """tool_recall(results, **config) -> the hindsight_recall tool's result text."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

    def run(results, **config):
        path = tmp_path / "hindsight" / "config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"mode": "cloud", "apiKey": "k", "api_url": "http://localhost:9999",
                                    "bank_id": "b", "auto_retain": False, **config}))
        p = HindsightMemoryProvider()
        p.initialize(session_id="tool-floor-test", hermes_home=str(tmp_path), platform="cli")
        client = MagicMock()
        client.arecall = AsyncMock(return_value=SimpleNamespace(results=results))
        p._client = p._reflect_client = client
        return json.loads(p.handle_tool_call("hindsight_recall", {"query": "q"}))["result"]
    return run


def _fact_lines(out):
    return [line for line in out.splitlines() if re.match(r"^\d+\. ", line)]


def _note_count(out):
    last = out.splitlines()[-1]
    m = _NOTE.match(last)
    return int(m.group(1)) if m else 0


class TestToolFloor:
    ROWS = staticmethod(lambda: [_row("m0", 0.9), _row("m1", 0.3), _row("m2", 0.1), _row("m3", 0.05)])

    def test_floor_off_drops_nothing_even_with_ratio_set(self, tool_recall):
        out = tool_recall(self.ROWS(), recall_min_relative_reranker=0.2)
        assert _fact_lines(out) == ["1. m0", "2. m1", "3. m2", "4. m3"]
        assert _note_count(out) == 0

    def test_floor_on_drops_rows_below_ratio_and_counts_them(self, tool_recall):
        rows = self.ROWS() + [_row("unscored")]
        out = tool_recall(rows, recall_min_relative_reranker=0.2, recall_tool_relative_floor=True)
        assert _fact_lines(out) == ["1. m0", "2. m1", "3. unscored"]
        assert _note_count(out) == 2

    def test_floor_on_without_ratio_drops_nothing(self, tool_recall):
        out = tool_recall(self.ROWS(), recall_tool_relative_floor=True)
        assert len(_fact_lines(out)) == 4 and _note_count(out) == 0

    @pytest.mark.parametrize("top", [0.0, -0.5])
    def test_floor_on_with_non_positive_top_fails_open(self, tool_recall, top):
        rows = [_row("a", top), _row("b", top - 0.1)]
        out = tool_recall(rows, recall_min_relative_reranker=0.2, recall_tool_relative_floor=True)
        assert _fact_lines(out) == ["1. a", "2. b"] and _note_count(out) == 0

    def test_empty_results_with_floor_and_dates_on_is_the_fallback_line_alone(self, tool_recall):
        out = tool_recall([], recall_min_relative_reranker=0.2, recall_tool_relative_floor=True,
                          recall_show_dates=True)
        assert out == "No relevant memories found."

    def test_note_follows_the_dated_lines(self, tool_recall):
        rows = [_row("kept", 0.9, "2026-10-20"), _row("weak", 0.01, "2026-10-05")]
        out = tool_recall(rows, recall_min_relative_reranker=0.2, recall_tool_relative_floor=True,
                          recall_show_dates=True)
        assert out.splitlines() == [_DATED_RECALL_HEADER, "1. [2026-10-20] kept",
                                    "(1 lower-relevance results dropped; refine the query to see them)"]

    def test_declared_in_config_schema_default_off(self):
        schema = HindsightMemoryProvider().get_config_schema()
        field = next(f for f in schema if f["key"] == "recall_tool_relative_floor")
        assert field["default"] is False


def test_tool_recall_edth_replay(tool_recall):
    """The 2026-10-20 rows through today's tool path with the landing config."""
    cfg = {"recall_show_dates": True, "recall_min_relative_reranker": 0.2, "recall_tool_relative_floor": True}

    # A (the query as served): the floor cuts the weak tail, both stale plans included.
    a = tool_recall(_replay_rows("A"), **cfg)
    a_lines = _fact_lines(a)
    assert len(a_lines) == 9 and _note_count(a) == 31
    assert not any("stale-plan" in line for line in a_lines)

    # G (typo fixed): everything is relevant; the dates order the stale plan before its cancellation.
    g = tool_recall(_replay_rows("G"), **cfg)
    g_lines = _fact_lines(g)
    assert len(g_lines) == 49 and _note_count(g) == 0
    assert g.splitlines()[0] == _DATED_RECALL_HEADER
    assert "2. [2026-10-05] stale-plan-1" in g_lines
    assert "44. [2026-10-20] cancellation" in g_lines

    # E (cancel-status query): no row may carry a date newer than the cancellations, or the
    # "newer supersedes" header would rank an event-dated plan over them (t_f59f3b7d#1).
    e = tool_recall(_replay_rows("E"), **cfg)
    e_lines = _fact_lines(e)
    assert len(e_lines) == 41
    shown = [m.group(1) for line in e_lines if (m := re.match(r"^\d+\. \[(\d{4}-\d{2}-\d{2})\] ", line))]
    assert max(shown) == "2026-10-20"
    cancels = [line for line in e_lines if " cancellation-" in line]
    assert cancels and all("[2026-10-20] cancellation-" in line for line in cancels)
    assert all(re.match(r"^\d+\. event-\d+$", line) for line in e_lines if " event-" in line)
