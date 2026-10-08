"""Hindsight ``recall_show_dates``: dated auto-recall lines + one supersession header (t_4f2571c7)."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider
from plugins.memory.hindsight.settings import _DATED_RECALL_HEADER, _format_recall_lines, _recall_date


def _r(text, mentioned_at=None, occurred_start=None):
    return SimpleNamespace(text=text, mentioned_at=mentioned_at, occurred_start=occurred_start)


class TestFormatter:
    def test_off_renders_plain_lines_and_ignores_dates(self):
        rs = [_r("old fact", "2026-10-01T18:08:00+00:00"), _r("new fact", "2026-10-03T00:00:00+00:00")]
        assert _format_recall_lines(rs) == ["- old fact", "- new fact"]

    def test_on_prefixes_the_mentioned_date(self):
        rs = [_r("old fact", "2026-10-01T18:08:00.090000+00:00"), _r("new fact", "2026-10-03T00:00:00+00:00")]
        assert _format_recall_lines(rs, True) == ["- [2026-10-01] old fact", "- [2026-10-03] new fact"]

    def test_mentioned_at_wins_over_occurred_start(self):
        # mentioned_at = when the fact was recorded; occurred_start = when the event happened.
        assert _recall_date(_r("x", "2026-10-05T09:00:00Z", "2025-01-01T00:00:00Z")) == "2026-10-05"

    def test_falls_back_to_occurred_start(self):
        assert _recall_date(_r("x", None, "2026-09-30T21:22:00+00:00")) == "2026-09-30"

    def test_accepts_datetime_and_dict_shapes(self):
        assert _recall_date(_r("x", datetime(2026, 10, 4, 13, 1, tzinfo=timezone.utc))) == "2026-10-04"
        assert _recall_date(_r("x", None, date(2026, 10, 2))) == "2026-10-02"
        assert _recall_date({"text": "x", "mentioned_at": "2026-10-06T14:22:00Z"}) == "2026-10-06"

    @pytest.mark.parametrize("bad", [None, "", "yesterday", "10-04 13:01"])
    def test_undated_result_keeps_plain_form(self, bad):
        assert _format_recall_lines([_r("fact", bad, bad)], True) == ["- fact"]

    def test_textless_results_are_skipped(self):
        assert _format_recall_lines([_r(""), _r(None), _r("kept", "2026-10-01")], True) == ["- [2026-10-01] kept"]


@pytest.fixture
def make_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

    def make(**config):
        path = tmp_path / "hindsight" / "config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"mode": "cloud", "apiKey": "k", "api_url": "http://localhost:9999",
                                    "bank_id": "b", "auto_retain": False, "recall_sync": True, **config}))
        p = HindsightMemoryProvider()
        p.initialize(session_id="dates-test", hermes_home=str(tmp_path), platform="cli")
        results = [_r("Scheduled only parks a card.", "2026-10-01T18:08:00+00:00"),
                   _r("Kanban gained native dated scheduling.", "2026-10-03T00:00:00+00:00"),
                   _r("Undated fact.")]
        client = MagicMock()
        client.arecall = AsyncMock(return_value=SimpleNamespace(results=results))
        p._client = p._reflect_client = client
        return p
    return make


class TestProviderWiring:
    def test_default_off_injects_byte_identical_plain_block(self, make_provider):
        p = make_provider()
        assert p._recall_show_dates is False
        out = p.prefetch("kanban scheduling")
        assert _DATED_RECALL_HEADER not in out
        assert "- Scheduled only parks a card.\n- Kanban gained native dated scheduling.\n- Undated fact." in out
        assert "[2026-" not in out

    @pytest.mark.parametrize("value", [True, "true", "1"])
    def test_on_dates_each_line_under_one_header(self, make_provider, value):
        p = make_provider(recall_show_dates=value)
        out = p.prefetch("kanban scheduling")
        assert out.count(_DATED_RECALL_HEADER) == 1
        body = out.split(_DATED_RECALL_HEADER + "\n", 1)[1]
        assert body.splitlines() == ["- [2026-10-01] Scheduled only parks a card.",
                                     "- [2026-10-03] Kanban gained native dated scheduling.",
                                     "- Undated fact."]
        assert p.recall_status().count == 3

    def test_empty_recall_injects_nothing_even_with_dates(self, make_provider):
        p = make_provider(recall_show_dates=True)
        p._client.arecall = AsyncMock(return_value=SimpleNamespace(results=[]))
        assert p.prefetch("anything") == ""

    def test_delegation_block_gets_dates_too(self, make_provider):
        p = make_provider(recall_show_dates=True, recall_delegate_max_items=2)
        out = p.delegation_context("audit kanban scheduling")
        assert out.count(_DATED_RECALL_HEADER) == 1
        assert "- [2026-10-03] Kanban gained native dated scheduling." in out
        assert "Undated fact." not in out  # cap applies to fact lines, not the header

    def test_tool_recall_output_is_unchanged(self, make_provider):
        p = make_provider(recall_show_dates=True)
        out = p.handle_tool_call("hindsight_recall", {"query": "kanban"})
        assert _DATED_RECALL_HEADER not in out

    def test_declared_in_config_schema_default_off(self, make_provider):
        field = next(f for f in make_provider().get_config_schema() if f["key"] == "recall_show_dates")
        assert field["default"] is False
