"""hindsight_retain tool items: ``tool_retain_strategy`` gates strategy + agent-note envelope.

Regression for intake t_89c89c84 (slice 2 of t_e75177c6): a deliberate agent note stored via the
tool reached the bank with no strategy and no source envelope, so the bank read it under the
transcript rules and its ``agent-note`` strategy could never be selected. Unset, tool items stay
exactly as before; auto-retained chat turns never carry a strategy either way.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    for key in ("HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID", "HINDSIGHT_MODE",
                "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_SOURCE", "HINDSIGHT_AUTO_RETAIN",
                "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    monkeypatch.setattr("pm.extras.ensure_import", lambda *args, **kwargs: None)


def _provider(tmp_path, monkeypatch, *, platform="cli", hermes_home=None, **config):
    cfg = {"mode": "cloud", "apiKey": "k", "api_url": "http://localhost:9999", "bank_id": "test-bank",
           "retain_async": False, **config}
    path = tmp_path / "hindsight" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    p = HindsightMemoryProvider()
    p.initialize(session_id="sess-1", hermes_home=str(hermes_home or tmp_path), platform=platform)
    p._client = MagicMock()
    p._client.aretain_batch = AsyncMock()
    p._client.aclose = AsyncMock()
    return p


def _items(p) -> list:
    p._retain_queue.join()
    return [item for call in p._client.aretain_batch.call_args_list for item in call.kwargs["items"]]


def _tool_retain(p, **args) -> dict:
    assert p.handle_tool_call("hindsight_retain", {"content": "a fact", **args})
    return _items(p)[-1]


def _titled_state_db(home, session_id: str, title: str) -> None:
    from hermes_state import SessionDB
    db = SessionDB(db_path=Path(home) / "state.db")
    try:
        db.create_session(session_id, "telegram")
        db.set_session_title(session_id, title)
    finally:
        db.close()


@pytest.mark.parametrize("config", [{}, {"tool_retain_strategy": ""}, {"tool_retain_strategy": "  "}])
def test_tool_retain_unconfigured_sends_no_strategy_or_envelope(tmp_path, monkeypatch, config):
    item = _tool_retain(_provider(tmp_path, monkeypatch, **config))
    assert "strategy" not in item
    md = item["metadata"]
    assert not {"source_kind", "trust", "source_origin", "source_title", "session_topic"} & set(md)
    assert md["session_id"] == "sess-1"


def test_tool_retain_configured_sends_strategy_and_agent_note_envelope(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    home.mkdir()
    _titled_state_db(home, "sess-1", "Granola OAuth move")
    p = _provider(tmp_path, monkeypatch, platform="telegram", hermes_home=home,
                  tool_retain_strategy=" agent-note ")

    item = _tool_retain(p, occurred_at="2026-11-04T10:00:00Z", tags=["project:ai-factory"])

    assert item["strategy"] == "agent-note"
    md = item["metadata"]
    assert md["source_kind"] == "agent-note" and md["trust"] == "agent"
    assert md["source_origin"] == "telegram"
    assert md["source_site_or_author"] == "hermes agent via hindsight_retain (telegram)"
    assert md["source_title"] == md["session_topic"] == "Granola OAuth move"
    assert md["occurred_at"] == "2026-11-04T10:00:00Z"
    # Writer metadata and tool args still apply.
    assert md["session_id"] == "sess-1" and md["platform"] == "telegram"
    assert item["tags"] == ["project:ai-factory"]
    assert item["timestamp"] == "2026-11-04T10:00:00Z"


def test_tool_retain_untitled_session_omits_title_keys_and_still_retains(tmp_path, monkeypatch):
    (tmp_path / "state.db").write_text("not a sqlite database")  # lookup must fail open
    item = _tool_retain(_provider(tmp_path, monkeypatch, tool_retain_strategy="agent-note"))

    assert item["strategy"] == "agent-note"
    md = item["metadata"]
    assert md["source_kind"] == "agent-note" and md["source_origin"] == "cli"
    assert not {"source_title", "session_topic", "occurred_at"} & set(md)


def test_auto_retain_never_sends_strategy(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, tool_retain_strategy="agent-note")
    p.sync_turn("hello", "hi there")
    [item] = _items(p)

    assert "strategy" not in item
    assert item["metadata"]["source_kind"] == "chat"
    assert item["metadata"]["trust"] == "self"


def test_tool_retain_strategy_is_declared_in_config_schema():
    keys = {f["key"] for f in HindsightMemoryProvider().get_config_schema()}
    assert "tool_retain_strategy" in keys
