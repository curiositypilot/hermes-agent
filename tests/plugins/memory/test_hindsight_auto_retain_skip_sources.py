"""``auto_retain_skip_sources``: session sources whose turns are never auto-retained (t_4d646807).

Agent-run ``hermes chat -q`` / ``-z`` sessions (source ``oneshot``) and ``--source tool`` runs were
auto-retained into bank ``main`` like MB's chats: the provider sees platform ``cli`` and
agent_context ``primary`` for both. The source comes from the gateway's frozen launch policy when a
turn runs there, else from the process (``HERMES_SINGLE_QUERY_SESSION=1`` makes an in-process ``-q``
``oneshot``). The provider is initialized inside the scope because the block is evaluated once at
``initialize()``. Lives beside test_hindsight_provider.py, which is over the file-size cap.
"""

import contextlib
import json
import os

import pytest

from gateway.session_policy import build_policy, policy_scope
from tests.plugins.memory.test_hindsight_provider import (  # _clean_env: the suite's autouse env fixture
    _clean_env,
    _init_provider,
    _retained_items,
    _write_config,
)

SKIP = ["oneshot", "tool"]


@pytest.fixture(autouse=True)
def _no_session_source_env(monkeypatch):
    # The suite may run inside a Kanban worker or a ``-q`` child carrying these.
    for name in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def debug_log(caplog):
    caplog.set_level("DEBUG", logger="plugins.memory.hindsight")
    return lambda: [r.getMessage() for r in caplog.records if r.getMessage().startswith("sync_turn: skipped")]


def _turn(tmp_path, monkeypatch, *, source=None, env=None, **cfg):
    """One turn of a session of gateway *source* (None: no policy, in-process) with *env* set."""
    _write_config(tmp_path, monkeypatch, **cfg)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    scope = (policy_scope(build_policy({"source": source, "cwd": os.getcwd(), "model": "m"}, {}))
             if source else contextlib.nullcontext())
    with scope:
        p = _init_provider(tmp_path)
    p.sync_turn("hello", "hi there")
    return p


@pytest.mark.parametrize("source", ["oneshot", "tool"])
def test_gateway_session_of_a_skipped_source_does_not_auto_retain(tmp_path, monkeypatch, debug_log, source):
    p = _turn(tmp_path, monkeypatch, source=source, auto_retain_skip_sources=SKIP)
    assert _retained_items(p) == []
    assert debug_log() == [f"sync_turn: skipped (session source {source!r} in auto_retain_skip_sources)"]


def test_in_process_single_query_is_oneshot_and_the_tool_still_stores(tmp_path, monkeypatch, debug_log):
    p = _turn(tmp_path, monkeypatch, env={"HERMES_SINGLE_QUERY_SESSION": "1"}, auto_retain_skip_sources=SKIP)
    assert _retained_items(p) == []
    assert debug_log() == ["sync_turn: skipped (session source 'oneshot' in auto_retain_skip_sources)"]

    result = json.loads(p.handle_tool_call("hindsight_retain", {"content": "MB prefers tea"}))
    assert result == {"result": "Memory stored successfully."}
    assert [i["content"] for i in _retained_items(p)] == ["MB prefers tea"]


@pytest.mark.parametrize("source", ["cli", "gui", "tui", "acp"])
def test_interactive_gateway_sources_keep_auto_retain(tmp_path, monkeypatch, debug_log, source):
    p = _turn(tmp_path, monkeypatch, source=source, auto_retain_skip_sources=SKIP)
    assert len(_retained_items(p)) == 1
    assert debug_log() == []


@pytest.mark.parametrize("env", [{}, {"HERMES_SINGLE_QUERY_SESSION": "1", "HERMES_SESSION_SOURCE": "telegram"}],
                         ids=["interactive-chat", "single-query-from-telegram"])
def test_in_process_sessions_of_other_sources_keep_auto_retain(tmp_path, monkeypatch, debug_log, env):
    p = _turn(tmp_path, monkeypatch, env=env, auto_retain_skip_sources=SKIP)
    assert len(_retained_items(p)) == 1
    assert debug_log() == []


@pytest.mark.parametrize("cfg", [{}, {"auto_retain_skip_sources": []}], ids=["key-absent", "empty-list"])
def test_no_skip_list_keeps_oneshot_auto_retain(tmp_path, monkeypatch, cfg):
    p = _turn(tmp_path, monkeypatch, source="oneshot", **cfg)
    assert len(_retained_items(p)) == 1


@pytest.mark.parametrize(("source", "retained"), [("oneshot", 0), ("cli", 1)])
def test_csv_string_names_sources(tmp_path, monkeypatch, source, retained):
    p = _turn(tmp_path, monkeypatch, source=source, auto_retain_skip_sources="oneshot")
    assert len(_retained_items(p)) == retained


def test_look_alike_source_is_not_skipped(tmp_path, monkeypatch, debug_log):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "oneshot2")
    p = _turn(tmp_path, monkeypatch, auto_retain_skip_sources="oneshot")
    assert len(_retained_items(p)) == 1
    assert debug_log() == []
