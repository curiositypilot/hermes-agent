"""Data-class gate on Hindsight server egress (t_4ff4a177).

Topic 1720 binds a ``local-only`` class (``providers: [custom]``) per turn, yet the Hindsight plugin
retained its turns and served ``hindsight_retain``/``hindsight_reflect`` through a server whose
extraction/reflect LLM is Anthropic. The behaviour contract: a class whose allowlist omits
``server_llm_provider`` gets no auto-retain (never buffered, so a session switch flushes nothing),
no retain/reflect tool call and no reflect-method prefetch; local recall (embeddings + reranker, no
LLM) stays; ``mode: cloud`` gates recall too. The class is read per call, so binding after
``initialize()`` (as the gateway does each turn) still blocks. Lives beside
test_hindsight_provider.py, which is over the file-size cap.
"""

import contextlib
import json

import pytest

from agent.provider_policy import bind_data_class, reset_data_class
from tests.plugins.memory.test_hindsight_provider import (  # _clean_env: the suite's autouse env fixture
    _clean_env,
    _init_provider,
    _retained_items,
    _write_config,
)

_POLICY = """\
kanban:
  data_policies:
    default: internal
    classes:
      internal:
        providers: any
        fallback: true
        auxiliary: any
      confidential:
        providers: [anthropic]
        fallback: false
        auxiliary: same_provider_or_fail
      local-only:
        providers: [custom]
        fallback: false
        auxiliary: same_provider_or_fail
"""
LOCAL_ONLY_DENIAL = "data class 'local-only' does not permit Hindsight server LLM 'anthropic'"


@pytest.fixture(autouse=True)
def policy_home(request, tmp_path, monkeypatch):
    """``kanban.data_policies`` in the HERMES_HOME config.yaml ``provider_policy`` reads; no worker class."""
    request.getfixturevalue("_clean_env")  # first: it pins HERMES_DATA_CLASS=internal, removed below
    (tmp_path / "config.yaml").write_text(_POLICY, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("HERMES_DATA_CLASS", "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def debug_log(caplog):
    caplog.set_level("DEBUG", logger="plugins.memory.hindsight")
    return lambda prefix: [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


@contextlib.contextmanager
def bound(data_class):
    token = bind_data_class(data_class) if data_class else None
    try:
        yield
    finally:
        if token is not None:
            reset_data_class(token)


def _provider(tmp_path, monkeypatch, *, mode="local_external", **cfg):
    _write_config(tmp_path, monkeypatch, mode=mode, **cfg)
    p = _init_provider(tmp_path)  # initialized unbound: the gateway binds the class per turn
    p._reflect_client = p._client
    return p


def _tool(p, name, **args):
    return json.loads(p.handle_tool_call(name, args))


# -- blocked: local-only with an Anthropic server LLM ------------------------------------------


def test_local_only_turn_is_never_buffered_or_retained(tmp_path, monkeypatch, debug_log):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic", retain_every_n_turns=2)
    with bound("local-only"):
        p.sync_turn("my private note", "noted")
    assert p._session_turns == []  # the flush-on-switch below has nothing to ship
    p.on_session_switch("sess-2")
    assert _retained_items(p) == []
    assert debug_log("sync_turn: skipped") == [f"sync_turn: skipped ({LOCAL_ONLY_DENIAL})"]


@pytest.mark.parametrize("tool,arg", [("hindsight_retain", "content"), ("hindsight_reflect", "query")])
def test_local_only_retain_and_reflect_tools_error_without_a_server_call(tmp_path, monkeypatch, tool, arg):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic")
    with bound("local-only"):
        result = _tool(p, tool, **{arg: "my private note"})
    assert set(result) == {"error"} and LOCAL_ONLY_DENIAL in result["error"]
    p._client.aretain_batch.assert_not_called()
    p._client.areflect.assert_not_called()


def test_local_only_keeps_local_recall_tool_and_recall_prefetch(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic", recall_sync=True)
    with bound("local-only"):
        assert "Memory 1" in _tool(p, "hindsight_recall", query="tea")["result"]
        assert "Memory 1" in p.prefetch("tea")
    assert p._client.arecall.call_count == 2


def test_local_only_reflect_prefetch_makes_no_reflect_call(tmp_path, monkeypatch, debug_log):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic", recall_prefetch_method="reflect",
                  recall_sync=True)
    with bound("local-only"):
        assert p.prefetch("tea") == ""
    p._client.areflect.assert_not_called()
    assert debug_log("Prefetch: skipped") == [f"Prefetch: skipped ({LOCAL_ONLY_DENIAL})"]


def test_background_reflect_prefetch_sees_the_turns_class(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic", recall_prefetch_method="reflect",
                  prefetch_waits_for_retain=False)
    with bound("local-only"):
        p.queue_prefetch("tea")
    p._prefetch_thread.join(5)
    p._client.areflect.assert_not_called()


# -- must not match: a class that permits the server LLM ---------------------------------------


@pytest.mark.parametrize("data_class", [None, "confidential"], ids=["unbound-internal", "confidential"])
def test_permitted_class_retains_and_reaches_the_server(tmp_path, monkeypatch, data_class):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic")
    with bound(data_class):
        p.sync_turn("hello", "hi there")
        assert _tool(p, "hindsight_retain", content="MB prefers tea") == {"result": "Memory stored successfully."}
        assert _tool(p, "hindsight_reflect", query="tea") == {"result": "Synthesized answer"}
    assert len(_retained_items(p)) == 2
    p._client.areflect.assert_called_once()


# -- server_llm_provider unset ------------------------------------------------------------------


def test_unset_server_llm_fails_closed_under_an_allowlist_class(tmp_path, monkeypatch, debug_log):
    p = _provider(tmp_path, monkeypatch)
    with bound("local-only"):
        p.sync_turn("my private note", "noted")
        assert "error" in _tool(p, "hindsight_retain", content="x")
    assert _retained_items(p) == []
    assert debug_log("sync_turn: skipped") == [
        "sync_turn: skipped (data class 'local-only' does not permit Hindsight server LLM (server_llm_provider unset))"]


def test_unset_server_llm_still_retains_under_providers_any(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    p.sync_turn("hello", "hi there")
    assert len(_retained_items(p)) == 1


# -- per-call evaluation and cloud mode ---------------------------------------------------------


def test_class_bound_after_a_permitted_turn_blocks_the_next_one(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, server_llm_provider="anthropic")
    p.sync_turn("public", "ok")
    with bound("local-only"):
        p.sync_turn("my private note", "noted")
    p.sync_turn("public again", "ok")
    contents = [i["content"] for i in _retained_items(p)]
    assert len(contents) == 2 and not any("my private note" in c for c in contents)


def test_cloud_mode_gates_recall_too(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, mode="cloud", server_llm_provider="anthropic", recall_sync=True)
    with bound("local-only"):
        result = _tool(p, "hindsight_recall", query="tea")
        assert p.prefetch("tea") == ""
    assert "hindsight-cloud" in result["error"]
    p._client.arecall.assert_not_called()
