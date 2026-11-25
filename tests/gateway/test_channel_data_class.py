"""A chat's ``channel_overrides.<id>.data_class`` binds that data class for its turns (intake t_36fb835b).

Privacy topic 1720 pinned its main model to a local endpoint, but every auxiliary call (compression,
title, vision, web_extract) still followed the profile's cloud ``auxiliary.*`` routes: no gateway path
bound a class, so a chat always ran as ``internal``. These tests drive the real ``GatewayRunner``
binding methods and the real auxiliary resolver against a temp ``HERMES_HOME``.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from agent import auxiliary_client as ac
from agent.provider_policy import ProviderDenied, current_data_class
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from tools.thread_context import propagate_context_to_thread

CHAT = "-1003882011045"
LOCAL_RUNTIME = {
    "provider": "custom", "model": "huihui-local.gguf",
    "base_url": "http://127.0.0.1:8081/v1", "api_key": "no-key-required", "api_mode": "chat_completions",
}

_CONFIG = """
model:
  default: claude-opus-5-5
  provider: anthropic
auxiliary:
  compression:
    provider: anthropic
    model: claude-haiku-5-5
kanban:
  data_policies:
    default: internal
    classes:
      internal:
        providers: any
        fallback: true
        auxiliary: any
      local-only:
        providers: [custom]
        fallback: false
        auxiliary: same_provider_or_fail
"""


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_KANBAN_TASK", "HERMES_DATA_CLASS", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    ac._aux_denials_seen.clear()
    yield home
    ac._aux_denials_seen.clear()


def _runner(overrides: dict) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides=overrides)})
    return runner


def _privacy_runner(data_class: str = "local-only") -> GatewayRunner:
    return _runner({
        "1720": ChannelOverride(provider="custom:local", model="huihui-local.gguf", data_class=data_class),
        "4003": ChannelOverride(system_prompt="You are a visual analysis specialist."),
    })


def _source(thread_id=None, chat_id=CHAT) -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="group",
                         user_id="1", thread_id=thread_id)


def _context(source: SessionSource) -> SessionContext:
    return SessionContext(source=source, connected_platforms=[], home_channels={})


def test_session_env_binds_the_channel_class_and_clear_resets_it():
    runner = _privacy_runner()
    tokens = runner._set_session_env(_context(_source("1720")))
    try:
        assert current_data_class() == "local-only"
    finally:
        runner._clear_session_env(tokens)
    assert current_data_class() == "internal"


def test_profile_scope_binds_the_channel_class():
    runner = _privacy_runner()
    with runner._profile_scope_for_source(_source("1720")):
        assert current_data_class() == "local-only"
    assert current_data_class() == "internal"


def test_async_profile_scope_binds_the_channel_class():
    runner = _privacy_runner()

    async def _turn():
        async with runner._async_profile_scope_for_source(_source("1720")):
            inside = current_data_class()
        return inside, current_data_class()

    assert asyncio.run(_turn()) == ("local-only", "internal")


@pytest.mark.parametrize("source", [_source("4003"), _source(None), _source("17", chat_id="-100999")],
                         ids=["override-without-class", "no-thread", "other-chat"])
def test_chats_without_a_channel_class_stay_internal(source):
    runner = _privacy_runner()
    tokens = runner._set_session_env(_context(source))
    try:
        assert current_data_class() == "internal"
        with runner._profile_scope_for_source(source):
            assert current_data_class() == "internal"
    finally:
        runner._clear_session_env(tokens)


def test_unknown_channel_class_fails_closed():
    runner = _privacy_runner(data_class="no-such-class")
    with pytest.raises(ProviderDenied):
        runner._set_session_env(_context(_source("1720")))
    with pytest.raises(ProviderDenied):
        with runner._profile_scope_for_source(_source("1720")):
            pass
    assert current_data_class() == "internal"


def test_bound_class_reaches_a_propagated_thread():
    runner = _privacy_runner()
    seen = []
    with runner._profile_scope_for_source(_source("1720")):
        worker = threading.Thread(target=propagate_context_to_thread(lambda: seen.append(current_data_class())))
        worker.start()
        worker.join()
    assert seen == ["local-only"]


def test_compression_route_stays_on_the_local_endpoint():
    """The 1720 transcript's compression call: configured anthropic, rerouted to the 8081 endpoint."""
    runner = _privacy_runner()
    with runner._profile_scope_for_source(_source("1720")), ac.scoped_runtime_main(LOCAL_RUNTIME):
        route = ac._enforce_data_class_on_task_route(
            ("anthropic", "claude-haiku-5-5", None, None, None), caller_chose_route=False)
        client, model = ac.resolve_provider_client(route[0], route[1], main_runtime=LOCAL_RUNTIME)
    assert route == ("custom", None, None, None, None)
    assert client is not None
    assert "127.0.0.1:8081" in str(client.base_url)
    assert model == "huihui-local.gguf"


def test_compression_route_is_unchanged_outside_the_privacy_chat():
    runner = _privacy_runner()
    configured = ("anthropic", "claude-haiku-5-5", None, None, None)
    with runner._profile_scope_for_source(_source("4003")), ac.scoped_runtime_main(LOCAL_RUNTIME):
        assert ac._enforce_data_class_on_task_route(configured, caller_chose_route=False) == configured
