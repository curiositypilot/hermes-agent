"""HINDSIGHT_AUTO_RETAIN from a session's client env (t_248c3f9e).

A ``hermes chat`` turn runs in the shared gateway, so the shell's kill switch reaches the provider
only as the session's client env (``client_env_scope``), never as the gateway's ``os.environ``. The
switch is on when EITHER side sets it false. The provider is initialized inside the scope because
the gateway builds the session's agent there and the switch is evaluated once at ``initialize()``.
"""

import json

import pytest

from agent.session_client_env import client_env_scope
from tests.plugins.memory.test_hindsight_provider import (  # _clean_env: the suite's autouse env fixture
    _clean_env,
    _init_provider,
    _retained_items,
    _write_config,
)

SKIPPED = "sync_turn: skipped (HINDSIGHT_AUTO_RETAIN=0 kill switch)"


@pytest.fixture()
def debug_log(caplog):
    caplog.set_level("DEBUG", logger="plugins.memory.hindsight")
    return lambda: [r.getMessage() for r in caplog.records if r.getMessage().startswith("sync_turn: skipped")]


def _provider_in_scope(tmp_path, monkeypatch, env, *, local=True):
    _write_config(tmp_path, monkeypatch)
    with client_env_scope(env, local=local):
        p = _init_provider(tmp_path)
        p.sync_turn("hello", "hi there")
    return p


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_client_switch_disables_auto_retain_but_not_the_tool(tmp_path, monkeypatch, debug_log, value, local):
    p = _provider_in_scope(tmp_path, monkeypatch, {"HINDSIGHT_AUTO_RETAIN": value}, local=local)
    assert _retained_items(p) == []
    assert debug_log() == [SKIPPED]

    result = json.loads(p.handle_tool_call("hindsight_retain", {"content": "MB prefers tea"}))
    assert result == {"result": "Memory stored successfully."}
    assert [i["content"] for i in _retained_items(p)] == ["MB prefers tea"]


@pytest.mark.parametrize("env", [{}, {"HINDSIGHT_AUTO_RETAIN": "1"}, {"HINDSIGHT_AUTO_RETAIN_X": "0"}],
                         ids=["client-silent", "client-one", "look-alike-name"])
def test_client_env_without_the_switch_keeps_auto_retain(tmp_path, monkeypatch, debug_log, env):
    p = _provider_in_scope(tmp_path, monkeypatch, env)
    assert len(_retained_items(p)) == 1
    assert debug_log() == []


def test_gateway_switch_still_blocks_a_silent_client(tmp_path, monkeypatch, debug_log):
    monkeypatch.setenv("HINDSIGHT_AUTO_RETAIN", "0")
    p = _provider_in_scope(tmp_path, monkeypatch, {})
    assert _retained_items(p) == []
    assert debug_log() == [SKIPPED]
