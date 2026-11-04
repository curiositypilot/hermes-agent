"""Connecting an MCP server that is disabled in config logs who did it (pid + caller).

Regression for intake t_53c1ac88: an unattributed process connected a disabled OAuth server,
refreshed its grant and registered a client, re-creating token files its owner had removed to
disconnect it. The connect is not refused (`hermes mcp test|login` on a disabled entry is an
explicit user action); the log line names the writer next time.
"""
import asyncio
import logging
import os

import pytest

from tools import mcp_tool_config
from tools import mcp_tool_discovery as discovery


@pytest.fixture
def stub_server(monkeypatch):
    class _StubServerTask:
        def __init__(self, name):
            self.name = name

        async def start(self, config):
            pass

        async def shutdown(self):
            pass

    # Origin state is read through _core; patch the origin module (see mcp_tool_discovery docstring).
    monkeypatch.setattr("tools.mcp_tool.MCPServerTask", _StubServerTask)


async def _probe_like_caller(name, config):
    """Stands in for a probe coroutine scheduled onto the MCP loop (e.g. hermes_cli.mcp_config)."""
    return await asyncio.wait_for(discovery._connect_server(name, config), timeout=5)


def _disabled_lines(caplog):
    return [r.getMessage() for r in caplog.records if "is disabled in config" in r.getMessage()]


def test_connecting_a_disabled_server_logs_pid_and_caller(stub_server, monkeypatch, caplog):
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config",
                        lambda: {"granola": {"url": "https://mcp.example/mcp", "enabled": False}})
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        # The caller passes an entry without ``enabled``: the on-disk config is what decides.
        asyncio.run(_probe_like_caller("granola", {"command": "true"}))

    lines = _disabled_lines(caplog)
    assert len(lines) == 1, lines
    assert "'granola'" in lines[0]
    assert f"pid {os.getpid()}" in lines[0]
    assert f"caller {__name__}:_probe_like_caller" in lines[0]


def test_connecting_with_a_disabled_entry_logs_even_when_absent_on_disk(stub_server, monkeypatch, caplog):
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", dict)
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        asyncio.run(_probe_like_caller("granola", {"command": "true", "enabled": False}))

    assert len(_disabled_lines(caplog)) == 1


def test_connecting_an_enabled_server_logs_nothing(stub_server, monkeypatch, caplog):
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config",
                        lambda: {"granola": {"command": "true"}})
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        asyncio.run(_probe_like_caller("granola", {"command": "true"}))

    assert _disabled_lines(caplog) == []
