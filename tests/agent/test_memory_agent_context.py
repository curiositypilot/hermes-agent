"""Regression for #80646: the ``agent_context`` handed to memory providers follows the platform
(``cron`` / ``subagent`` skip writes per the ``MemoryProvider.initialize`` contract) instead of a
hardcoded ``"primary"`` that let cron turns land in stores configured to skip them.
"""

from types import SimpleNamespace

import pytest

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _memory_provider_init_kwargs


def _fake_agent():
    """The attribute surface ``_memory_provider_init_kwargs`` reads."""
    return SimpleNamespace(
        session_id="sess-80646", _session_db=None, _emit_warning=None, _emit_status=None,
        session_cwd=None, **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
    )


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("cron", "cron"), ("subagent", "subagent"), ("telegram", "primary"), (None, "primary")],
)
def test_agent_context_follows_the_platform(platform, expected):
    assert _memory_provider_init_kwargs(_fake_agent(), platform)["agent_context"] == expected


def test_cron_session_disables_supermemory_writes(tmp_path, monkeypatch):
    """Through the real bundled provider: the scheduler's kwargs must switch writes off,
    an interactive session's must leave them on (empty hermes_home → config defaults)."""
    from plugins.memory.supermemory import SupermemoryMemoryProvider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    by_platform = {}
    for platform in ("cron", "cli"):
        provider = SupermemoryMemoryProvider()
        provider.initialize(**_memory_provider_init_kwargs(_fake_agent(), platform))
        by_platform[platform] = provider._write_enabled
    assert by_platform == {"cron": False, "cli": True}


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, None), ({}, None), ({"external_prefetch_timeout_seconds": 3.5}, 3.5),
     ({"external_prefetch_timeout_seconds": "12"}, 12.0),
     ({"external_prefetch_timeout_seconds": 0}, None), ({"external_prefetch_timeout_seconds": -1}, None),
     ({"external_prefetch_timeout_seconds": "nope"}, None)],
)
def test_external_prefetch_timeout_config_reader(configured, expected):
    """``memory.external_prefetch_timeout_seconds`` → MemoryManager ctor; None = keep the 8 s default."""
    from agent.agent_init import _external_prefetch_timeout

    assert _external_prefetch_timeout(configured) == expected


def test_external_prefetch_timeout_default_matches_registered_config():
    """The DEFAULT_CONFIG value and the MemoryManager fallback agree, so an unset key and the
    registered default are the same wait."""
    from agent.memory_manager import MemoryManager
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    registered = DEFAULT_CONFIG["memory"]["external_prefetch_timeout_seconds"]
    assert MemoryManager()._external_prefetch_timeout == registered
    assert MemoryManager(external_prefetch_timeout=registered / 2)._external_prefetch_timeout == registered / 2
