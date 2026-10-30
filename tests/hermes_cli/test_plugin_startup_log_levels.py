"""Per-process plugin startup events log at DEBUG; audit allows and failures stay loud.

Every Hermes process runs full plugin discovery, so a successful provider registration or a
default capability denial logged at INFO repeats ~40 times per process start in agent.log.
These tests pin the level contract: routine registration -> DEBUG, capability allow -> INFO,
registration failure -> WARNING.
"""
from __future__ import annotations

import logging

import pytest

from hermes_cli.dashboard_auth import registry as auth_registry
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, LoginStart, Session
from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

CAP_LOGGER = "hermes_cli.plugin_capabilities"
PLUGINS_LOGGER = "hermes_cli.plugins"
AUTH_LOGGER = "hermes_cli.dashboard_auth.registry"


def _records(caplog, logger_name: str, needle: str):
    return [r for r in caplog.records if r.name == logger_name and needle in r.getMessage()]


def test_capability_deny_is_debug_and_allow_is_info(caplog):
    caplog.set_level(logging.DEBUG, logger=CAP_LOGGER)

    assert plugin_capability_granted("probe", "tools.override", config={"plugins": {"entries": {}}}) is False
    deny = _records(caplog, CAP_LOGGER, "decision=deny")
    assert deny and all(r.levelno == logging.DEBUG for r in deny)
    assert not [r for r in caplog.records if r.name == CAP_LOGGER and r.levelno >= logging.INFO]

    caplog.clear()
    granted = {"plugins": {"entries": {"probe": {"granted_capabilities": ["tools.override"]}}}}
    assert plugin_capability_granted("probe", "tools.override", config=granted) is True
    allow = _records(caplog, CAP_LOGGER, "decision=allow")
    assert allow and all(r.levelno == logging.INFO for r in allow)


class _AuthStub(DashboardAuthProvider):
    name = "startup-log-probe"
    display_name = "Startup Log Probe"

    def start_login(self, *, redirect_uri):
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        return Session("u", "e", "n", "o", self.name, 0, "a", "r")

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        return Session("u", "e", "n", "o", self.name, 0, "a", "r")

    def revoke_session(self, *, refresh_token):
        return None


@pytest.fixture
def auth_stub():
    provider = _AuthStub()
    yield provider
    auth_registry.unregister_global_provider(provider.name, provider)


def test_dashboard_auth_registration_logs_at_debug(caplog, auth_stub):
    caplog.set_level(logging.DEBUG, logger=AUTH_LOGGER)

    auth_registry.register_provider(auth_stub)

    assert auth_registry.get_provider(auth_stub.name) is auth_stub
    assert not [r for r in caplog.records if r.name == AUTH_LOGGER and r.levelno >= logging.INFO]
    assert [r for r in _records(caplog, AUTH_LOGGER, "registered") if r.levelno == logging.DEBUG]


def test_plugin_provider_registration_is_debug_but_failure_still_warns(caplog):
    from agent.image_gen_provider import ImageGenProvider
    from agent.image_gen_registry import restore_registration, snapshot_registration

    class _ImageProbe(ImageGenProvider):
        @property
        def name(self):
            return "startup_log_probe"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"prompt": prompt}

    caplog.set_level(logging.DEBUG, logger=PLUGINS_LOGGER)
    manager = PluginManager()
    scope = manager.scope_key
    baseline = snapshot_registration("startup_log_probe", scope=scope)
    ctx = PluginContext(PluginManifest(name="startup-log-probe", key="startup-log-probe"), manager)
    try:
        provider = _ImageProbe()
        assert ctx.register_image_gen_provider(provider) is not None
        registered = _records(caplog, PLUGINS_LOGGER, "registered")
        assert registered and all(r.levelno == logging.DEBUG for r in registered)

        caplog.clear()
        assert ctx.register_image_gen_provider(object()) is None
        warnings = [r for r in caplog.records if r.name == PLUGINS_LOGGER and r.levelno == logging.WARNING]
        assert warnings and "does not inherit" in warnings[0].getMessage()
    finally:
        manager.unload("startup-log-probe")
        current = snapshot_registration("startup_log_probe", scope=scope)
        if current is not baseline and current is not None:
            restore_registration("startup_log_probe", current, baseline, scope=scope)
