"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import (
    effective_runtime_provider, resolve_entry_api_key, same_provider_only_chain)


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"


    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None


    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_resolves_from_active_secret_scope_not_raw_env(self, monkeypatch):
        # Multiplexed gateway: os.environ holds another profile's key, but the
        # active per-turn secret scope holds this profile's key. The scoped
        # value must win — a raw os.getenv() would leak the other profile's
        # credential (issue #74311).
        monkeypatch.setenv("FB_KEY", "fake-other-profile-key")
        token = set_secret_scope({"FB_KEY": "fake-active-profile-key"})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "fake-active-profile-key"
        finally:
            reset_secret_scope(token)

    def test_key_env_falls_back_to_env_when_no_active_scope(self, monkeypatch):
        # Non-multiplexed / single-profile behavior must be unchanged: with no
        # secret scope installed, resolution still reads os.environ.
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"


class TestEffectiveRuntimeProvider:
    """Named custom fallback entries must keep their configured identity (#98739)."""

    def test_named_custom_entry_keeps_configured_id(self):
        entry = {"provider": "my-custom-provider", "model": "some-model"}
        runtime = {"provider": "custom", "requested_provider": "my-custom-provider"}
        assert effective_runtime_provider(entry, runtime) == "my-custom-provider"

    def test_requested_provider_missing_falls_back_to_entry(self):
        entry = {"provider": "my-custom-provider", "model": "some-model"}
        runtime = {"provider": "custom"}
        assert effective_runtime_provider(entry, runtime) == "my-custom-provider"

    def test_builtin_provider_untouched(self):
        entry = {"provider": "openrouter", "model": "glm"}
        runtime = {"provider": "openrouter", "requested_provider": "openrouter"}
        assert effective_runtime_provider(entry, runtime) == "openrouter"

    def test_genuinely_bare_custom_stays_custom(self):
        # Ad-hoc endpoint: user literally configured provider: custom.
        entry = {"provider": "custom", "model": "some-model"}
        runtime = {"provider": "custom", "requested_provider": "custom"}
        assert effective_runtime_provider(entry, runtime) == "custom"

    def test_none_inputs_are_safe(self):
        assert effective_runtime_provider(None, None) == ""


class TestSameProviderOnlyChain:
    """A chain that never leaves the primary's endpoint cannot take over when it fails (#133454)."""

    CODEX_ONLY = [{"provider": "openai-codex", "model": "a"}, {"provider": "openai-codex", "model": "b"}]

    def test_empty_chain_is_false(self):
        assert same_provider_only_chain([], "openai-codex") is False

    def test_unknown_primary_fails_open(self):
        assert same_provider_only_chain(self.CODEX_ONLY, "") is False
        assert same_provider_only_chain(self.CODEX_ONLY, None) is False

    def test_same_label_any_case_is_true(self):
        assert same_provider_only_chain(self.CODEX_ONLY, "OpenAI-Codex") is True

    def test_mixed_chain_is_false(self):
        chain = [self.CODEX_ONLY[0], {"provider": "anthropic", "model": "c"}]
        assert same_provider_only_chain(chain, "openai-codex") is False

    def test_same_label_on_two_distinct_explicit_base_urls_is_false(self):
        chain = [{"provider": "custom", "model": "m", "base_url": "http://host-b:8000/v1"}]
        assert same_provider_only_chain(chain, "custom", "http://host-a:8000/v1") is False
        # ...and the same explicit URL is the same endpoint.
        assert same_provider_only_chain(chain, "custom", "http://host-b:8000/v1") is True
