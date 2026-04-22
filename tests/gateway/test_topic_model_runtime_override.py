from gateway import run as gateway_run


def test_apply_topic_runtime_override_reresolves_provider(monkeypatch):
    runtime = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "chatgpt-account-token",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
    }
    topic_cfg = {
        "provider": "openrouter",
    }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "openrouter-key",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )

    result = gateway_run._apply_topic_runtime_override(runtime, topic_cfg)

    assert result["provider"] == "openrouter"
    assert result["base_url"] == "https://openrouter.ai/api/v1"
    assert result["api_key"] == "openrouter-key"
    assert result["api_mode"] == "chat_completions"


def test_apply_topic_runtime_override_passes_explicit_endpoint(monkeypatch):
    runtime = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "chatgpt-account-token",
        "api_mode": "codex_responses",
    }
    topic_cfg = {
        "provider": "openrouter",
        "base_url": "https://proxy.example/v1",
        "api_key": "proxy-key",
    }
    seen = {}

    def _fake_resolve_runtime_provider(**kwargs):
        seen.update(kwargs)
        return {
            "provider": "openrouter",
            "base_url": kwargs["explicit_base_url"],
            "api_key": kwargs["explicit_api_key"],
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )

    result = gateway_run._apply_topic_runtime_override(runtime, topic_cfg)

    assert seen == {
        "requested": "openrouter",
        "explicit_api_key": "proxy-key",
        "explicit_base_url": "https://proxy.example/v1",
    }
    assert result["base_url"] == "https://proxy.example/v1"
    assert result["api_key"] == "proxy-key"
