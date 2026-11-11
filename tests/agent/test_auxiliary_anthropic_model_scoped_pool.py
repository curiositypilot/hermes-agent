"""Aux pool selection is model-scoped.

An unnamed ``pool.select()`` is blocked by any active model cooldown. Aux calls
must pass the model they will send, so an Opus/Fable cooldown does not hide a
usable Haiku seat (and a cooldown on the named model still blocks it).
"""

import json

import pytest

from agent.credential_pool import load_pool
from agent.credential_pool_model_cooldowns import FAILURE_REASON_USAGE_WINDOW


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    for name in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _write_pool(tmp_path, provider, entry):
    (tmp_path / "auth.json").write_text(json.dumps({
        "credential_pool": {provider: [entry]},
    }))


@pytest.fixture
def anthropic_pool(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    _write_pool(tmp_path, "anthropic", {
        "id": "seat",
        "label": "seat",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-ant-aux-seat",
    })
    pool = load_pool("anthropic")
    assert pool.has_credentials()
    return pool


def _cool(pool, model, *, failure_reason="rate_limit"):
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"message": "rate"},
        api_key_hint="sk-ant-aux-seat",
        failure_reason=failure_reason,
        model=model,
    )


def _default_haiku():
    from agent.auxiliary_client import _anthropic_aux_model
    model = _anthropic_aux_model()
    assert model.startswith("claude-haiku"), model
    return model


def test_opus_cooldown_does_not_block_default_haiku(anthropic_pool):
    from agent.auxiliary_client import _try_anthropic

    _cool(anthropic_pool, "claude-opus-5-5")
    client, model = _try_anthropic()
    assert client is not None
    assert model == _default_haiku()


def test_named_model_cooldown_still_blocks(anthropic_pool):
    """Must-not-match: a cooldown on the model the call will send still blocks."""
    from agent.auxiliary_client import _try_anthropic

    haiku = _default_haiku()
    _cool(anthropic_pool, haiku)
    assert _try_anthropic() == (None, None)
    assert _try_anthropic(model="claude-opus-5-5")[0] is not None


def test_explicit_opus_is_blocked_only_when_opus_is_cooled(anthropic_pool):
    from agent.auxiliary_client import _try_anthropic

    _cool(anthropic_pool, "claude-sonnet-5-5")
    client, model = _try_anthropic(model="claude-opus-5-5")
    assert client is not None
    assert model == "claude-opus-5-5"

    _cool(anthropic_pool, "claude-opus-5-5")
    assert _try_anthropic(model="claude-opus-5-5") == (None, None)


def test_compression_haiku_survives_opus_and_fable_cooldown(anthropic_pool, tmp_path):
    """Replay of 2026-11-09 21:08:17Z: opus-5-5 and fable-5-2 cooled, compression is haiku-5-5."""
    from agent.auxiliary_client import (
        _client_cache,
        _client_cache_lock,
        get_text_auxiliary_client,
        resolve_provider_client,
    )

    _cool(anthropic_pool, "claude-opus-5-5")
    _cool(anthropic_pool, "claude-fable-5-2")
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: anthropic\n    model: claude-haiku-5-5\n"
    )
    with _client_cache_lock:
        _client_cache.clear()

    client, model = get_text_auxiliary_client("compression")
    assert client is not None
    assert model == "claude-haiku-5-5"

    # The routed model, not the aux default, is what the pool is asked about.
    blocked, blocked_model = resolve_provider_client("anthropic", model="claude-opus-5-5")
    assert blocked is None and blocked_model is None


def test_strict_vision_passes_the_requested_model(anthropic_pool):
    from agent.auxiliary_client import _resolve_strict_vision_backend

    _cool(anthropic_pool, "claude-opus-5-5")
    assert _resolve_strict_vision_backend("anthropic", "claude-opus-5-5") == (None, None)
    client, model = _resolve_strict_vision_backend("anthropic", "claude-haiku-5-5")
    assert client is not None
    assert model == "claude-haiku-5-5"


def test_codex_model_cooldown_does_not_block_a_sibling_model(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    token = "codex-aux-seat-token"
    _write_pool(tmp_path, "openai-codex", {
        "id": "codex-seat",
        "label": "codex-seat",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": token,
    })
    pool = load_pool("openai-codex")
    assert pool.has_credentials()
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"message": "usage"},
        api_key_hint=token,
        failure_reason=FAILURE_REASON_USAGE_WINDOW,
        model="gpt-5.4",
    )

    from agent.auxiliary_client import _build_codex_client

    other, other_model = _build_codex_client("gpt-5.4-mini")
    assert other is not None
    assert other_model == "gpt-5.4-mini"
    assert _build_codex_client("gpt-5.4") == (None, None)
