"""A generic Anthropic 429 benches only the model that was rate-limited (#111769, #61451).

Real ``agent.credential_pool`` against a real temp auth store: one API-key credential,
one ``mark_exhausted_and_rotate(status_code=429, model=A)``.
"""
import json

import pytest

KEY = "sk-ant-api03-synthetic-test-key-0000"
MODEL_A = "claude-sonnet-4-5"
MODEL_B = "claude-haiku-4-5"


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "seat", "label": "seat", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": KEY,
    }]}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def test_generic_429_benches_only_the_rate_limited_model(pool, monkeypatch):
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool

    ctx = {"message": "This request would exceed your account's rate limit. Please try again later."}
    assert pool.mark_exhausted_and_rotate(
        status_code=429, error_context=ctx, api_key_hint=KEY, failure_reason="rate_limit", model=MODEL_A,
    ) is None  # a sole credential has nothing to rotate to for model A

    assert pool.entries()[0].last_status is None  # credential-wide state untouched
    assert pool.select(model=MODEL_A) is None
    assert pool.select(model=MODEL_B) is not None
    assert pool.select() is None  # a caller that names no model honours every active cooldown
    assert pool.next_available_at(model=MODEL_A) is not None
    assert pool.next_available_at(model=MODEL_B) is None

    # The cooldown is persisted, so another process (and the env/borrowed token resolver,
    # which reads the store fresh) sees the same per-model verdict.
    fresh = load_pool("anthropic")
    assert fresh.select(model=MODEL_A) is None and fresh.select(model=MODEL_B) is not None
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert resolve_anthropic_token(model=MODEL_A) is None
    assert resolve_anthropic_token(model=MODEL_B) == KEY
    assert resolve_anthropic_token() == KEY  # model-less diagnostics keep the key


@pytest.mark.parametrize("status_code, failure_reason", [(401, None), (402, "billing"), (429, "billing")])
def test_auth_and_billing_failures_stay_credential_wide(pool, status_code, failure_reason):
    from agent.credential_pool import STATUS_EXHAUSTED

    pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason, model=MODEL_A,
    )
    assert pool.entries()[0].last_status == STATUS_EXHAUSTED
    assert not pool.entries()[0].model_cooldowns
    assert pool.select(model=MODEL_B) is None


SONNET, OPUS, FABLE = "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"
OAUTH_TOKEN = "sk-ant-oat01-" + "x" * 40


@pytest.fixture
def oauth_pool(pool, tmp_path):
    """One manual OAuth row with per-model cooldowns: sonnet's lapsed, opus/fable active (the 21:19 incident)."""
    import time
    now = time.time()
    auth = tmp_path / "hermes-root" / "auth.json"
    auth.write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "oauth-1", "label": "oauth-1", "auth_type": "oauth", "priority": 0, "source": "manual",
        "access_token": OAUTH_TOKEN, "refresh_token": "rt-" + "y" * 20,
        "expires_at_ms": int((now + 86400) * 1000),
        "model_cooldowns": {SONNET: now - 86400, OPUS: now + 3000, FABLE: now + 3000},
    }]}}))
    return now


def test_pool_token_follows_requested_model_replay(oauth_pool, monkeypatch):
    """A cooldown on Opus/Fable must not bench the shared OAuth row for Sonnet (t_155f5ad6)."""
    import time
    from agent.anthropic_credentials import resolve_anthropic_token

    assert resolve_anthropic_token(model=SONNET) == OAUTH_TOKEN
    assert resolve_anthropic_token(model=OPUS) is None
    assert resolve_anthropic_token(model=FABLE) is None
    assert resolve_anthropic_token() is None  # model-less diagnostics stay conservative

    monkeypatch.setattr(time, "time", lambda: oauth_pool + 4000)
    for model in (SONNET, OPUS, FABLE):
        assert resolve_anthropic_token(model=model) == OAUTH_TOKEN


def test_pool_token_refuses_model_with_its_own_cooldown(oauth_pool, tmp_path):
    from agent.anthropic_credentials import resolve_anthropic_token

    auth = tmp_path / "hermes-root" / "auth.json"
    data = json.loads(auth.read_text())
    data["credential_pool"]["anthropic"][0]["model_cooldowns"][SONNET] = oauth_pool + 3000
    auth.write_text(json.dumps(data))
    assert resolve_anthropic_token(model=SONNET) is None
    assert resolve_anthropic_token(model=MODEL_B) == OAUTH_TOKEN
