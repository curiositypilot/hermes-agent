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


# --- Account-wide subscription walls (t_e0c3a616) ------------------------------------------------

ACCOUNT_MSG = "This request would exceed your account's rate limit. Please try again later."
ALL_MODELS = (OPUS, SONNET, FABLE)


def _write_row(tmp_path, auth_type, token, now=None):
    import time
    now = time.time() if now is None else now
    row = {"id": "row", "label": "row", "auth_type": auth_type, "priority": 0, "source": "manual",
           "access_token": token}
    if auth_type == "oauth":
        row["expires_at_ms"] = int((now + 86400) * 1000)
    (tmp_path / "hermes-root" / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [row]}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _assert_walled(pool, reset):
    from agent.credential_pool import STATUS_EXHAUSTED
    row = pool.entries()[0]
    assert row.last_status == STATUS_EXHAUSTED
    assert row.last_error_reset_at == reset
    assert not row.model_cooldowns
    for model in ALL_MODELS:
        assert pool.select(model=model) is None


def test_oauth_account_wall_benches_every_model(pool, tmp_path, monkeypatch):
    import time
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool

    reset = time.time() + 3 * 3600
    oauth = _write_row(tmp_path, "oauth", OAUTH_TOKEN)
    assert oauth.mark_exhausted_and_rotate(
        status_code=429, error_context={"message": ACCOUNT_MSG, "reset_at": reset},
        api_key_hint=OAUTH_TOKEN, failure_reason="rate_limit", model=OPUS,
    ) is None
    _assert_walled(oauth, reset)
    fresh = load_pool("anthropic")
    assert fresh.limit_state(list(ALL_MODELS))["scope"] == "account"
    # The env path hands out the same token without selecting it: the account bench must gate it.
    monkeypatch.setenv("ANTHROPIC_TOKEN", OAUTH_TOKEN)
    assert resolve_anthropic_token(model=SONNET) is None
    assert resolve_anthropic_token() == OAUTH_TOKEN  # model-less diagnostics keep the token


def test_account_wall_error_names_the_account_not_other_models(pool, tmp_path, monkeypatch):
    import time
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    oauth = _write_row(tmp_path, "oauth", OAUTH_TOKEN)
    oauth.mark_exhausted_and_rotate(
        status_code=429, error_context={"message": ACCOUNT_MSG, "reset_at": time.time() + 3600},
        api_key_hint=OAUTH_TOKEN, failure_reason="rate_limit", model=OPUS,
    )
    monkeypatch.setenv("ANTHROPIC_TOKEN", OAUTH_TOKEN)
    with pytest.raises(AuthError) as raised:
        _anthropic_token_or_raise(model=SONNET)
    text = str(raised.value)
    assert text.startswith("Anthropic account is rate-limited until ") and " UTC " in text
    assert "remain available" not in text and "not listed here" not in text


def _unified_error(claim, reset, message=ACCOUNT_MSG):
    from unittest.mock import MagicMock
    err = Exception(f"HTTP 429: {message}")
    err.body = {"type": "error", "error": {"type": "rate_limit_error", "message": message}}
    err.response = MagicMock(headers={
        "anthropic-ratelimit-unified-status": "rejected",
        "anthropic-ratelimit-unified-representative-claim": claim,
        "anthropic-ratelimit-unified-reset": str(int(reset)),
    })
    return err


@pytest.mark.parametrize("auth_type, token", [("api_key", KEY), ("oauth", OAUTH_TOKEN)])
@pytest.mark.parametrize("claim, account_wide", [("five_hour", True), ("seven_day", True), ("seven_day_opus", False)])
def test_unified_claim_decides_scope(pool, tmp_path, auth_type, token, claim, account_wide):
    import time
    from agent.agent_runtime_helpers import extract_api_error_context

    reset = int(time.time()) + 7200
    row_pool = _write_row(tmp_path, auth_type, token)
    row_pool.mark_exhausted_and_rotate(
        status_code=429, error_context=extract_api_error_context(_unified_error(claim, reset)),
        api_key_hint=token, failure_reason="rate_limit", model=OPUS,
    )
    if account_wide:
        _assert_walled(row_pool, reset)
    else:  # a model-family claim stays a model cooldown, even with the account body on an OAuth row
        row = row_pool.entries()[0]
        assert row.last_status is None
        assert row.model_cooldowns == {OPUS: reset}
        assert row_pool.select(model=OPUS) is None
        assert row_pool.select(model=SONNET) is not None


@pytest.mark.parametrize("message", [
    "This request would exceed the rate limit for your organization (org) of 50,000 input tokens per minute",
    "Number of request tokens has exceeded your per-minute rate limit",
    "rate limit for your organization's account",
    "",
])
def test_other_oauth_429_bodies_stay_model_scoped(pool, tmp_path, message):
    oauth = _write_row(tmp_path, "oauth", OAUTH_TOKEN)
    oauth.mark_exhausted_and_rotate(
        status_code=429, error_context={"message": message}, api_key_hint=OAUTH_TOKEN,
        failure_reason="rate_limit", model=OPUS,
    )
    row = oauth.entries()[0]
    assert row.last_status is None and OPUS in row.model_cooldowns
    assert oauth.select(model=SONNET) is not None


# 2026-11-21 incident: logs/agent.log + data/quota/history.jsonl (session window 100% at 04:58,
# resets 08:00; 100% at 09:11, resets 13:00). (instant, failed model, window reset), all UTC epochs.
INCIDENT_20261121 = [
    (1795237448, OPUS, 1795248000),    # 05:04:08 opus
    (1795237534, SONNET, 1795248000),  # 05:05:34 sonnet, same wall
    (1795253453, OPUS, 1795266000),    # 09:30:53 opus
    (1795254055, SONNET, 1795266000),  # 09:40:55 sonnet fallback onto the same walled row
]


@pytest.mark.parametrize("now, model, reset", INCIDENT_20261121)
def test_incident_20261121_account_wall_replay(pool, tmp_path, monkeypatch, now, model, reset):
    import time
    monkeypatch.setattr(time, "time", lambda: float(now))
    oauth = _write_row(tmp_path, "oauth", OAUTH_TOKEN, now=now)
    oauth.mark_exhausted_and_rotate(
        status_code=429, error_context={"message": ACCOUNT_MSG, "reset_at": reset},
        api_key_hint=OAUTH_TOKEN, failure_reason="rate_limit", model=model,
    )
    _assert_walled(oauth, reset)
