"""A benched Anthropic pool row must not become a keyless client.

``resolve_anthropic_token(model=...) or ""`` built the SDK client anyway. The SDK
then raised "Could not resolve authentication method", which the worker counted
as a terminal auth failure. A model cooldown is a rate limit, not a missing login.
"""

import json
import time
from types import SimpleNamespace

import pytest

OPUS = "claude-opus-4-6"
# Short on purpose: the postflight secret scan treats sk- plus 20 characters as a key.
OAUTH_TOKEN = "sk-ant-oat01-fix"
EXPLICIT_KEY = "sk-ant-api03-fix"
RATE_LIMITED = f"rate-limited for {OPUS}"
NO_CREDENTIALS = "No Anthropic credentials found"


@pytest.fixture
def benched_opus_home(tmp_path, monkeypatch):
    """One OAuth row, cooled for opus only. No env keys, no external login dir."""
    root = tmp_path / "hermes-home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    for name in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    now = time.time()
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"anthropic": [{
            "id": "oat1",
            "label": "subscription",
            "priority": 0,
            "last_status": "ok",
            "auth_type": "oauth",
            "source": "manual:hermes_pkce",
            "access_token": OAUTH_TOKEN,
            "refresh_token": "refresh-fixture",
            "expires_at_ms": int((now + 6 * 3600) * 1000),
            "model_cooldowns": {OPUS: now + 3000},
        }]},
    }))
    return root


def _empty_key_client():
    """Same shape ``_swap_fallback_clients`` reads off ``OpenAI(api_key="")``.

    The installed SDK rejects an empty key at construction, so this stands in for
    that client: a falsy ``api_key`` and no callable ``_api_key_provider``.
    """
    return SimpleNamespace(api_key="", base_url="http://localhost:1234/v1")


def test_benched_opus_init_raises_rate_limit_and_builds_no_client(benched_opus_home):
    from hermes_cli.auth import AuthError
    from agent.agent_init import _init_anthropic_client

    agent = SimpleNamespace(provider="anthropic", model=OPUS, quiet_mode=True)
    with pytest.raises(AuthError, match=RATE_LIMITED):
        _init_anthropic_client(agent, None, "https://api.anthropic.com", None)
    assert getattr(agent, "_anthropic_client", None) is None


def test_benched_opus_fallback_swap_raises_rate_limit(benched_opus_home):
    from hermes_cli.auth import AuthError
    from agent.client_lifecycle import _swap_fallback_clients

    agent = SimpleNamespace(provider="anthropic", model=OPUS)
    fb = _empty_key_client()
    with pytest.raises(AuthError, match=RATE_LIMITED):
        _swap_fallback_clients(
            agent, fb, "anthropic", OPUS, "https://api.anthropic.com", "anthropic_messages",
        )
    assert getattr(agent, "_anthropic_client", None) is None


def test_token_or_raise_names_the_bench_not_a_missing_login(benched_opus_home):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    with pytest.raises(AuthError, match=RATE_LIMITED) as exc:
        _anthropic_token_or_raise(model=OPUS)
    assert NO_CREDENTIALS not in str(exc.value)


def test_empty_pool_says_no_credentials_not_rate_limited(benched_opus_home):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    (benched_opus_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"anthropic": []},
    }))
    with pytest.raises(AuthError, match=NO_CREDENTIALS) as exc:
        _anthropic_token_or_raise(model=OPUS)
    assert "rate-limited" not in str(exc.value)


def _rewrite_row(home, **fields):
    data = json.loads((home / "auth.json").read_text())
    data["credential_pool"]["anthropic"][0].update(fields)
    (home / "auth.json").write_text(json.dumps(data))


def test_credential_wide_bench_does_not_claim_other_models(benched_opus_home):
    """t_8c736168: an aux 429 benched the whole row, yet the error said other models were fine."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    _rewrite_row(benched_opus_home, last_status="exhausted", last_status_at=time.time(),
                 last_error_code=429, model_cooldowns={})
    with pytest.raises(AuthError, match="every Claude model") as exc:
        _anthropic_token_or_raise(model=OPUS)
    assert "remain available" not in str(exc.value)
    assert is_rate_limited_auth_error(exc.value)


def test_every_benched_model_is_named(benched_opus_home):
    """09:21-13:00 on 2026-11-21 fable, opus and sonnet were all benched: name them, not "others are fine"."""
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    until = time.time() + 3000
    sonnet, fable = "claude-sonnet-4-6", "claude-fable-5"
    _rewrite_row(benched_opus_home, model_cooldowns={OPUS: until, sonnet: until, fable: until})
    with pytest.raises(AuthError, match=f"{RATE_LIMITED} until ") as exc:
        _anthropic_token_or_raise(model=OPUS)
    message = str(exc.value)
    assert f"also benched: {fable} until " in message and f"{sonnet} until " in message
    assert "remain available" not in message


def test_bench_auth_error_is_rate_limited_exit_75(benched_opus_home):
    """t_846f38f4: a per-model bench requeues the card (75), it is not a counted crash (1)."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
    from hermes_cli.runtime_provider import _anthropic_token_or_raise
    from hermes_cli.turn_exit import credential_failure_flags, turn_exit_code

    with pytest.raises(AuthError, match=RATE_LIMITED) as benched:
        _anthropic_token_or_raise(model=OPUS)
    assert is_rate_limited_auth_error(benched.value)
    flags = credential_failure_flags(benched.value)
    assert turn_exit_code(None, kanban_worker=True, **flags) == KANBAN_RATE_LIMIT_EXIT_CODE

    (benched_opus_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"anthropic": []},
    }))
    with pytest.raises(AuthError, match=NO_CREDENTIALS) as empty:
        _anthropic_token_or_raise(model=OPUS)
    assert not is_rate_limited_auth_error(empty.value)
    assert turn_exit_code(None, kanban_worker=True, **credential_failure_flags(empty.value)) == 1


def test_non_native_provider_still_builds_without_a_key(benched_opus_home):
    pytest.importorskip("anthropic")
    from agent.agent_init import _init_anthropic_client

    agent = SimpleNamespace(provider="minimax", model=OPUS, quiet_mode=True)
    _init_anthropic_client(agent, None, "https://api.minimax.io", None)
    client = agent._anthropic_client
    assert client is not None
    client.close()


def test_explicit_api_key_builds_on_a_benched_model(benched_opus_home):
    pytest.importorskip("anthropic")
    from agent.agent_init import _init_anthropic_client

    agent = SimpleNamespace(provider="anthropic", model=OPUS, quiet_mode=True)
    _init_anthropic_client(agent, EXPLICIT_KEY, "https://api.anthropic.com", None)
    client = agent._anthropic_client
    assert client is not None
    assert agent._anthropic_api_key == EXPLICIT_KEY
    client.close()


def test_model_cooldown_reads_as_quota_and_missing_login_does_not(benched_opus_home):
    """Kanban start failures and the cron preflight read quota from ``is_rate_limited_auth_error``:
    the model cooldown must requeue (75), a missing login must not look like a quota wall."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    with pytest.raises(AuthError, match=RATE_LIMITED) as benched:
        _anthropic_token_or_raise(model=OPUS)
    assert is_rate_limited_auth_error(benched.value)

    (benched_opus_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"anthropic": []},
    }))
    with pytest.raises(AuthError, match=NO_CREDENTIALS) as missing:
        _anthropic_token_or_raise(model=OPUS)
    assert not is_rate_limited_auth_error(missing.value)
