"""A Codex ChatGPT-account model entitlement 400 benches nothing and rotates nothing (fork).

Plans gain models as they roll out, so a rejection today says nothing about tomorrow: the
credential stays healthy for every model, no ``model_cooldowns`` row is written, and the
session falls back and skips the slug until restart (#106475 marker). Every other 400 stays
a plain request failure.
"""
import json
import types
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason, classify_api_error

MODEL = "gpt-5.3-codex"
OTHER_MODEL = "gpt-5.3-codex-mini"
TOKENS = ("tok-account-a", "tok-account-b")


class _Err(Exception):
    def __init__(self, status, body):
        self.status_code = status
        self.body = body
        self.response = types.SimpleNamespace(status_code=status, headers={}, text=json.dumps(body), json=lambda: body)
        self.message = f"Error code: {status} - {json.dumps(body)}"
        super().__init__(self.message)


def _entitlement_400():
    return _Err(400, {"detail": f"The '{MODEL}' model is not supported when using Codex with a ChatGPT account."})


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": f"cred-{i}", "label": f"acct-{i}", "auth_type": "oauth", "priority": i, "source": "manual",
         "access_token": tok, "refresh_token": f"rt-{i}", "expires_at_ms": 4_000_000_000_000}
        for i, tok in enumerate(TOKENS)
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("openai-codex")


def test_entitlement_400_falls_back_without_rotating_or_benching(pool):
    verdict = classify_api_error(_entitlement_400(), provider="openai-codex", model=MODEL)
    assert verdict.reason == FailoverReason.model_entitlement
    assert verdict.should_fallback and not verdict.should_rotate_credential and not verdict.retryable

    generic = classify_api_error(_Err(400, {"detail": "Invalid request: bad field"}), provider="openai-codex", model=MODEL)
    assert generic.reason == FailoverReason.format_error and not generic.should_rotate_credential

    # Drive the production recovery entry point (turn recovery -> recover_with_credential_pool):
    # the verdict must not reach the pool at all.
    assert pool.select(model=MODEL).id == "cred-0"
    agent = types.SimpleNamespace(
        provider="openai-codex", model=MODEL, base_url="https://chatgpt.com/backend-api/codex",
        api_key=TOKENS[0], _credential_pool=pool, _credential_pool_entry_id="cred-0",
        _swap_credential=MagicMock(return_value=True),
    )
    assert recover_with_credential_pool(
        agent, status_code=400, has_retried_429=False, classified_reason=verdict.reason,
    ) == (False, False)
    agent._swap_credential.assert_not_called()
    for entry in pool.entries():
        assert entry.last_status is None and not entry.model_cooldowns
    assert pool.select(model=MODEL).id == "cred-0"
    assert pool.select(model=OTHER_MODEL).id == "cred-0"


def test_first_rejection_marks_the_slug_for_the_session_even_with_a_pool(pool):
    from agent.fallback_cooldown import _is_entitlement_rejected, _mark_entitlement_rejected_model

    agent = types.SimpleNamespace(
        provider="openai-codex", model=MODEL, _credential_pool=pool,
        _buffer_diagnostic_status=lambda *_a, **_k: None,
    )
    assert _mark_entitlement_rejected_model(agent, _entitlement_400()) is True
    assert _is_entitlement_rejected(agent, "openai-codex", MODEL)
    assert not _is_entitlement_rejected(agent, "openai-codex", OTHER_MODEL)
    # Session-only: the pool on disk carries no trace, so a fresh session probes the model again.
    assert all(not e.model_cooldowns for e in pool.entries())
