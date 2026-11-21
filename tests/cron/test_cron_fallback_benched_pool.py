"""Resolve-time fallback walkers skip a candidate whose credential pool is benched.

Incident replay (kanban-orchestrator, 2026-11-21): Anthropic was rate-limited, the chain fell to
``xai-oauth``, whose only pool row sat in a billing bench (403 spending-limit). The resolve-time
walkers still picked it through the singleton OAuth rung, so every fire spent a call on the account
that had just refused it and the operator was told to sign in again. The in-loop ladder already
skipped such a candidate (``_candidate_pool_exhausted``, #89401); both walkers now share its rule
(``hermes_cli.fallback_config.benched_pool_until``).
"""

import base64
import json
import time
from unittest.mock import patch

import pytest

from cron.scheduler import _CronJobConfig, _resolve_job_runtime
from hermes_cli import runtime_provider as rp
from hermes_cli.auth import AuthError
from hermes_constants import get_hermes_home

_PRIMARY_ERROR = ("Anthropic credentials are rate-limited for claude-opus-5-5 "
                  "(pool cooldown until 10:20 UTC)")
_CHAIN = [{"provider": "anthropic", "model": "claude-sonnet-5"},
          {"provider": "xai-oauth", "model": "grok-4.7"}]
_BILLING_BENCH_S = 3600  # first billing mark: a one-hour bench (agent/credential_pool.py)


def _jwt(tag):
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return b64({"alg": "none"}) + "." + b64({"exp": int(time.time()) + 86400, "sub": tag}) + ".sig"


def _write_store(*, benched_for=None):
    """auth.json as on the incident host: the xai-oauth singleton plus one device_code pool row,
    billing-benched ``benched_for`` seconds ago (None: the row is ok)."""
    access = _jwt("a1")
    row = {"id": "bc0708", "label": "device_code", "auth_type": "oauth", "priority": 0,
           "source": "device_code", "access_token": access, "refresh_token": "r1",
           "base_url": "https://api.x.ai/v1", "last_refresh": "2026-11-21T09:45:00Z",
           "request_count": 77, "last_status": "ok", "last_status_at": None}
    if benched_for is not None:
        at = time.time() - benched_for
        row.update({"last_status": "exhausted", "last_status_at": at, "last_error_code": 403,
                    "last_error_reason": "personal-team-blocked:spending-limit",
                    "failure_reason": "billing", "billing_streak": 1, "billing_streak_at": at})
    store = {"version": 1,
             "providers": {"xai-oauth": {
                 "tokens": {"access_token": access, "refresh_token": "r1"},
                 "last_refresh": "2026-11-21T09:45:00Z", "auth_mode": "oauth_device_code",
                 "discovery": {"token_endpoint": "https://invalid.invalid/token"}}},
             "credential_pool": {"xai-oauth": [row]}}
    (get_hermes_home() / "auth.json").write_text(json.dumps(store), encoding="utf-8")


@pytest.fixture
def incident(monkeypatch):
    """Anthropic (primary and sonnet entry) refuses at resolve; xai-oauth resolves for real off
    the tmp auth store. No token refresh may reach the network."""
    from hermes_cli import auth

    def no_refresh(*_a, **_k):
        raise AssertionError("token refresh attempted")

    monkeypatch.setattr(auth, "_refresh_xai_oauth_tokens", no_refresh)
    real = rp.resolve_runtime_provider

    def resolve(**kwargs):
        if (kwargs.get("requested") or "anthropic") == "anthropic":
            raise AuthError(_PRIMARY_ERROR)
        return real(**kwargs)

    monkeypatch.setattr(rp, "resolve_runtime_provider", resolve)


def _cron_resolve(chain=_CHAIN):
    jc = _CronJobConfig(cfg={"fallback_providers": list(chain)}, model="claude-opus-5-5",
                        model_cfg={"provider": "anthropic"}, cron_default_provider="")
    job = {"id": "580e25b75ad8", "name": "kanban-orchestrator", "provider": None, "model": None,
           "base_url": None}
    return _resolve_job_runtime(job, job["id"], jc)


def _walker_resolve(chain=_CHAIN):
    return rp.resolve_runtime_with_fallback({"fallback_providers": list(chain)},
                                            requested="anthropic", target_model="claude-opus-5-5")


def test_cron_benched_candidate_is_skipped_and_the_primary_error_surfaces(incident):
    _write_store(benched_for=600)
    with pytest.raises(RuntimeError, match="rate-limited for claude-opus-5-5"):
        _cron_resolve()


def test_walker_benched_candidate_is_skipped_and_the_primary_error_surfaces(incident):
    _write_store(benched_for=600)
    with pytest.raises(AuthError, match="rate-limited for claude-opus-5-5"):
        _walker_resolve()


def test_available_candidate_still_resolves(incident):
    _write_store(benched_for=None)
    runtime, model = _cron_resolve()
    assert (runtime["provider"], model) == ("xai-oauth", "grok-4.7")
    runtime, entry = _walker_resolve()
    assert runtime["provider"] == "xai-oauth" and (entry or {}).get("model") == "grok-4.7"


def test_bench_about_to_end_is_not_skipped(incident):
    """A bench with 60 s left is shorter than the retry loop's longest wait: still a candidate."""
    _write_store(benched_for=_BILLING_BENCH_S - 60)
    from hermes_cli.fallback_config import benched_pool_until
    assert benched_pool_until("xai-oauth", "grok-4.7") is None
    runtime, model = _cron_resolve()
    assert (runtime["provider"], model) == ("xai-oauth", "grok-4.7")
    runtime, _entry = _walker_resolve()
    assert runtime["provider"] == "xai-oauth"


def test_entry_with_its_own_key_is_not_skipped(incident):
    """An inline key does not use the pool, so the pool's bench does not apply to it."""
    _write_store(benched_for=600)
    chain = [{"provider": "xai-oauth", "model": "grok-4.7", "api_key": "xai-inline-key"}]
    resolved = []
    real = rp.resolve_runtime_provider

    def spy(**kwargs):
        resolved.append(kwargs)
        if kwargs.get("requested") == "xai-oauth":
            return {"provider": "xai-oauth", "api_key": kwargs.get("explicit_api_key"),
                    "base_url": "https://api.x.ai/v1", "api_mode": "chat_completions"}
        return real(**kwargs)

    with patch.object(rp, "resolve_runtime_provider", side_effect=spy):
        runtime, model = _cron_resolve(chain)
        assert (runtime["provider"], runtime["api_key"], model) == (
            "xai-oauth", "xai-inline-key", "grok-4.7")
        runtime, _entry = _walker_resolve(chain)
        assert runtime["api_key"] == "xai-inline-key"
