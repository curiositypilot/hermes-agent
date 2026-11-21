"""Regression for t_282f4dff (2026-11-21): one OAuth login plus a keyless ``env:ANTHROPIC_API_KEY`` row.

1. The runtime counted the keyless row as a second credential, so the model-less 429 mark written at
   12:37:12 benched the only login for 60 min while Kanban routing (which drops the row) gave it 60 s:
   59 worker runs failed at start between 13:00 and 13:35.
2. The usage fetch resolved its token through the bench, so the quota snapshot was blind 09:22-13:35.

Pool rebuilt from the record (dummy tokens); each instant is replayed with the clock frozen.
"""
from __future__ import annotations

import calendar
import json
import time
from unittest import mock

import pytest

FABLE, OPUS, SONNET = "claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5"
OAUTH_TOKEN = "cc-incident-dummy"
ROW_MARK_AT = 1795264632.0  # 12:37:12, auxiliary title_generation 429: whole row, no model, no reset time


def _at(hms: str) -> float:
    return float(calendar.timegm(time.strptime(f"2026-11-21 {hms}", "%Y-%m-%d %H:%M:%S")))


def _oauth_row(*, row_mark: bool = False, **extra) -> dict:
    row = {"id": "6afabf", "label": "anthropic-oauth-1", "auth_type": "oauth", "priority": 0,
           "source": "manual:hermes_pkce", "base_url": "https://api.anthropic.com",
           "access_token": OAUTH_TOKEN, "refresh_token": "dummy-refresh", "expires_at_ms": 1795400000000,
           "last_status": "ok",
           "model_cooldowns": {FABLE: 1795266000.0, SONNET: 1795266000.0, OPUS: 1795265999.0}}
    if row_mark:
        row.update(last_status="exhausted", last_status_at=ROW_MARK_AT, last_error_code=429,
                   last_error_message="HTTP 429", last_error_reset_at=None)
    row.update(extra)
    return row


ENV_ROW = {"id": "2c315c", "label": "ANTHROPIC_API_KEY", "auth_type": "api_key", "priority": 1,
           "source": "env:ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com", "last_status": "ok"}

# (instant, row mark present, expected verdict for workers and routing alike)
INSTANTS = [("09:46:49", False, False), ("12:30:00", False, False), ("12:59:00", True, False),
            ("13:00:59", True, True), ("13:17:10", True, True), ("13:34:23", True, True),
            ("13:38:00", True, True)]


@pytest.fixture
def write_pool(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)

    def write(rows: list[dict]) -> None:
        (home / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {}, "credential_pool": {"anthropic": rows},
            "suppressed_sources": {"anthropic": ["env:ANTHROPIC_API_KEY"]}}), encoding="utf-8")
    return write


def _frozen(now: float):
    return mock.patch("time.time", return_value=now)


@pytest.mark.parametrize("hms,row_mark,expected", INSTANTS)
def test_routing_and_workers_agree_at_every_instant(write_pool, hms, row_mark, expected):
    from agent.anthropic_credentials import resolve_anthropic_token
    from hermes_cli.kanban_routing import pool_availability

    now = _at(hms)
    write_pool([_oauth_row(row_mark=row_mark), ENV_ROW])
    with _frozen(now):
        for model in (FABLE, OPUS):
            routing = pool_availability("anthropic", model, now=now).available
            worker = bool(resolve_anthropic_token(model=model))
            assert (routing, worker) == (expected, expected), (hms, model)


def _usage_payload(url, headers, timeout=15.0):
    assert headers["Authorization"] == f"Bearer {OAUTH_TOKEN}"
    return {"five_hour": {"utilization": 100.0, "resets_at": "2026-11-21T13:00:00+00:00"}}


@pytest.mark.parametrize("hms,row_mark,_expected", INSTANTS)
def test_usage_snapshot_stays_live_through_the_bench(write_pool, monkeypatch, hms, row_mark, _expected):
    from agent import account_usage

    monkeypatch.setattr(account_usage, "_get_json", _usage_payload)
    write_pool([_oauth_row(row_mark=row_mark), ENV_ROW])
    with _frozen(_at(hms)):
        snapshot = account_usage.fetch_account_usage("anthropic")
    assert snapshot is not None
    assert [w.used_percent for w in snapshot.windows] == [100.0]


def test_dead_only_row_gets_no_usage_token(write_pool):
    from agent.anthropic_credentials import resolve_anthropic_usage_token

    write_pool([_oauth_row(last_status="dead"), ENV_ROW])
    assert resolve_anthropic_usage_token() is None


def test_two_keyed_rows_are_not_sole_and_keep_the_full_429_bench(write_pool):
    from agent.credential_pool import load_pool

    second = _oauth_row(id="7b0c1d", label="anthropic-oauth-2", priority=1, access_token="cc-second-dummy",
                        model_cooldowns={})
    write_pool([_oauth_row(row_mark=True, model_cooldowns={}), second, ENV_ROW])
    with _frozen(_at("13:17:10")):  # 40 min after the mark: inside the 60 min bench
        pool = load_pool("anthropic")
        assert pool._is_sole_credential() is False
        entries, _ = pool._available_entries(clear_expired=False, refresh=False)
    assert [e.id for e in entries] == ["7b0c1d"]


def test_model_less_resolve_still_honours_a_model_bench(write_pool):
    from agent.anthropic_credentials import resolve_anthropic_token

    write_pool([_oauth_row(), ENV_ROW])
    with _frozen(_at("12:30:00")):
        assert resolve_anthropic_token() is None
