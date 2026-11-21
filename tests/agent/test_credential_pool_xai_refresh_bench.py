"""A token refresh does not end an xai-oauth billing bench (t_7f8066a9).

xAI device-code JWTs live ~15 min, and any process (cron fallback, auxiliary client, a pinned
job) refreshes the ``providers.xai-oauth`` singleton. The pool used to treat every rotated pair as
a re-login: it cleared the 403 spending-limit bench and dropped the billing streak, so the spent
account was probed again after every rotation and the 1 h -> 24 h escalation never happened. Only
a login stamped after the bench (``logged_in_at``) ends it now; the rotated pair is still adopted.

Every test drives the real pool against a temp ``HERMES_HOME`` with a fake clock.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime

import pytest

MIN = 60
HOUR = 3600
T0 = 1_791_000_000.0
ROW_ID = "bc0708"


class _Clock:
    def __init__(self, now: float = T0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(time, "time", fake)
    return fake


def _jwt(tag: str) -> str:
    def b64(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()
    return f"{b64({'alg': 'none'})}.{b64({'exp': int(T0) + 30 * 24 * HOUR, 'sub': tag})}.sig"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def home(tmp_path, monkeypatch, clock):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in ("XAI_API_KEY", "HERMES_XAI_BASE_URL", "XAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

    def no_network(*_a, **_k):
        raise AssertionError("token refresh attempted")
    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", no_network)
    return hermes_home


A1, R1 = _jwt("a1"), "r1"
A2, R2 = _jwt("a2"), "r2"


def _billing_bench(at: float, streak: int = 1) -> dict:
    return {"last_status": "exhausted", "last_status_at": at, "last_error_code": 403,
            "last_error_reason": "personal-team-blocked:spending-limit", "failure_reason": "billing",
            "billing_streak": streak, "billing_streak_at": at}


def _write(home, *, singleton: tuple[str, str], row: tuple[str, str], row_status: dict | None = None,
           logged_in_at: str | None = None, provider: str = "xai-oauth") -> None:
    state = {"tokens": {"access_token": singleton[0], "refresh_token": singleton[1]},
             "last_refresh": _iso(time.time()), "auth_mode": "oauth_device_code"}
    if logged_in_at is not None:
        state["logged_in_at"] = logged_in_at
    entry = {"id": ROW_ID, "label": "device_code", "auth_type": "oauth", "priority": 0,
             "source": "device_code", "access_token": row[0], "refresh_token": row[1],
             "base_url": "https://api.x.ai/v1", "last_refresh": _iso(time.time() - 30 * MIN)}
    entry.update(row_status or {})
    store = {"version": 1, "providers": {provider: state}, "credential_pool": {provider: [entry]}}
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")


def _rotate_singleton(home, access: str, refresh: str, *, logged_in_at: str | None = None) -> None:
    store = json.loads((home / "auth.json").read_text())
    state = store["providers"]["xai-oauth"]
    state["tokens"] = {"access_token": access, "refresh_token": refresh}
    state["last_refresh"] = _iso(time.time())
    if logged_in_at is not None:
        state["logged_in_at"] = logged_in_at
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")


def _load(provider: str = "xai-oauth"):
    from agent.credential_pool import load_pool

    return load_pool(provider)


def _row(pool):
    return next(e for e in pool.entries() if e.id == ROW_ID)


def _disk_row(home, provider: str = "xai-oauth") -> dict:
    rows = json.loads((home / "auth.json").read_text())["credential_pool"][provider]
    return next(r for r in rows if r["id"] == ROW_ID)


def _bench_until(entry) -> float:
    from agent.credential_pool import _exhausted_until

    until = _exhausted_until(entry)
    assert until is not None
    return until


def test_refresh_rotation_keeps_billing_bench(home, clock):
    """Replay of 12:32:07: bench marked 16 min ago, the singleton holds a refreshed pair."""
    bench_at = T0 - 16 * MIN
    _write(home, singleton=(A2, R2), row=(A1, R1), row_status=_billing_bench(bench_at))

    pool = _load()
    entry = _row(pool)

    assert entry.last_status == "exhausted"
    assert pool.has_available() is False
    assert (entry.access_token, entry.refresh_token) == (A2, R2)
    assert entry.billing_streak == 1
    assert entry.extra["billing_streak_at"] == bench_at
    disk = _disk_row(home)
    assert disk["last_status"] == "exhausted"
    assert (disk["access_token"], disk["refresh_token"]) == (A2, R2)
    assert disk["billing_streak"] == 1


def test_refresh_rotation_keeps_billing_bench_on_a_loaded_pool(home, clock):
    """Site 1: a live pool re-reads the singleton on every selection of an exhausted row."""
    bench_at = T0 - 16 * MIN
    _write(home, singleton=(A1, R1), row=(A1, R1), row_status=_billing_bench(bench_at))
    pool = _load()
    assert pool.has_available() is False

    _rotate_singleton(home, A2, R2)

    assert pool.select() is None
    entry = _row(pool)
    assert entry.last_status == "exhausted"
    assert (entry.access_token, entry.refresh_token) == (A2, R2)
    assert entry.billing_streak == 1
    assert _bench_until(entry) == bench_at + HOUR


def test_unparseable_login_stamp_counts_as_refresh(home, clock):
    _write(home, singleton=(A2, R2), row=(A1, R1), row_status=_billing_bench(T0 - 16 * MIN),
           logged_in_at="not-a-time")

    pool = _load()

    assert _row(pool).last_status == "exhausted"
    assert pool.has_available() is False


@pytest.mark.parametrize("site", ["load_pool", "loaded_pool"])
def test_login_rotation_clears_billing_bench(home, clock, site):
    """A login after the bench start (but before now) may be a topped-up account: clear it."""
    bench_at = T0 - 16 * MIN
    login_at = _iso(T0 - 5 * MIN)
    if site == "load_pool":
        _write(home, singleton=(A2, R2), row=(A1, R1), row_status=_billing_bench(bench_at),
               logged_in_at=login_at)
        pool = _load()
    else:
        _write(home, singleton=(A1, R1), row=(A1, R1), row_status=_billing_bench(bench_at))
        pool = _load()
        _rotate_singleton(home, A2, R2, logged_in_at=login_at)
        assert pool.select() is not None

    entry = _row(pool)
    assert entry.last_status != "exhausted"
    assert pool.has_available() is True
    assert entry.access_token == A2
    if site == "load_pool":
        assert "billing_streak" not in entry.extra
        assert "billing_streak_at" not in entry.extra


def test_login_before_the_bench_does_not_clear_it(home, clock):
    _write(home, singleton=(A2, R2), row=(A1, R1), row_status=_billing_bench(T0 - 16 * MIN),
           logged_in_at=_iso(T0 - 3 * HOUR))

    assert _load().has_available() is False


def test_refresh_rotation_clears_a_non_billing_exhaustion(home, clock):
    """A 401 bench is about the old token: a refreshed pair is the fix, so it clears as before."""
    status = {"last_status": "exhausted", "last_status_at": T0 - 10 * MIN, "last_error_code": 401,
              "last_error_reason": "invalid_token"}
    _write(home, singleton=(A2, R2), row=(A1, R1), row_status=status)

    pool = _load()

    assert _row(pool).last_status is None
    assert pool.has_available() is True


def test_ok_row_adopts_the_refreshed_pair_and_stays_ok(home, clock):
    _write(home, singleton=(A2, R2), row=(A1, R1), row_status={"last_status": "ok"})

    pool = _load()
    entry = _row(pool)

    assert entry.access_token == A2
    assert entry.last_status in (None, "ok")
    assert pool.has_available() is True


def test_streak_escalates_across_refreshes(home, clock):
    """Bench at T, rotation at T+16 min, bench expires, re-mark at T+61 min -> 2 h bench."""
    _write(home, singleton=(A1, R1), row=(A1, R1))
    pool = _load()
    assert pool.select() is not None
    pool.mark_exhausted_and_rotate(status_code=403, credential_id=ROW_ID, failure_reason="billing",
                                   error_context={"message": "personal-team-blocked:spending-limit"})
    assert _row(pool).billing_streak == 1

    clock.now = T0 + 16 * MIN
    _rotate_singleton(home, A2, R2)
    pool = _load()
    assert pool.has_available() is False
    assert _row(pool).billing_streak == 1

    clock.now = T0 + 61 * MIN
    pool = _load()
    assert pool.select() is not None
    pool.mark_exhausted_and_rotate(status_code=403, credential_id=ROW_ID, failure_reason="billing",
                                   error_context={"message": "personal-team-blocked:spending-limit"})
    entry = _row(pool)
    assert entry.billing_streak == 2
    assert _bench_until(entry) == clock.now + 2 * HOUR


def test_incident_sequence_reaches_xai_only_after_the_bench(home, clock):
    """Marks at 12:16:11, a refresh before every 16-min cron fire: no fire reaches xAI before 13:16:11."""
    bench_at = T0
    _write(home, singleton=(A1, R1), row=(A1, R1), row_status=_billing_bench(bench_at))
    for n, fire in enumerate((16, 32, 48), start=1):
        clock.now = bench_at + fire * MIN
        _rotate_singleton(home, _jwt(f"fire-{n}"), f"r-fire-{n}")
        assert _load().has_available() is False, f"fire at +{fire} min reached xAI"
    clock.now = bench_at + 64 * MIN
    _rotate_singleton(home, _jwt("fire-4"), "r-fire-4")
    assert _load().has_available() is True


def test_auth_add_row_serves_while_the_device_code_row_stays_benched(home, clock):
    """``hermes auth add xai-oauth`` adds its own ``manual:device_code`` row (no singleton save):
    that new login is selectable at once although the singleton row keeps its bench."""
    from agent.credential_pool import PooledCredential

    _write(home, singleton=(A2, R2), row=(A1, R1), row_status=_billing_bench(T0 - 16 * MIN))
    pool = _load()
    added = pool.add_entry(PooledCredential(
        provider="xai-oauth", id="add001", label="added", auth_type="oauth", priority=0,
        source="manual:device_code", access_token=_jwt("added"), refresh_token="r-added",
        base_url="https://api.x.ai/v1"))

    assert _row(pool).last_status == "exhausted"
    selected = pool.select()
    assert selected is not None and selected.id == added.id


def test_codex_rotation_still_clears_billing_bench(home, clock):
    """openai-codex has no login stamp: any rotation keeps clearing the bench (out of scope)."""
    _write(home, singleton=(A2, R2), row=(A1, R1), row_status=_billing_bench(T0 - 16 * MIN),
           provider="openai-codex")

    pool = _load("openai-codex")
    entry = next(e for e in pool.entries() if e.source == "device_code")

    assert entry.access_token == A2
    assert entry.last_status is None
    assert "billing_streak" not in entry.extra
