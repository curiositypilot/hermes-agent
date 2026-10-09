"""Billing benches escalate while billing keeps failing (t_76146656).

A depleted account (xAI 403 spending-limit, a 402) does not refill in an hour. With a flat
1 h billing bench every router re-picked the spent key each hour, paid one failed request
and fell back: 43 billing fallbacks on 26 cards from one xai-oauth credential in 39 hours. The
bench now doubles per consecutive billing mark (1 h, 2 h, 4 h, ...) up to a daily probe, and
``hermes auth reset`` / a non-billing mark / a rotated secret end the streak.

Every test drives the real pool against a temp ``HERMES_HOME`` with a fake clock.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

HOUR = 3600
DAY = 24 * HOUR
T0 = 1_791_000_000.0


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


def _write_store(tmp_path, entry_overrides: dict | None = None) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": "cred-1",
        "label": "api-key-1",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-test",
        "base_url": "https://api.deepseek.com",
    }
    entry.update(entry_overrides or {})
    store = {"version": 1, "credential_pool": {"deepseek": [entry]}}
    (hermes_home / "auth.json").write_text(json.dumps(store, indent=2), encoding="utf-8")


def _disk_entry(tmp_path) -> dict:
    store = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = store["credential_pool"]["deepseek"]
    assert len(entries) == 1, entries
    return entries[0]


@pytest.fixture
def home(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _write_store(tmp_path)
    return tmp_path


def _pool():
    from agent.credential_pool import load_pool

    return load_pool("deepseek")


def _bill(pool, status_code: int = 403, failure_reason: str | None = "billing"):
    pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint="sk-test", failure_reason=failure_reason,
        error_context={"message": "personal-team-blocked:spending-limit"},
    )
    return pool.entries()[0]


def _bench(entry) -> float:
    from agent.credential_pool import _exhausted_until

    until = _exhausted_until(entry)
    assert until is not None
    return until - entry.last_status_at


def _probe_after_bench(pool, clock, entry, slack: float = 60.0):
    """Advance past the bench and let ``select`` (clear_expired=True) lift it to OK."""
    from agent.credential_pool import STATUS_OK

    clock.now = entry.last_status_at + _bench(entry) + slack
    selected = pool.select()
    assert selected is not None
    assert selected.last_status == STATUS_OK
    assert selected.last_status_at is None  # _MARK_OK clears it; the streak must not depend on it
    return selected


def test_consecutive_billing_marks_double_the_bench(home, clock):
    pool = _pool()
    benches = []
    entry = _bill(pool)
    benches.append(_bench(entry))
    for _ in range(2):
        entry = _probe_after_bench(pool, clock, entry)
        assert entry.billing_streak is not None  # survives _MARK_OK
        entry = _bill(pool)
        benches.append(_bench(entry))
    assert benches == [HOUR, 2 * HOUR, 4 * HOUR]
    assert _disk_entry(home)["billing_streak"] == 3


def test_bench_is_capped_at_a_day_and_holds_there(home, clock):
    """Post-cap probes keep the daily bench (the continuation window is measured from the
    end of the previous bench, so a 24 h bench does not sawtooth back to 1 h)."""
    pool = _pool()
    entry = _bill(pool)
    benches = [_bench(entry)]
    for _ in range(8):
        entry = _probe_after_bench(pool, clock, entry)
        entry = _bill(pool)
        benches.append(_bench(entry))
    assert max(benches) == DAY
    assert benches[-3:] == [DAY, DAY, DAY]


def test_a_mark_long_after_the_previous_bench_restarts_the_streak(home, clock):
    from agent.credential_pool import BILLING_STREAK_CONTINUE_SECONDS

    pool = _pool()
    entry = _bill(pool)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool)
    assert _bench(entry) == 2 * HOUR
    clock.now = entry.last_status_at + _bench(entry) + BILLING_STREAK_CONTINUE_SECONDS + 60
    assert pool.select() is not None
    entry = _bill(pool)
    assert _bench(entry) == HOUR
    assert entry.billing_streak == 1


def test_a_mark_inside_a_running_bench_does_not_escalate(home, clock):
    """Requests already in flight when the first one failed are the same probe."""
    pool = _pool()
    first = _bill(pool)
    clock.now += 30
    second = _bill(pool)
    assert second.billing_streak == first.billing_streak == 1
    assert _bench(second) == HOUR


def test_a_non_billing_mark_ends_the_streak(home, clock):
    pool = _pool()
    entry = _bill(pool)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool)
    assert entry.billing_streak == 2
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool, status_code=429, failure_reason="rate_limit")
    assert entry.billing_streak is None and entry.billing_streak_at is None
    assert "billing_streak" not in _disk_entry(home)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool)
    assert _bench(entry) == HOUR


def test_unverified_billing_keeps_the_short_cooldown_and_never_escalates(home, clock):
    from agent.credential_pool import EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS

    pool = _pool()
    for _ in range(3):
        entry = _bill(pool, status_code=400, failure_reason="billing_unverified")
        assert _bench(entry) == EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS
        assert entry.billing_streak is None
        entry = _probe_after_bench(pool, clock, entry)


def test_402_without_a_classifier_verdict_escalates(home, clock):
    pool = _pool()
    entry = _bill(pool, status_code=402, failure_reason=None)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool, status_code=402, failure_reason=None)
    assert _bench(entry) == 2 * HOUR


def test_streak_round_trips_through_to_dict_and_from_dict(home, clock):
    from agent.credential_pool import PooledCredential

    pool = _pool()
    entry = _bill(pool)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool)
    again = PooledCredential.from_dict("deepseek", entry.to_dict())
    assert again.billing_streak == 2
    assert again.billing_streak_at == entry.billing_streak_at
    assert _bench(again) == 2 * HOUR
    reloaded = _pool().entries()[0]  # a fresh process reads the same bench
    assert _bench(reloaded) == 2 * HOUR


@pytest.mark.parametrize("reset", ["all", "one"])
def test_auth_reset_clears_the_streak(home, clock, reset):
    pool = _pool()
    entry = _bill(pool)
    entry = _probe_after_bench(pool, clock, entry)
    _bill(pool)
    clock.now += 5
    admin = _pool()
    if reset == "all":
        assert admin.reset_statuses() == 1
    else:
        assert admin.reset_status("cred-1") is not None
    disk = _disk_entry(home)
    assert "billing_streak" not in disk and "billing_streak_at" not in disk
    clock.now += 5
    entry = _bill(_pool())
    assert _bench(entry) == HOUR


def test_reset_from_another_process_clears_a_live_pools_streak(home, clock):
    """The live pool honours the reset (``_resync_stale_entry``) and its next flush
    neither resurrects the cooldown nor the streak."""
    live = _pool()
    entry = _bill(live)
    entry = _probe_after_bench(live, clock, entry)
    _bill(live)
    clock.now += 5
    assert _pool().reset_statuses() == 1
    clock.now += 5
    live._persist()
    assert "billing_streak" not in _disk_entry(home)
    selected = live.select()
    assert selected is not None
    assert selected.billing_streak is None
    clock.now += 5
    entry = _bill(live)
    assert _bench(entry) == HOUR


def test_a_stale_writer_keeps_another_processs_newer_streak(home, clock):
    """A writer holding an older snapshot must not overwrite a newer streak on disk."""
    stale = _pool()
    worker = _pool()
    entry = _bill(worker)
    entry = _probe_after_bench(worker, clock, entry)
    entry = _bill(worker)
    assert _disk_entry(home)["billing_streak"] == 2
    clock.now += 5
    stale._persist()  # snapshot from before both marks
    disk = _disk_entry(home)
    assert disk["billing_streak"] == 2
    assert disk["billing_streak_at"] == entry.billing_streak_at


def test_a_stale_writer_does_not_resurrect_a_streak_another_process_ended(home, clock):
    """The mirror image: a writer holding its own older streak must not restore it over
    another process's newer non-billing mark, which ended the streak on purpose."""
    stale = _pool()
    entry = _bill(stale)
    entry = _probe_after_bench(stale, clock, entry)
    entry = _bill(stale)
    assert _disk_entry(home)["billing_streak"] == 2
    clock.now += 5
    other = _pool()
    _bill(other, status_code=429, failure_reason="rate_limit")
    assert "billing_streak" not in _disk_entry(home)
    clock.now += 5
    stale._persist()  # snapshot still carries streak 2
    disk = _disk_entry(home)
    assert "billing_streak" not in disk and "billing_streak_at" not in disk
    fresh = _pool()
    entry = fresh.entries()[0]
    assert entry.billing_streak is None
    entry = _probe_after_bench(fresh, clock, entry)
    entry = _bill(fresh)
    assert _bench(entry) == HOUR
    assert entry.billing_streak == 1


def test_a_rotated_secret_drops_the_streak(home, clock):
    from agent.credential_pool import _upsert_entry

    pool = _pool()
    entry = _bill(pool)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool)
    assert entry.billing_streak == 2
    entries = list(pool.entries())
    assert _upsert_entry(entries, "deepseek", "manual", {"access_token": "sk-new"})
    assert entries[0].billing_streak is None and entries[0].billing_streak_at is None
    assert entries[0].last_status is None


def test_pool_availability_reports_the_escalated_bench(home, clock):
    from hermes_cli.kanban_routing import pool_availability

    pool = _pool()
    entry = _bill(pool)
    entry = _probe_after_bench(pool, clock, entry)
    entry = _bill(pool)
    assert entry.last_error_reset_at is None  # a reset header would override the TTL
    avail = pool_availability("deepseek", "deepseek-chat", now=entry.last_status_at + HOUR + 60)
    assert avail.available is False
    assert avail.until == entry.last_status_at + 2 * HOUR


# ── Replay of the xai-oauth incident ────────────────────────────────────────
# Pinned from the board: every ``provider_fallback`` event from xai-oauth with reason=billing,
# 2026-10-06 21:00 .. 2026-10-08 12:00 UTC (``sqlite3 -readonly ~/.hermes/kanban.db "select
# task_id, datetime(created_at,'unixepoch') from task_events where kind='provider_fallback' and
# payload like '%\"from_provider\": \"xai-oauth\"%' and payload like '%\"reason\": \"billing\"%'"``):
# 43 events, 26 cards, 36 distinct seconds. The first row is the first mark.
#
# The replay models the credential-pool gate only: a route reaches xai when ``select`` hands
# out the credential, and every probe fails billing again (the account stays empty). Events
# inside a running bench on the real board (a worker's in-run primary restore every ~16 min on
# t_0963c119 / t_223daa8b; parallel spawns in the same second) came from paths the pool gate
# did not stop; the replay refuses them under both rules, so the two counts compare the bench
# rules, not the board total.
_EVENTS = [
    ("2026-10-06 21:02:36", "t_223daa8b"),
    ("2026-10-06 21:18:39", "t_223daa8b"),
    ("2026-10-06 21:21:40", "t_9210012f"),
    ("2026-10-06 21:21:40", "t_9b365cbd"),
    ("2026-10-06 21:21:40", "t_a1744ab7"),
    ("2026-10-06 21:21:40", "t_d198f919"),
    ("2026-10-06 21:34:44", "t_223daa8b"),
    ("2026-10-06 21:50:56", "t_223daa8b"),
    ("2026-10-06 22:16:13", "t_19515a6a"),
    ("2026-10-06 22:17:15", "t_0963c119"),
    ("2026-10-06 22:33:21", "t_0963c119"),
    ("2026-10-06 22:54:28", "t_0963c119"),
    ("2026-10-06 23:10:45", "t_0963c119"),
    ("2026-10-06 23:16:45", "t_9b365cbd"),
    ("2026-10-06 23:16:45", "t_d198f919"),
    ("2026-10-06 23:26:48", "t_0963c119"),
    ("2026-10-06 23:42:54", "t_0963c119"),
    ("2026-10-06 23:59:01", "t_0963c119"),
    ("2026-10-07 00:15:06", "t_0963c119"),
    ("2026-10-07 00:17:07", "t_9b365cbd"),
    ("2026-10-07 00:17:07", "t_d198f919"),
    ("2026-10-07 00:31:12", "t_0963c119"),
    ("2026-10-07 01:17:30", "t_9b365cbd"),
    ("2026-10-07 01:17:30", "t_d198f919"),
    ("2026-10-07 01:34:38", "t_8ef71af3"),
    ("2026-10-07 01:50:45", "t_b9588260"),
    ("2026-10-07 02:44:05", "t_2288ad56"),
    ("2026-10-07 04:36:45", "t_a0967dc9"),
    ("2026-10-07 05:37:08", "t_69b5b8bc"),
    ("2026-10-07 07:14:45", "t_9428f40c"),
    ("2026-10-07 08:34:12", "t_1245cb11"),
    ("2026-10-07 18:46:58", "t_8b0bb66d"),
    ("2026-10-07 20:12:49", "t_65d2e7ac"),
    ("2026-10-07 22:24:46", "t_3cfce2dc"),
    ("2026-10-07 23:24:30", "t_11ac8569"),
    ("2026-10-08 00:14:35", "t_54408c2a"),
    ("2026-10-08 00:14:35", "t_5694f655"),
    ("2026-10-08 01:15:15", "t_ac42b55f"),
    ("2026-10-08 02:16:37", "t_b33b1db0"),
    ("2026-10-08 05:02:43", "t_ff7249b0"),
    ("2026-10-08 07:54:50", "t_a5971262"),
    ("2026-10-08 09:34:40", "t_f2ccbaa6"),
    ("2026-10-08 11:52:54", "t_7f073842"),  # the pinned critic
]


def _utc(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def _replay(clock) -> list[str]:
    """Route at each recorded event; a route reaches xai only when the pool hands out the
    credential, and every such probe fails billing again (the account stays empty)."""
    pool = _pool()
    first, *routes = _EVENTS
    clock.now = _utc(first[0])
    assert pool.select() is not None
    _bill(pool)
    probes = []
    for stamp, _card in routes:
        clock.now = _utc(stamp)
        if pool.select() is None:
            continue
        probes.append(stamp)
        _bill(pool)
    return probes


def test_billing_backoff_replays_xai_incident(home, clock, monkeypatch):
    assert len(_EVENTS) == 43
    assert len({card for _stamp, card in _EVENTS}) == 26
    assert len({stamp for stamp, _card in _EVENTS}) == 36

    new = _replay(clock)
    assert new == [
        "2026-10-06 22:16:13",  # t_19515a6a: 1 h bench ended 22:02, streak 2 -> 2 h
        "2026-10-07 00:17:07",  # 2 h bench ended 00:16, streak 3 -> 4 h
        "2026-10-07 04:36:45",  # 4 h bench ended 04:17, streak 4 -> 8 h
        "2026-10-07 18:46:58",  # 8 h bench ended 12:36, streak 5 -> 16 h
        "2026-10-08 11:52:54",  # 16 h bench ended 10:46: the daily-scale probe (t_7f073842)
    ]

    # Old logic = a flat 1 h billing bench: every event more than an hour after the previous
    # probe reaches xai again (t_11ac8569 at 23:24:30 is the one hourly-scale route refused,
    # 59 min 44 s after t_3cfce2dc).
    _write_store(home)
    monkeypatch.setattr("agent.credential_pool.EXHAUSTED_TTL_BILLING_MAX_SECONDS", HOUR)
    old = _replay(clock)
    assert len(old) == 19
    assert "2026-10-07 23:24:30" not in old
    assert set(new) <= set(old)
