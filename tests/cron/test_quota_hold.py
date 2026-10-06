"""Provider quota windows park a cron job instead of re-firing into them (#89376).

Contract (cron/quota_hold.py): a failed run whose cause is a rate-limited ``AuthError`` with a
``retry after <N>s`` hint parks a recurring job's ``next_run_at`` past the window and stamps
``quota_hold_until``; the stale-error re-arm leaves a held job alone; a run that reaches the
model clears the marker. The hint is read only from the AuthError in the cause chain, never from
arbitrary failure text.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import cron.incidents as incidents
import cron.jobs as cron_jobs
import cron.scheduler as sched
from cron import quota_hold as qh
from cron.jobs import (
    _job_is_stale_error_recurring, compute_next_run, create_job, get_due_jobs, get_job,
    mark_job_run, pause_job, update_job,
)
from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, AuthError
from tests.cron.test_cron_incidents import _tick_failing

QUOTA_MSG = "Codex provider quota exhausted (429); retry after 123518s. Credentials are still valid."


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _quota_error() -> AuthError:
    return AuthError(QUOTA_MSG, provider="openai-codex", code=CODEX_RATE_LIMITED_CODE)


def test_hold_seconds_only_from_rate_limited_auth_error_in_cause_chain():
    """The scheduler wraps the resolve failure in a RuntimeError ``from`` the AuthError; the
    hint survives through the cause chain, and text alone (or a re-login AuthError) never
    parks a job."""
    try:
        raise RuntimeError(QUOTA_MSG) from _quota_error()
    except RuntimeError as wrapped:
        assert qh.hold_seconds_from_failure(wrapped) == 123518.0

    assert qh.hold_seconds_from_failure(RuntimeError(QUOTA_MSG)) is None
    assert qh.hold_seconds_from_failure(RuntimeError("HTTP 429: retry after 60s")) is None
    relogin = AuthError(QUOTA_MSG, provider="openai-codex", code="expired", relogin_required=True)
    assert qh.hold_seconds_from_failure(relogin) is None
    structured = AuthError("quota", code=CODEX_RATE_LIMITED_CODE, retry_after=900)
    assert qh.hold_seconds_from_failure(structured) == 900.0


def _raise_quota(**_kw):
    raise _quota_error()


def _tick(job, home, deliveries, resolve):
    """One real scheduler tick (preflight ON) with the provider resolver replaced by *resolve*."""
    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
         patch.object(sched, "_deliver_result",
                      side_effect=lambda jb, content, **kw: deliveries.append(content)), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.side_effect = RuntimeError("model said no")
        sched.run_one_job(dict(job))


def test_quota_hold_parks_past_window_survives_stale_rearm_and_clears_on_model_reach(tmp_cron_home):
    """A 30-minute job whose provider resolve raises the Codex quota AuthError ('retry after
    123518s', over the 24h cap) is parked for the cap by the real scheduler tick: preflight lets
    the rate-limited AuthError through (it is not a missing credential), the one delivered alert
    carries the hold notice, and the job does not fire again inside the capped hold (not even
    after the stale-error re-arm's cadence+grace). The marker clears once a run reaches the
    model."""
    job = create_job("portfolio triage", "every 30m", deliver="local")
    job_id = job["id"]
    now = datetime.now(timezone.utc)
    deliveries: list = []

    _tick(get_job(job_id), tmp_cron_home, deliveries, _raise_quota)
    j = get_job(job_id)
    assert j["last_status"] == "error"
    # 123518s is over MAX_HOLD_SECONDS, so this alert takes the capped wording; it must still say
    # the job is held (the alert on entering a hold).
    assert len(deliveries) == 1 and "This job is held" in deliveries[0], deliveries
    assert "provider credential missing" not in deliveries[0]
    parked = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(seconds=qh.MAX_HOLD_SECONDS) <= parked - now < timedelta(seconds=123518), \
        "next_run_at lands one capped hold out, not at the provider's full window"
    assert j[qh.STATE_KEY] == j["next_run_at"]
    assert "_quota_hold_seconds" not in j
    # The hold is stamped with the route the fire was dispatched on (the fixture writes no config,
    # so the provider is whatever the unconfigured resolve reports).
    assert j[qh.ROUTE_KEY] == qh.route_of(j) and j[qh.ROUTE_KEY].endswith("@")
    assert "_quota_hold_route" not in j

    # Two hours later the job looks like a wedged stale-error record (#62002) — the hold says
    # it is parked on purpose, so it is neither re-armed nor due.
    update_job(job_id, {"last_run_at": (now - timedelta(hours=2)).isoformat()})
    j = get_job(job_id)
    assert not _job_is_stale_error_recurring(j, j["schedule"], now)
    assert all(d["id"] != job_id for d in get_due_jobs())
    assert datetime.fromisoformat(get_job(job_id)["next_run_at"]) == parked

    # A run that reached the model (either outcome) clears the marker.
    assert mark_job_run(job_id, False, "RuntimeError: model said no")
    j = get_job(job_id)
    assert qh.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - now < timedelta(hours=1)

    # Editing the schedule recomputes next_run_at from the new cadence; the stale marker must
    # not linger on a record that is no longer parked where it says.
    assert mark_job_run(job_id, False, QUOTA_MSG, quota_hold_seconds=123518)
    assert qh.STATE_KEY in get_job(job_id)
    j = update_job(job_id, {"schedule": "every 15m"})
    assert qh.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - now < timedelta(hours=1)


MONTHLY = 2581776  # Codex monthly 429: "retry after 2581776s" (~29.9 days)
FROZEN = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
CAP = timedelta(seconds=qh.MAX_HOLD_SECONDS)
SLACK = timedelta(seconds=qh.HOLD_SLACK_SECONDS)


@pytest.fixture
def frozen_now(monkeypatch):
    """Pin the hold module's clock (plan_hold/hold_notice read ``qh._hermes_now``) so equality
    assertions do not race the wall clock."""
    monkeypatch.setattr(qh, "_hermes_now", lambda: FROZEN)
    return FROZEN


def _interval_job(minutes=15):
    return {"id": "j1", "name": "probe", "schedule": {"kind": "interval", "minutes": minutes},
            "next_run_at": (FROZEN + timedelta(minutes=minutes)).isoformat()}


def test_monthly_window_parks_one_day_and_notices_the_reprobe(tmp_cron_home, frozen_now):
    """The provider says closed for ~30 days; the job is parked for the cap, not the window, and
    the alert says it re-probes instead of quoting the provider's 717h."""
    job = _interval_job()
    assert qh.plan_hold(job, MONTHLY)
    parked = datetime.fromisoformat(job["next_run_at"])
    assert parked == frozen_now + CAP + SLACK
    assert job[qh.STATE_KEY] == job["next_run_at"]

    notice = qh.hold_notice(job, MONTHLY)
    assert "This job is held" in notice and "re-probes" in notice
    assert f"{(qh.MAX_HOLD_SECONDS + qh.HOLD_SLACK_SECONDS) / 3600:.1f}h" in notice
    assert f"{MONTHLY / 3600:.1f}h" not in notice
    assert "no further alerts" not in notice.lower()  # day 2 alerts again; do not promise silence

    # Stale-error re-arm (#62002) leaves the parked job alone up to the instant it becomes due,
    # and the marker stops shielding it exactly then.
    held = dict(job, last_status="error", last_run_at=(FROZEN - timedelta(hours=3)).isoformat())
    just_before = parked - timedelta(seconds=1)
    assert qh.hold_active(held, just_before)
    assert not _job_is_stale_error_recurring(held, held["schedule"], just_before)
    assert not qh.hold_active(held, parked + timedelta(seconds=1))


def test_reprobe_on_a_still_closed_window_parks_again_with_its_own_alert(tmp_cron_home, frozen_now):
    """Day 2: the capped hold ends, the job re-probes, the window is still closed. It parks
    another cap (the notice does not promise silence; the delivered day-2 alert is exercised
    through the tick in ``test_day_two_reprobe_delivers_its_alert_through_the_tick``)."""
    job = _interval_job()
    assert qh.plan_hold(job, MONTHLY)
    first = job["next_run_at"]

    day2 = FROZEN + CAP + SLACK
    qh_now = lambda: day2  # noqa: E731
    with patch.object(qh, "_hermes_now", qh_now):
        job["next_run_at"] = (day2 + timedelta(minutes=15)).isoformat()  # _advance_after_run
        assert qh.plan_hold(job, MONTHLY - qh.MAX_HOLD_SECONDS)
        assert datetime.fromisoformat(job["next_run_at"]) == day2 + CAP + SLACK
        assert job["next_run_at"] != first and job[qh.STATE_KEY] == job["next_run_at"]
        assert "re-probes" in qh.hold_notice(job, MONTHLY - qh.MAX_HOLD_SECONDS)


def test_cron_schedule_parks_at_first_legal_occurrence_after_the_cap(tmp_cron_home, frozen_now):
    """For a cron expression the park is the first LEGAL occurrence after the capped window: up
    to one cadence past the cap, never before it."""
    job = {"id": "j2", "name": "daily", "schedule": {"kind": "cron", "expr": "0 9 * * *"},
           "next_run_at": (FROZEN + timedelta(hours=1)).isoformat()}
    window_end = frozen_now + CAP + SLACK
    assert qh.plan_hold(job, MONTHLY)
    parked = datetime.fromisoformat(job["next_run_at"])
    assert parked == datetime.fromisoformat(compute_next_run(job["schedule"], window_end.isoformat()))
    assert window_end < parked <= window_end + timedelta(days=1)
    assert "re-probes" in qh.hold_notice(job, MONTHLY)


def test_cron_with_no_computable_next_occurrence_is_not_parked(tmp_cron_home, frozen_now):
    """``compute_next_run`` returns None (croniter missing, no ``expr``): there is no legal
    occurrence to park at, and parking at the window boundary would fire at a time the
    expression excludes. The job keeps its natural next run and no alert promises a hold."""
    natural = (FROZEN + timedelta(hours=1)).isoformat()
    job = {"id": "j4", "name": "daily", "schedule": {"kind": "cron", "expr": "0 9 * * *"},
           "next_run_at": natural, qh.STATE_KEY: "stale"}
    with patch("cron.jobs.compute_next_run", return_value=None):
        assert not qh.plan_hold(job, MONTHLY)
        assert qh.hold_notice(job, MONTHLY) == ""
    assert job["next_run_at"] == natural
    assert qh.STATE_KEY not in job


def test_hold_log_reports_the_real_park_delay(tmp_cron_home, frozen_now, caplog):
    """The logged hold length is the actual delay to ``next_run_at`` (cap + slack + the cron
    cadence to the next legal occurrence), not the bare cap, so it agrees with the ``until``
    printed beside it."""
    job = {"id": "j5", "name": "daily", "schedule": {"kind": "cron", "expr": "0 9 * * *"},
           "next_run_at": (FROZEN + timedelta(hours=1)).isoformat()}
    with caplog.at_level("WARNING", logger="cron.scheduler"):
        assert qh.plan_hold(job, MONTHLY)
    parked = datetime.fromisoformat(job["next_run_at"])
    delay = (parked - frozen_now).total_seconds()
    assert delay > qh.MAX_HOLD_SECONDS + qh.HOLD_SLACK_SECONDS, "cron park is past cap + slack"
    record = next(r for r in caplog.records if "usage window closed" in r.getMessage())
    assert f"(holding for {delay:.0f}s)" in record.getMessage()
    assert f"closed for {MONTHLY}s" in record.getMessage()


def test_job_the_cap_does_not_park_gets_no_hold_notice(tmp_cron_home, frozen_now):
    """A sparse cron job whose natural next run is past the window is not parked, so the alert
    must not claim a hold."""
    job = {"id": "j3", "name": "yearly", "schedule": {"kind": "cron", "expr": "0 9 1 1 *"},
           "next_run_at": "2099-01-01T09:00:00+00:00", qh.STATE_KEY: "stale"}
    assert not qh.plan_hold(job, MONTHLY)
    assert qh.STATE_KEY not in job
    assert qh.hold_notice(job, MONTHLY) == ""


def test_window_inside_the_cap_keeps_the_original_notice(tmp_cron_home, frozen_now):
    job = _interval_job()
    assert qh.plan_hold(job, 3600)
    assert datetime.fromisoformat(job["next_run_at"]) == frozen_now + timedelta(seconds=3600) + SLACK
    notice = qh.hold_notice(job, 3600)
    assert "closed for about 1.0h" in notice and "no further alerts until then" in notice
    assert "re-probes" not in notice


def _held_job(provider="openai-codex", model="gpt-6.1-sol", **create_kw):
    """A 15-minute job (pinned to Codex unless *provider*/*model* say otherwise; ``None`` = unpinned,
    follows the main model) parked by a monthly 429 (real clock, loose bounds)."""
    job = create_job("probe", "every 15m", deliver="local",
                     provider=provider, model=model, **create_kw)
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY)
    held = get_job(job["id"])
    assert qh.STATE_KEY in held
    return held


def _main_runs_on(monkeypatch, provider, model):
    """What ``_main_model_pin`` (the main agent's provider+model) answers for this test."""
    monkeypatch.setattr("cron.jobs._main_model_pin", lambda: (provider, model))


def _minutes_out(job_id):
    return (datetime.fromisoformat(get_job(job_id)["next_run_at"]) - datetime.now(timezone.utc)
            ).total_seconds() / 60


@pytest.mark.parametrize("edit", [
    {"model": "claude-sonnet-5-5", "provider": "anthropic"},
    {"model": "gpt-6.2-sol"},
    {"provider": "anthropic"},
    {"base_url": "https://llm.example.invalid/v1"},
], ids=["model+provider", "model-only", "provider-only", "base_url"])
def test_repointing_a_held_job_clears_the_hold_and_reanchors(tmp_cron_home, edit):
    held = _held_job()
    assert _minutes_out(held["id"]) > 23 * 60
    updated = update_job(held["id"], dict(edit))
    assert qh.STATE_KEY not in updated
    assert 0 < _minutes_out(held["id"]) < 16, "next_run_at follows the 15m schedule again"


def test_unpinning_a_job_pinned_off_the_main_route_clears_the_hold_and_reanchors(
        tmp_cron_home, monkeypatch):
    """Unpin means "follow the main model": a repoint only when the main model is somewhere else."""
    held = _held_job()
    _main_runs_on(monkeypatch, "anthropic", "claude-sonnet-5-5")
    updated = update_job(held["id"], {"pinned": False})
    assert qh.STATE_KEY not in updated
    assert 0 < _minutes_out(held["id"]) < 16


def _write_cron_model_config(home, *, cron_model=True):
    """config.yaml whose main model IS the held job's pin (gpt-6.1-sol) while ``cron.model`` /
    ``cron.model_provider`` send unpinned jobs elsewhere (when *cron_model*)."""
    import yaml

    cfg = {"model": {"default": "gpt-6.1-sol"}}
    if cron_model:
        cfg["cron"] = {"model": "other-model", "model_provider": "other"}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _resolve_to_codex(**_kw):
    return {"provider": "openai-codex"}


def test_unpin_under_cron_model_default_clears_the_hold(tmp_cron_home, monkeypatch):
    """Pinned to the main model, but ``cron.model`` sends unpinned jobs elsewhere: unpin IS a
    repoint (the scheduler fires on cron.model), so the hold clears and the job re-anchors."""
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    _write_cron_model_config(tmp_cron_home)
    held = _held_job()
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve_to_codex):
        updated = update_job(held["id"], {"pinned": False})
    assert updated["provider"] is None and updated["model"] is None
    assert qh.STATE_KEY not in updated
    assert 0 < _minutes_out(held["id"]) < 16


def test_cron_default_route_matches_scheduler_model(tmp_cron_home, monkeypatch):
    """The route compare resolves an unpinned job to the model the scheduler will really run."""
    from cron.jobs import _cron_default_route

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve_to_codex):
        _write_cron_model_config(tmp_cron_home)
        provider, model = _cron_default_route()
        assert model == sched._load_cron_job_config({}, "j", "j").model == "other-model"
        assert provider == sched._load_cron_job_config({}, "j", "j").cron_default_provider

        _write_cron_model_config(tmp_cron_home, cron_model=False)
        assert _cron_default_route()[1] == sched._load_cron_job_config({}, "j", "j").model \
            == "gpt-6.1-sol"


def test_unpinning_a_job_pinned_to_the_main_route_keeps_the_hold(tmp_cron_home, monkeypatch):
    """Pinned to the model the main agent runs on, unpin changes the stored pin but not where the
    job runs: it must not re-fire into the closed window."""
    held = _held_job()
    _main_runs_on(monkeypatch, "openai-codex", "gpt-6.1-sol")
    updated = update_job(held["id"], {"pinned": False})
    assert updated["provider"] is None and updated["model"] is None
    assert updated[qh.STATE_KEY] == held[qh.STATE_KEY]
    assert get_job(held["id"])["next_run_at"] == held["next_run_at"]


@pytest.mark.parametrize("held_kw, main, edit", [
    ({}, ("openai-codex", "gpt-6.1-sol"), {"name": "renamed"}),
    ({}, ("openai-codex", "gpt-6.1-sol"), {"prompt": "something else"}),
    ({}, ("openai-codex", "gpt-6.1-sol"),
     {"model": "gpt-6.1-sol", "provider": "openai-codex"}),   # restates the current route
    ({}, ("openai-codex", "gpt-6.1-sol"), {"pinned": True}),  # already pinned: nothing changes
    ({"provider": None, "model": None}, ("openai-codex", "gpt-6.1-sol"),
     {"pinned": True}),                                       # unpinned -> pin at the same route
    ({"provider": None, "model": None}, ("openai-codex", "gpt-6.1-sol"),
     {"provider": "openai-codex"}),                           # partial edit, same effective route
], ids=["rename", "prompt", "restated-route", "pin-already-pinned", "pin-unpinned-at-main",
        "provider-restated-on-unpinned"])
def test_edits_that_do_not_change_the_route_keep_the_hold(tmp_cron_home, monkeypatch,
                                                          held_kw, main, edit):
    held = _held_job(**held_kw)
    _main_runs_on(monkeypatch, *main)
    updated = update_job(held["id"], dict(edit))
    assert updated[qh.STATE_KEY] == held[qh.STATE_KEY]
    assert get_job(held["id"])["next_run_at"] == held["next_run_at"]


def test_repointing_an_unpinned_held_job_off_the_main_route_clears_the_hold(
        tmp_cron_home, monkeypatch):
    held = _held_job(provider=None, model=None)
    _main_runs_on(monkeypatch, "openai-codex", "gpt-6.1-sol")
    updated = update_job(held["id"], {"model": "claude-sonnet-5-5", "provider": "anthropic"})
    assert qh.STATE_KEY not in updated
    assert 0 < _minutes_out(held["id"]) < 16


def test_route_compare_falls_back_to_the_stored_pin_when_the_main_model_is_unreadable(
        tmp_cron_home, monkeypatch):
    """The main model cannot be read (config/OAuth hiccup): the stored comparison stands, which is
    the pre-existing behaviour (a pin change is a repoint) and never fails the edit."""
    held = _held_job()

    def broken():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("cron.jobs._main_model_pin", broken)
    updated = update_job(held["id"], {"pinned": False})
    assert qh.STATE_KEY not in updated
    assert 0 < _minutes_out(held["id"]) < 16


def test_a_fully_explicit_route_edit_never_reads_the_main_model(tmp_cron_home, monkeypatch):
    """Explicit provider+model on both sides decide on the stored keys; no config work runs under
    the jobs lock."""
    held = _held_job()

    def forbidden():
        raise AssertionError("main model read for a fully explicit edit")

    monkeypatch.setattr("cron.jobs._main_model_pin", forbidden)
    updated = update_job(held["id"], {"model": "claude-sonnet-5-5", "provider": "anthropic"})
    assert qh.STATE_KEY not in updated


def test_explicit_next_run_at_wins_over_the_repoint_reanchor(tmp_cron_home):
    held = _held_job()
    requested = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    updated = update_job(held["id"], {"provider": "anthropic", "next_run_at": requested})
    assert qh.STATE_KEY not in updated
    assert datetime.fromisoformat(updated["next_run_at"]) == datetime.fromisoformat(requested)
    assert datetime.fromisoformat(get_job(held["id"])["next_run_at"]) == \
        datetime.fromisoformat(requested)


def test_repointing_a_paused_held_job_clears_the_marker_but_does_not_reanchor(tmp_cron_home):
    held = _held_job()
    pause_job(held["id"])
    updated = update_job(held["id"], {"model": "claude-sonnet-5-5", "provider": "anthropic"})
    assert qh.STATE_KEY not in updated
    assert updated["state"] == "paused"
    assert updated["next_run_at"] == held["next_run_at"]


def test_repointing_an_unheld_job_keeps_its_next_run(tmp_cron_home):
    job = create_job("probe", "every 15m", deliver="local",
                     provider="openai-codex", model="gpt-6.1-sol")
    before = get_job(job["id"])["next_run_at"]
    update_job(job["id"], {"model": "claude-sonnet-5-5", "provider": "anthropic"})
    assert get_job(job["id"])["next_run_at"] == before


def test_reanchoring_a_repointed_job_drops_a_stale_pending_slot(tmp_cron_home):
    held = _held_job()
    update_job(held["id"], {"pending_slot": {"instant": held["next_run_at"]}})
    assert get_job(held["id"]).get("pending_slot")
    update_job(held["id"], {"provider": "anthropic"})
    assert "pending_slot" not in get_job(held["id"])


def test_no_hold_notice_when_the_last_repeat_retires_the_job(tmp_cron_home):
    """``mark_job_run`` retires a repeat-exhausted job before ``plan_hold`` runs and leaves no
    hold, so the alert (composed first) must not say the job is held."""
    job = create_job("probe", "every 15m", deliver="local", repeat=1)
    assert qh.hold_notice(job, MONTHLY) == ""
    assert qh.hold_notice(job, 3600) == ""

    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY)
    after = get_job(job["id"])
    assert after["state"] == "completed" and qh.STATE_KEY not in after


def test_hold_notice_still_promises_the_hold_while_repeats_remain(tmp_cron_home):
    job = create_job("probe", "every 15m", deliver="local", repeat=2)
    assert "This job is held" in qh.hold_notice(job, MONTHLY)
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY)
    assert qh.STATE_KEY in get_job(job["id"])


def test_no_hold_notice_when_next_run_cannot_be_computed(tmp_cron_home):
    """A recurring job whose next run cannot be computed (croniter missing) ends in
    ``state=error`` with no park; the alert must not promise one."""
    job = create_job("probe", "every 15m", deliver="local")
    with patch("cron.jobs.compute_next_run", return_value=None):
        assert qh.hold_notice(job, MONTHLY) == ""
        assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY)
    after = get_job(job["id"])
    assert after["state"] == "error" and qh.STATE_KEY not in after


def test_terminal_after_run_leaves_the_stored_record_untouched(tmp_cron_home):
    job = create_job("probe", "every 15m", deliver="local", repeat=1)
    before = get_job(job["id"])
    assert cron_jobs.terminal_after_run(before)
    assert before == get_job(job["id"]) and before["repeat"]["completed"] == 0


def _quota_hint_error(seconds: int) -> AuthError:
    return AuthError(f"Codex provider quota exhausted (429); retry after {seconds}s. "
                     "Credentials are still valid.",
                     provider="openai-codex", code=CODEX_RATE_LIMITED_CODE)


@pytest.fixture
def day_clock(monkeypatch, tmp_path):
    """Every clock the failure path reads (hold plan, job store, incident ledger and its
    cooldown) follows one movable instant, and the incident ledger lives under *tmp_path*.
    Jobs under test deliver to a real lane (not ``local``): only a ping that leaves the process
    marks its incident ``alerted``, which is what arms the cooldown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The platform preflight would refuse the unconnected telegram lane before the agent runs.
    (tmp_path / "config.yaml").write_text("cron:\n  preflight: false\n", encoding="utf-8")
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    box = [datetime.now(timezone.utc)]
    for module in (qh, cron_jobs, sched, incidents):
        monkeypatch.setattr(module, "_hermes_now", lambda: box[0])
    return box


def test_day_two_reprobe_delivers_its_alert_through_the_tick(tmp_path, day_clock, monkeypatch):
    """The re-probe's alert is delivered by the real scheduler path, not just composed: the
    provider's remaining window shrinks, so the failure signature changes and a new incident
    alerts. The job is parked again each time. The repeat-alert cooldown is pinned above the
    24 h cap, so only the signature change can deliver the day-2 alert."""
    monkeypatch.setattr(sched, "_failure_repeat_alert_hours", lambda: 48.0)
    with cron_jobs.use_cron_store(tmp_path):
        job = create_job("probe", "every 15m", deliver="telegram:123")
        deliveries: list = []

        _tick_failing(get_job(job["id"]), tmp_path, deliveries, _quota_hint_error(MONTHLY))
        assert len(deliveries) == 1 and "This job is held" in deliveries[0], deliveries
        first = get_job(job["id"])
        assert first[qh.STATE_KEY] == first["next_run_at"]

        day_clock[0] += CAP + SLACK  # the capped hold ends; the job re-probes
        _tick_failing(get_job(job["id"]), tmp_path, deliveries,
                      _quota_hint_error(MONTHLY - 86400))
        assert len(deliveries) == 2 and "This job is held" in deliveries[1], deliveries
        second = get_job(job["id"])
        assert datetime.fromisoformat(second[qh.STATE_KEY]) == day_clock[0] + CAP + SLACK
        assert second[qh.STATE_KEY] != first[qh.STATE_KEY]


def test_day_two_reprobe_alert_is_withheld_inside_the_incident_cooldown(
        tmp_path, day_clock, monkeypatch):
    """A provider that repeats the same hint, with a cooldown longer than the hold, gets no day-2
    alert (normal failure-incident rules) but the job is still parked again."""
    monkeypatch.setattr(sched, "_failure_repeat_alert_hours", lambda: 48.0)
    with cron_jobs.use_cron_store(tmp_path):
        job = create_job("probe", "every 15m", deliver="telegram:123")
        deliveries: list = []

        _tick_failing(get_job(job["id"]), tmp_path, deliveries, _quota_hint_error(MONTHLY))
        assert len(deliveries) == 1

        day_clock[0] += CAP + SLACK
        _tick_failing(get_job(job["id"]), tmp_path, deliveries, _quota_hint_error(MONTHLY))
        assert len(deliveries) == 1, "same signature inside the cooldown: no second alert"
        second = get_job(job["id"])
        assert datetime.fromisoformat(second[qh.STATE_KEY]) == day_clock[0] + CAP + SLACK


# --- The hold is keyed to the route it was measured on (t_30440fb0) ---------------------------

import yaml  # noqa: E402

from cron.jobs import load_jobs, save_jobs  # noqa: E402

CODEX = {"default": "gpt-6.1-sol", "provider": "openai-codex"}
ANTHROPIC = {"default": "claude-opus-5-5", "provider": "anthropic"}


def _write_config(home, model, cron=None):
    cfg = {"model": dict(model)}
    if cron is not None:
        cfg["cron"] = dict(cron)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _held_unpinned(**create_kw):
    """A 15-minute job that follows the configured route, parked by a monthly 429 on it."""
    job = create_job("probe", "every 15m", deliver="local", **create_kw)
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY)
    held = get_job(job["id"])
    assert qh.hold_active(held) and held[qh.ROUTE_KEY] == qh.route_of(held)
    return held


def _assert_released(job_id):
    j = get_job(job_id)
    assert qh.STATE_KEY not in j and qh.ROUTE_KEY not in j
    assert not qh.hold_active(j)
    assert 0 < _minutes_out(job_id) < 16, "next_run_at is back within one cadence"


def _assert_kept(held):
    j = get_job(held["id"])
    assert j[qh.STATE_KEY] == held[qh.STATE_KEY] and qh.hold_active(j)
    assert j["next_run_at"] == held["next_run_at"]


def test_main_provider_switch_releases_the_hold_on_the_next_scan(tmp_cron_home):
    _write_config(tmp_cron_home, CODEX)
    held = _held_unpinned()
    assert held[qh.ROUTE_KEY] == "openai-codex@"
    _write_config(tmp_cron_home, ANTHROPIC)
    assert all(d["id"] != held["id"] for d in get_due_jobs())
    _assert_released(held["id"])


def test_same_provider_main_model_change_keeps_the_hold(tmp_cron_home):
    _write_config(tmp_cron_home, CODEX)
    held = _held_unpinned()
    _write_config(tmp_cron_home, {"default": "gpt-6.2-sol-mini", "provider": "openai-codex"})
    get_due_jobs()
    _assert_kept(held)


def test_unchanged_route_keeps_the_hold(tmp_cron_home):
    _write_config(tmp_cron_home, CODEX)
    held = _held_unpinned()
    get_due_jobs()
    get_due_jobs()
    _assert_kept(held)


def test_job_pinned_provider_ignores_main_switch(tmp_cron_home):
    _write_config(tmp_cron_home, ANTHROPIC)
    held = _held_unpinned(provider="openai-codex", model="gpt-6.1-sol")
    assert held[qh.ROUTE_KEY] == "openai-codex@"
    _write_config(tmp_cron_home, {"default": "glm-5", "provider": "zai"})
    get_due_jobs()
    _assert_kept(held)


def test_cron_model_provider_change_releases_the_hold(tmp_cron_home):
    _write_config(tmp_cron_home, ANTHROPIC, cron={"model_provider": "openai-codex"})
    held = _held_unpinned()
    assert held[qh.ROUTE_KEY] == "openai-codex@"
    _write_config(tmp_cron_home, ANTHROPIC, cron={"model_provider": "anthropic"})
    get_due_jobs()
    _assert_released(held["id"])


def test_stampless_legacy_hold_is_kept(tmp_cron_home):
    """A hold written before the route stamp existed is never released on its absence."""
    _write_config(tmp_cron_home, CODEX)
    held = _held_unpinned()
    jobs = load_jobs()
    for j in jobs:
        j.pop(qh.ROUTE_KEY, None)
    save_jobs(jobs)
    _write_config(tmp_cron_home, ANTHROPIC)
    get_due_jobs()
    _assert_kept(held)


def test_inflight_repoint_is_not_parked_on_the_old_window(tmp_cron_home):
    _write_config(tmp_cron_home, ANTHROPIC)
    job = create_job("race", "every 15m", deliver="local", provider="openai-codex", model="gpt-6.1-sol")
    dispatched_on = qh.route_of(get_job(job["id"]))
    update_job(job["id"], {"provider": "anthropic", "model": "claude-opus-5-5"})
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY,
                        quota_hold_route=dispatched_on)
    j = get_job(job["id"])
    assert not qh.hold_active(j) and qh.STATE_KEY not in j and qh.ROUTE_KEY not in j
    assert j["last_status"] == "error", "the run outcome is still recorded"
    assert 0 < _minutes_out(job["id"]) < 16


def _unreadable_route(*_a, **_kw):
    raise RuntimeError("config unreadable")


def test_route_read_failure_stamps_the_fire_route_and_keeps_the_outcome(
        tmp_cron_home, monkeypatch):
    """The mark-time read fails but the fire's route is known: the hold carries that route, so
    a later scan can still release it on a provider change (t_060ade5f#3)."""
    _write_config(tmp_cron_home, CODEX)
    job = create_job("probe", "every 15m", deliver="local")
    monkeypatch.setattr(qh, "route_of", _unreadable_route)
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY,
                        quota_hold_route="openai-codex@")
    j = get_job(job["id"])
    assert j["last_status"] == "error" and qh.hold_active(j)
    assert j[qh.ROUTE_KEY] == "openai-codex@"
    get_due_jobs()  # a scan that cannot read the route keeps the hold
    assert qh.hold_active(get_job(job["id"]))


def test_route_read_failure_without_a_fire_route_parks_unstamped(tmp_cron_home, monkeypatch):
    _write_config(tmp_cron_home, CODEX)
    job = create_job("probe", "every 15m", deliver="local")
    monkeypatch.setattr(qh, "route_of", _unreadable_route)
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=MONTHLY)
    j = get_job(job["id"])
    assert j["last_status"] == "error" and qh.hold_active(j) and qh.ROUTE_KEY not in j


def test_config_switch_while_the_fire_is_in_flight_does_not_park_it(tmp_cron_home):
    """The real tick snapshots the route BEFORE the provider resolve; a main-model switch that
    lands while the codex fire is failing must not park the job on codex's window."""
    _write_config(tmp_cron_home, CODEX)
    job = create_job("portfolio triage", "every 30m", deliver="local")

    def switch_then_raise(**_kw):
        _write_config(tmp_cron_home, ANTHROPIC)
        raise _quota_error()

    _tick(get_job(job["id"]), tmp_cron_home, [], switch_then_raise)
    j = get_job(job["id"])
    assert j["last_status"] == "error"
    assert not qh.hold_active(j) and qh.ROUTE_KEY not in j


def _stamp_daily(job_id, parked_at, route="openai-codex@"):
    """Rewrite a job's stored hold to park it until ``parked_at`` on ``route``."""
    jobs = load_jobs()
    rec = next(j for j in jobs if j["id"] == job_id)
    rec["next_run_at"] = rec[qh.STATE_KEY] = parked_at.isoformat()
    rec[qh.ROUTE_KEY] = route
    save_jobs(jobs)


def test_expired_marker_with_a_stale_stamp_is_still_due_and_not_moved(tmp_cron_home):
    """An expired marker is inert: the route release must not push a due job a cadence out."""
    _write_config(tmp_cron_home, ANTHROPIC)
    job = create_job("daily", "every 1d", deliver="local")
    from cron.jobs import _hermes_now
    _stamp_daily(job["id"], _hermes_now() - timedelta(minutes=5))
    parked = get_job(job["id"])["next_run_at"]
    assert job["id"] in [d["id"] for d in get_due_jobs()]
    assert get_job(job["id"])["next_run_at"] == parked


def test_releasing_a_near_end_hold_never_fires_later_than_the_hold(tmp_cron_home):
    _write_config(tmp_cron_home, ANTHROPIC)
    job = create_job("daily", "every 1d", deliver="local")
    from cron.jobs import _hermes_now, _parse_aware
    parked_at = _hermes_now() + timedelta(hours=1)
    _stamp_daily(job["id"], parked_at)
    get_due_jobs()
    j = get_job(job["id"])
    assert qh.STATE_KEY not in j and qh.ROUTE_KEY not in j, "the stale-route hold is released"
    assert _parse_aware(j["next_run_at"]) <= parked_at
