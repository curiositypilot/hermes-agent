"""Complexity labels + tier routing (hermes_cli/kanban_routing.py).

Covers: the ``complexity`` column / setter / create path, config parsing, the
read-only credential-pool availability check, tier walk + escalation, the
dispatcher integration (routed model reaches the worker, ``routed`` event on
the run, ``tier_exhausted`` hold, rate-limit cooldown bypass when the route
moves off the limited model), CLI verbs and the kanban_create tool.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_routing as kr


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    yield c
    c.close()


TIERS = {
    "S": [{"model": "cheap-1", "provider": "prov-a"}, {"model": "cheap-2", "provider": "prov-b"}],
    "M": [{"model": "mid-1", "provider": "prov-b"}],
    "L": [{"model": "big-1", "provider": "prov-c", "reasoning": "high"}],
}


_LOAD_ROUTING_CONFIG = kr.load_routing_config  # real parser, immune to _use_routing's patch


def _cfg(**over) -> kr.RoutingConfig:
    raw = {"enabled": True, "tiers": TIERS, **over}
    return _LOAD_ROUTING_CONFIG({"routing": raw})


def _use_routing(monkeypatch, **over):
    """Route the dispatcher through ``_cfg(**over)`` and an all-available pool."""
    cfg = _cfg(**over)
    monkeypatch.setattr(kr, "load_routing_config", lambda kanban_cfg=None: cfg)
    return cfg


def _pool(monkeypatch, blocked: dict[str, str] | None = None):
    """Stub the pool check: providers in ``blocked`` are unavailable with that reason."""
    blocked = blocked or {}
    until = time.time() + 600  # fixed per stub: a hold's retry_at must not drift between ticks

    def fake(provider, model, *, now=None):
        if provider in blocked:
            return kr.Availability(False, blocked[provider], until=until)
        return kr.Availability(True, "pool ok")

    monkeypatch.setattr(kr, "pool_availability", fake)


def _spawns(monkeypatch):
    seen: list = []

    def fake_spawn(task, workspace, board=None):
        seen.append((task.id, task.model_override, task.provider_override, task.reasoning_effort))
        return 4242

    return seen, fake_spawn


# ---------------------------------------------------------------------------
# Complexity field
# ---------------------------------------------------------------------------


def test_normalize_complexity():
    assert kb.normalize_complexity("s") == "S"
    assert kb.normalize_complexity(" Medium ") == "M"
    assert kb.normalize_complexity("LARGE") == "L"
    assert kb.normalize_complexity("xl") == "XL"
    assert kb.normalize_complexity("X-Large") == "XL"
    for empty in (None, "", "none", "-", "NULL"):
        assert kb.normalize_complexity(empty) is None
    with pytest.raises(ValueError):
        kb.normalize_complexity("XXL")


def test_create_and_set_complexity(conn):
    tid = kb.create_task(conn, title="t", assignee="w", complexity="m")
    assert kb.get_task(conn, tid).complexity == "M"
    created = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
    assert created.payload["complexity"] == "M"

    assert kb.set_complexity(conn, tid, "L", source="estimator")
    assert kb.get_task(conn, tid).complexity == "L"
    ev = [e for e in kb.list_events(conn, tid) if e.kind == "complexity_set"][-1]
    assert ev.payload == {"complexity": "L", "source": "estimator"}

    assert kb.set_complexity(conn, tid, "none")
    assert kb.get_task(conn, tid).complexity is None
    assert not kb.set_complexity(conn, "t_missing", "S")


def test_migration_adds_complexity_column(conn):
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "complexity" in cols


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_disabled():
    cfg = kr.load_routing_config({})
    assert cfg.enabled is False and cfg.tiers == {}
    assert cfg.unlabeled == "profile" and cfg.on_exhausted == "wait" and cfg.escalate is True


def test_config_parses_tiers_and_drops_bad_entries():
    cfg = kr.load_routing_config({"routing": {
        "enabled": True, "unlabeled": "m", "on_exhausted": "bogus", "cooldown_seconds": "60",
        "tiers": {
            "s": [{"model": "a", "provider": "p"}, {"provider": "no-model"}, "junk"],
            "XXL": [{"model": "ignored"}],
            "L": {"model": "single", "reasoning": "not-a-level"},
        },
    }})
    assert [c.model for c in cfg.tiers["S"]] == ["a"]
    assert "XXL" not in cfg.tiers and set(cfg.tiers) == {"S", "L"}
    assert cfg.tiers["L"][0].model == "single" and cfg.tiers["L"][0].reasoning is None
    assert cfg.unlabeled == "M" and cfg.on_exhausted == "wait" and cfg.cooldown_seconds == 60
    assert cfg.review == ()


def test_config_parses_review_candidates():
    cfg = kr.load_routing_config({"routing": {"enabled": True, "review": [
        {"model": "rev-1", "provider": "prov-r", "reasoning": "high"}, {"provider": "no-model"}]}})
    assert cfg.review == (kr.TierCandidate("rev-1", "prov-r", "high"),)
    single = kr.load_routing_config({"routing": {"review": {"model": "rev-1"}}})
    assert [c.model for c in single.review] == ["rev-1"]


# ---------------------------------------------------------------------------
# Pool availability (read-only)
# ---------------------------------------------------------------------------


def _write_pool(home: Path, provider: str, entries: list[dict]) -> None:
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {provider: entries}}))


@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    """A HERMES_HOME whose auth.json is NOT ``Path.home()/.hermes/auth.json`` —
    ``kanban_home`` patches ``Path.home`` so its auth.json trips the auth
    module's real-store seat belt."""
    home = tmp_path / "authhome"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_pool_availability_states(auth_home):
    kanban_home = auth_home
    now = time.time()
    assert kr.pool_availability("prov", "m").reason.startswith("no pool entries")

    _write_pool(kanban_home, "prov", [
        {"id": "a", "label": "a", "access_token": "k1", "last_status": "exhausted",
         "last_status_at": now - 10, "last_error_code": 429, "last_error_reset_at": now + 900},
    ])
    avail = kr.pool_availability("prov", "m", now=now)
    assert not avail.available and avail.until == pytest.approx(now + 900)

    _write_pool(kanban_home, "prov", [
        {"id": "a", "label": "a", "access_token": "k1", "last_status": "exhausted",
         "last_status_at": now - 10, "last_error_code": 429, "last_error_reset_at": now + 900},
        {"id": "b", "label": "b", "access_token": "k2", "last_status": "ok"},
    ])
    assert kr.pool_availability("prov", "m", now=now).available

    _write_pool(kanban_home, "prov", [
        {"id": "a", "label": "a", "access_token": "k1", "last_status": "exhausted",
         "last_status_at": now - 7200, "last_error_code": 429},
    ])
    assert kr.pool_availability("prov", "m", now=now).available  # TTL elapsed

    _write_pool(kanban_home, "prov", [
        {"id": "a", "label": "a", "access_token": "k1", "last_status": "dead"},
    ])
    assert not kr.pool_availability("prov", "m", now=now).available

    _write_pool(kanban_home, "prov", [
        {"id": "a", "label": "a", "access_token": "k1", "last_status": "ok",
         "model_cooldowns": {"m-benched": now + 600}},
    ])
    assert not kr.pool_availability("prov", "m-benched", now=now).available
    assert kr.pool_availability("prov", "other", now=now).available


def test_pool_availability_never_writes_auth_json(auth_home):
    kanban_home = auth_home
    now = time.time()
    _write_pool(kanban_home, "prov", [
        {"id": "a", "label": "a", "access_token": "k1", "last_status": "exhausted",
         "last_status_at": now - 7200, "last_error_code": 429},
    ])
    before = (kanban_home / "auth.json").read_text()
    kr.pool_availability("prov", "m", now=now)
    assert (kanban_home / "auth.json").read_text() == before


FABLE = "claude-fable-5-2"
FABLE_BENCH_UNTIL = 1794961199.98  # incident 2026-11-18: fable benched until 00:19:59 UTC
OAUTH_ROW = {"id": "6afabf", "label": "oauth", "auth_type": "oauth", "source": "manual:hermes_pkce",
             "access_token": "tok", "last_status": "ok",
             "model_cooldowns": {FABLE: FABLE_BENCH_UNTIL, "claude-opus-5-5": 1794961200.0}}
ENV_ROW = {"id": "2c315c", "label": "ANTHROPIC_API_KEY", "auth_type": "api_key",
           "source": "env:ANTHROPIC_API_KEY", "last_status": "ok"}


def _pin_bench_clock(monkeypatch, now: float) -> float:
    """``model_cooldown_until`` reads the wall clock, not ``pool_availability(now=)``."""
    from agent import credential_pool_model_cooldowns as mc
    monkeypatch.setattr(mc, "time", SimpleNamespace(time=lambda: now))
    return now


def test_pool_availability_ignores_env_row_with_unset_var(auth_home, monkeypatch):
    """t_846f38f4: a token-less ``env:`` row whose var is unset is not capacity."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _write_pool(auth_home, "anthropic", [OAUTH_ROW, ENV_ROW])
    for now in (1794960273, 1794960645):  # 00:04:33 and 00:10:45, when the cards crashed
        avail = kr.pool_availability("anthropic", FABLE, now=_pin_bench_clock(monkeypatch, now))
        assert avail.available is False
        assert avail.until == FABLE_BENCH_UNTIL
    assert kr.pool_availability("anthropic", FABLE, now=_pin_bench_clock(monkeypatch, 1794961208)).available is True

    _write_pool(auth_home, "anthropic", [ENV_ROW])
    alone = kr.pool_availability("anthropic", FABLE, now=1794960273)
    assert alone.available is False and alone.reason.startswith("no usable credentials")


def test_pool_availability_counts_env_row_when_var_set(auth_home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set-in-env")
    _write_pool(auth_home, "anthropic", [OAUTH_ROW, ENV_ROW])
    assert kr.pool_availability("anthropic", FABLE, now=_pin_bench_clock(monkeypatch, 1794960273)).available is True


def test_pool_availability_keeps_other_rows_without_env_rule(auth_home, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _pin_bench_clock(monkeypatch, 1794960273)
    manual_key = {"id": "m1", "label": "m", "auth_type": "api_key", "source": "manual",
                  "access_token": "k-on-disk", "last_status": "ok"}
    _write_pool(auth_home, "anthropic", [OAUTH_ROW, manual_key])
    assert kr.pool_availability("anthropic", FABLE, now=1794960273).available is True

    borrowed = {"id": "cc", "label": "claude code", "auth_type": "oauth", "source": "claude_code",
                "last_status": "ok"}
    _write_pool(auth_home, "anthropic", [OAUTH_ROW, borrowed])
    assert kr.pool_availability("anthropic", FABLE, now=1794960273).available is True


def test_routing_pool_check_binds_assignee_secret_scope_under_multiplex(conn, kanban_home, monkeypatch):
    """t_846f38f4: the multiplexed dispatch tick runs in a fresh Context with no secret scope, so
    the routing seam must bind the assignee's scope or the token-less ``env:`` row stays capacity."""
    import contextvars
    from agent.secret_scope import set_multiplex_active

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    profile = kanban_home / "profiles" / "elias"
    profile.mkdir(parents=True)
    _write_pool(profile, "anthropic", [OAUTH_ROW, ENV_ROW])
    _pin_bench_clock(monkeypatch, 1794960273)  # 00:04:33, fable benched

    def check():
        return kr.RoutingContext(conn, _cfg()).availability(kr.TierCandidate(FABLE, "anthropic"), str(profile))

    set_multiplex_active(True)
    try:
        unset = contextvars.Context().run(check)
        profile.joinpath(".env").write_text("ANTHROPIC_API_KEY=elias-key\n", encoding="utf-8")
        key_set = contextvars.Context().run(check)
    finally:
        set_multiplex_active(False)
    assert unset.available is False and unset.until == FABLE_BENCH_UNTIL
    assert key_set.available is True


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _task(**kw):
    defaults: dict[str, Any] = dict(id="t_x", title="x", body=None, assignee="w", status="ready", priority=0,
                                     created_by=None, created_at=0, started_at=None, completed_at=None,
                                     workspace_kind="scratch", workspace_path=None, claim_lock=None,
                                     claim_expires=None, tenant=None)
    defaults.update(kw)
    return kb.Task(**defaults)


def test_decide_precedence_and_walk(conn, monkeypatch):
    _pool(monkeypatch, {"prov-a": "credential pool exhausted"})
    ctx = kr.RoutingContext(conn, _cfg())

    pinned = ctx.decide(_task(model_override="x", provider_override="p", complexity="S"))
    assert pinned.source == "pinned" and not pinned.applies_model

    d = ctx.decide(_task(complexity="S"))
    assert d.source == "tier" and d.tier == "S" and d.candidate.model == "cheap-2"
    assert d.skipped[0]["model"] == "cheap-1"

    unl = ctx.decide(_task())
    assert unl.source == "profile" and unl.note == "unlabeled"


def test_pinned_unavailable_holds(conn, monkeypatch):
    _pool(monkeypatch, {"p": "credential pool exhausted"})
    for cfg in (_cfg(), _cfg(on_exhausted="profile")):
        d = kr.RoutingContext(conn, cfg).decide(_task(model_override="x", provider_override="p", complexity="S"))
        assert d.source == "exhausted" and d.retry_at and not d.applies_model
        assert d.skipped == [{"model": "x", "provider": "p", "reason": "credential pool exhausted"}]
        assert d.label() == "pinned p:x: unavailable (credential pool exhausted)"
    ok = kr.RoutingContext(conn, _cfg()).decide(_task(model_override="x", provider_override="q"))
    assert ok.source == "pinned" and ok.label() == "pinned -> q:x"


def test_dispatch_holds_pinned_on_unavailable_provider(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch)
    _pool(monkeypatch, {"prov-z": "credential pool exhausted"})
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="p", assignee="w", model_override="mine", provider_override="prov-z")

    for _ in range(2):
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert (tid, "tier_exhausted") in res.respawn_guarded

    assert seen == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.consecutive_failures == 0
    holds = [e for e in kb.list_events(conn, tid) if e.kind == "routing_held"]
    assert len(holds) == 1 and holds[0].payload["retry_at"]


def test_decide_escalates_then_waits_or_falls_back(conn, monkeypatch):
    _pool(monkeypatch, {"prov-a": "x", "prov-b": "x"})
    d = kr.RoutingContext(conn, _cfg()).decide(_task(complexity="S"))
    assert d.source == "tier" and d.tier == "L" and d.requested_tier == "S"
    assert "escalated from S" in d.label()

    no_esc = kr.RoutingContext(conn, _cfg(escalate=False)).decide(_task(complexity="S"))
    assert no_esc.source == "exhausted" and no_esc.retry_at

    _pool(monkeypatch, {"prov-a": "x", "prov-b": "x", "prov-c": "x"})
    assert kr.RoutingContext(conn, _cfg()).decide(_task(complexity="S")).source == "exhausted"
    fb = kr.RoutingContext(conn, _cfg(on_exhausted="profile")).decide(_task(complexity="S"))
    assert fb.source == "profile" and fb.note == "all tier candidates unavailable"


XL_TIER = [{"model": "deep-1", "provider": "prov-d", "reasoning": "high"},
           {"model": "big-1", "provider": "prov-c", "reasoning": "high"}]


def test_xl_label_routes_to_xl_tier_first_entry(conn, monkeypatch):
    _pool(monkeypatch)
    d = kr.RoutingContext(conn, _cfg(tiers={**TIERS, "XL": XL_TIER})).decide(_task(complexity="XL"))
    assert d.source == "tier" and d.tier == "XL" and d.requested_tier == "XL"
    assert (d.candidate.model, d.candidate.provider, d.candidate.reasoning) == ("deep-1", "prov-d", "high")


def test_l_escalates_to_xl_only_when_l_is_exhausted(conn, monkeypatch):
    cfg = _cfg(tiers={**TIERS, "XL": XL_TIER})
    _pool(monkeypatch)
    assert kr.RoutingContext(conn, cfg).decide(_task(complexity="L")).tier == "L"
    _pool(monkeypatch, {"prov-c": "x"})
    d = kr.RoutingContext(conn, cfg).decide(_task(complexity="L"))
    assert d.source == "tier" and d.tier == "XL" and d.requested_tier == "L"
    assert d.candidate.model == "deep-1" and "escalated from L" in d.label()
    # XL is the top tier: nothing above it to escalate into.
    _pool(monkeypatch, {"prov-c": "x", "prov-d": "x"})
    assert kr.RoutingContext(conn, cfg).decide(_task(complexity="XL")).source == "exhausted"


def test_estimator_never_assigns_xl(monkeypatch):
    from types import SimpleNamespace

    from agent import auxiliary_client

    def fake_llm(**_kw):
        msg = SimpleNamespace(content='{"est_tokens": 9, "complexity": "XL", "rationale": "deep"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="aux")

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_llm)
    out = kr.estimate_complexity("research the thing", None, task_id=None)
    assert out["ok"] is True and out["complexity"] is None


def test_decide_unlabeled_tier_and_empty_tier(conn, monkeypatch):
    _pool(monkeypatch)
    d = kr.RoutingContext(conn, _cfg(unlabeled="M")).decide(_task())
    assert d.source == "tier" and d.candidate.model == "mid-1"
    empty = kr.RoutingContext(conn, _cfg(tiers={"S": []}, escalate=False)).decide(_task(complexity="S"))
    assert empty.source == "profile" and "no candidates" in empty.note


def test_apply_route_keeps_task_reasoning():
    cand = kr.TierCandidate("big-1", "prov-c", "high")
    d = kr.RouteDecision("tier", requested_tier="L", tier="L", candidate=cand)
    t = _task(reasoning_effort="low")
    kr.apply_route(t, d)
    assert (t.model_override, t.provider_override, t.reasoning_effort) == ("big-1", "prov-c", "low")
    t2 = _task()
    kr.apply_route(t2, d)
    assert t2.reasoning_effort == "high"


def test_apply_route_marks_pinned():
    """Only a pinned route (or, with no route, a card pin) marks ``route_pinned``; the
    mark lands even though apply_route's applies-model guard returns early for pins."""
    pin = kr.RouteDecision("pinned", candidate=kr.TierCandidate("mine", "prov-z"))
    tier = kr.RouteDecision("tier", requested_tier="S", tier="S", candidate=kr.TierCandidate("c", "prov-a"))
    review = kr.RouteDecision("review", lane="review", candidate=kr.TierCandidate("r", "prov-r"))
    profile = kr.RouteDecision("profile", candidate=kr.TierCandidate("p", "prov-p"))

    t = _task(model_override="mine", provider_override="prov-z")
    kr.apply_route(t, pin)
    assert t.route_pinned is True and (t.model_override, t.provider_override) == ("mine", "prov-z")
    for decision in (tier, review, profile):
        t = _task()
        kr.apply_route(t, decision)
        assert t.route_pinned is False
    # A review run of a pinned card runs on the reviewer model, not the pin.
    t = _task(model_override="mine", provider_override="prov-z")
    kr.apply_route(t, review)
    assert t.route_pinned is False and t.model_override == "r"
    # No decision (routing off / no review candidates): the card pin drives -m, so it counts.
    t = _task(model_override="mine", provider_override="prov-z")
    kr.apply_route(t, None)
    assert t.route_pinned is True
    t = _task()
    kr.apply_route(t, None)
    assert t.route_pinned is False


def test_dispatch_marks_pinned_card_with_routing_off(conn, monkeypatch, all_assignees_spawnable):
    """Routing disabled: apply_route still runs, so a card pin still reaches the spawn as pinned."""
    _use_routing(monkeypatch, enabled=False)
    seen = []

    def spawn(task, workspace, board=None):
        seen.append((task.id, getattr(task, "route_pinned", None)))
        return 4242

    pinned = kb.create_task(conn, title="p", assignee="w", model_override="mine", provider_override="prov-z")
    plain = kb.create_task(conn, title="u", assignee="w")
    kbd.dispatch_once(conn, spawn_fn=spawn)
    assert dict(seen) == {pinned: True, plain: False}


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_dispatch_routes_and_records_event(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch)
    _pool(monkeypatch, {"prov-a": "credential pool exhausted"})
    seen, spawn = _spawns(monkeypatch)
    small = kb.create_task(conn, title="s", assignee="w", complexity="S")
    large = kb.create_task(conn, title="l", assignee="w", complexity="L")
    pinned = kb.create_task(conn, title="p", assignee="w", complexity="S",
                            model_override="mine", provider_override="prov-z")
    plain = kb.create_task(conn, title="u", assignee="w")

    res = kbd.dispatch_once(conn, spawn_fn=spawn)

    by_id = {s[0]: s[1:] for s in seen}
    assert by_id[small] == ("cheap-2", "prov-b", None)
    assert by_id[large] == ("big-1", "prov-c", "high")
    assert by_id[pinned] == ("mine", "prov-z", None)
    assert by_id[plain] == (None, None, None)
    assert {tid for tid, _ in res.routed} == {small, large, pinned, plain}

    # The card itself is never mutated — every retry routes afresh.
    assert kb.get_task(conn, small).model_override is None
    ev = next(e for e in kb.list_events(conn, small) if e.kind == "routed")
    assert ev.run_id == kb.get_task(conn, small).current_run_id
    assert ev.payload["model"] == "cheap-2" and ev.payload["tier"] == "S"
    assert ev.payload["skipped"][0]["model"] == "cheap-1"


def test_dispatch_routing_disabled_is_inert(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch, enabled=False)
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")
    res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert seen == [(tid, None, None, None)] and res.routed == []
    assert not [e for e in kb.list_events(conn, tid) if e.kind == "routed"]


def test_dispatch_holds_when_tier_exhausted(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch)
    _pool(monkeypatch, {"prov-a": "x", "prov-b": "x", "prov-c": "x"})
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")

    for _ in range(3):
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert (tid, "tier_exhausted") in res.respawn_guarded

    assert seen == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.consecutive_failures == 0
    holds = [e for e in kb.list_events(conn, tid) if e.kind == "routing_held"]
    assert len(holds) == 1  # recorded once per hold streak, not every tick
    assert holds[0].payload["requested_tier"] == "S" and holds[0].payload["retry_at"]

    dry = kbd.dispatch_once(conn, spawn_fn=spawn, dry_run=True)
    assert (tid, "tier_exhausted") in dry.respawn_guarded


def _routing_holds(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "routing_held"]


def test_routing_hold_dedupe_survives_comment(conn, monkeypatch, all_assignees_spawnable):
    """A comment between ticks must not restart the hold streak."""
    _use_routing(monkeypatch)
    _pool(monkeypatch, {"prov-a": "x", "prov-b": "x", "prov-c": "x"})
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")

    kbd.dispatch_once(conn, spawn_fn=spawn)
    kb.add_comment(conn, tid, "w", "still waiting")
    kbd.dispatch_once(conn, spawn_fn=spawn)

    assert seen == []
    holds = _routing_holds(conn, tid)
    assert len(holds) == 1
    assert holds[0].payload["reason"] == "tier_exhausted"


def test_routing_hold_reemits_on_lifecycle_or_new_note(conn, monkeypatch, all_assignees_spawnable, tmp_path):
    """A lifecycle event, or a changed reason/note, ends the streak."""
    _use_routing(monkeypatch)
    _pool(monkeypatch, {"prov-a": "x", "prov-b": "x", "prov-c": "x"})
    _, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")

    kbd.dispatch_once(conn, spawn_fn=spawn)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "claimed", {"lock": "test"})
    kbd.dispatch_once(conn, spawn_fn=spawn)
    holds = _routing_holds(conn, tid)
    assert len(holds) == 2
    assert [h.payload["reason"] for h in holds] == ["tier_exhausted", "tier_exhausted"]

    path = tmp_path / "headroom.json"
    path.write_text(json.dumps({
        "schema": 1,
        "written_at_epoch": time.time() - 60,
        "providers": {
            name: {"ok": True, "binding": {"name": "session", "headroom": 0.01,
                                            "resets_at": "2026-11-04T00:00:00Z"}, "windows": []}
            for name in ("prov-a", "prov-b", "prov-c")
        },
    }))
    _use_routing(monkeypatch, quota_gate={
        "enabled": True,
        "headroom_file": str(path),
    })
    kbd.dispatch_once(conn, spawn_fn=spawn)
    holds = _routing_holds(conn, tid)
    assert len(holds) == 3
    assert holds[-1].payload["reason"] == "quota_hold"
    assert holds[-1].payload.get("note")


def test_routing_hold_reemits_on_new_retry_at(conn):
    """A later retry_at is a new row (stranded_in_ready muting reads the newest); the same one is not."""
    from hermes_cli import kanban_db_dispatch_holds as holds_mod

    tid = kb.create_task(conn, title="m", assignee="w", complexity="M")
    for retry_at in (1795267081, 1795267081, 1795269085, 1795269085):
        route = kr.RouteDecision("exhausted", requested_tier="M", retry_at=retry_at)
        holds_mod._record_routing_hold(conn, tid, route)
    assert [(h.payload or {}).get("retry_at") for h in _routing_holds(conn, tid)] == [1795267081, 1795269085]


def test_routing_hold_swallows_only_sqlite_errors(conn, monkeypatch):
    """A DB error writing the hold is logged and skipped; any other error is a bug and raises."""
    import sqlite3

    from hermes_cli import kanban_db_dispatch_holds as holds_mod

    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")

    def db_error(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb, "_append_event", db_error)
    holds_mod._record_routing_hold(conn, tid, None)  # swallowed

    def bug(*_a, **_k):
        raise RuntimeError("bug")

    monkeypatch.setattr(kb, "_append_event", bug)
    with pytest.raises(RuntimeError, match="bug"):
        holds_mod._record_routing_hold(conn, tid, None)
    assert _routing_holds(conn, tid) == []


def _guard_reasons(conn, tid):
    return [(e.payload or {}).get("reason") for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"]


def test_respawn_guard_event_written_once_per_streak(conn, monkeypatch, all_assignees_spawnable):
    """A card held by a respawn guard gets one ``respawn_guarded`` row per streak, not
    one per tick; a comment between ticks does not restart the streak (t_6190ed34)."""
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="already PRed", assignee="w")
    kb.add_comment(conn, tid, "w", "Opened https://github.com/example/repo/pull/123")

    for i in range(3):
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert (tid, "active_pr") in res.respawn_guarded
        if i == 0:
            kb.add_comment(conn, tid, "w", "still waiting on review")

    assert seen == []
    assert _guard_reasons(conn, tid) == ["active_pr"]


def test_respawn_guard_event_reemits_on_new_reason_or_lifecycle(conn, monkeypatch, all_assignees_spawnable):
    """A changed reason, or a non-noise event between two same-reason ticks, opens a new streak."""
    reasons = iter(["rate_limit_cooldown", "blocker_auth", "blocker_auth", "blocker_auth"])
    monkeypatch.setattr(kbd, "check_respawn_guard", lambda conn, task_id, lane="ready": next(reasons))
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="held", assignee="w")

    for expected in ("rate_limit_cooldown", "blocker_auth", "blocker_auth"):
        assert (tid, expected) in kbd.dispatch_once(conn, spawn_fn=spawn).respawn_guarded
    assert _guard_reasons(conn, tid) == ["rate_limit_cooldown", "blocker_auth"]

    with kb.write_txn(conn):
        kb._append_event(conn, tid, "reclaimed", {"reason": "test"})
    assert (tid, "blocker_auth") in kbd.dispatch_once(conn, spawn_fn=spawn).respawn_guarded
    assert _guard_reasons(conn, tid) == ["rate_limit_cooldown", "blocker_auth", "blocker_auth"]
    assert seen == []


def _seed_rate_limited_run(conn, tid, *, model=None, provider=None, ended_at=None):
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    if model:
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "routed", {"source": "tier", "model": model, "provider": provider},
                             run_id=run_id)
    conn.execute("UPDATE task_runs SET outcome='rate_limited', status='rate_limited', ended_at=? WHERE id=?",
                 (ended_at or int(time.time()), run_id))
    conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, "
                 "worker_pid=NULL, last_failure_error='rate-limited' WHERE id=?", (tid,))
    conn.commit()


def test_rate_limited_model_is_skipped_board_wide(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch)
    _pool(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    seen, spawn = _spawns(monkeypatch)
    first = kb.create_task(conn, title="first", assignee="w", complexity="S")
    _seed_rate_limited_run(conn, first, model="cheap-1", provider="prov-a")
    other = kb.create_task(conn, title="other", assignee="w", complexity="S")

    kbd.dispatch_once(conn, spawn_fn=spawn)

    by_id = {s[0]: s[1] for s in seen}
    # The limited card routes off cheap-1 so the cooldown guard no longer idles it,
    # and a sibling S card also skips the model that just hit its wall.
    assert by_id == {first: "cheap-2", other: "cheap-2"}


def test_cooldown_still_applies_when_route_is_unchanged(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch, cooldown_seconds=0)  # board history ignored → same model re-picked
    _pool(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")
    _seed_rate_limited_run(conn, tid, model="cheap-1", provider="prov-a")
    res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert seen == [] and (tid, "rate_limit_cooldown") in res.respawn_guarded


def test_unrouted_rate_limit_is_escaped_by_routing(conn, monkeypatch, all_assignees_spawnable):
    """A card that hit a wall on the profile default gets a tier model at once."""
    _use_routing(monkeypatch)
    _pool(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="M")
    _seed_rate_limited_run(conn, tid)
    kbd.dispatch_once(conn, spawn_fn=spawn)
    assert seen == [(tid, "mid-1", "prov-b", None)]


# ---------------------------------------------------------------------------
# No escalation onto a provider already unavailable at a lower tier (t_9375becf)
# ---------------------------------------------------------------------------

_OPUS = {"model": "claude-opus-5-5", "provider": "anthropic"}
_FABLE = {"model": "claude-fable-5-2", "provider": "anthropic"}
_LIVE_TIERS = {"M": [_OPUS], "L": [_OPUS], "XL": [_FABLE, _OPUS]}


def _seed_hit(conn, model, provider, ended_at):
    tid = kb.create_task(conn, title=f"hit {model} {ended_at}", assignee="w", complexity="M")
    _seed_rate_limited_run(conn, tid, model=model, provider=provider, ended_at=ended_at)


def test_no_escalation_onto_rate_limited_provider(conn, monkeypatch):
    _pool(monkeypatch)
    now = time.time()
    _seed_hit(conn, "claude-opus-5-5", "anthropic", int(now) - 60)
    ctx = kr.RoutingContext(conn, _cfg(tiers=_LIVE_TIERS, cooldown_seconds=900), now=now)

    m = ctx.decide(_task(complexity="M"))
    assert m.source == "exhausted" and m.retry_at == int(now) - 60 + 900
    assert {"tier": "XL", "model": "claude-fable-5-2", "provider": "anthropic",
            "reason": "no escalation: anthropic unavailable at tier M"} in m.skipped
    xl = ctx.decide(_task(complexity="XL"))
    assert xl.source == "tier" and xl.tier == "XL" and xl.candidate.model == "claude-fable-5-2"

    # Escalation onto another provider is kept.
    other = {**_LIVE_TIERS, "XL": [{"model": "x-1", "provider": "prov-x"}]}
    d = kr.RoutingContext(conn, _cfg(tiers=other, cooldown_seconds=900), now=now).decide(_task(complexity="M"))
    assert d.source == "tier" and d.tier == "XL" and d.candidate.model == "x-1"


# Incident 2026-11-21 (UTC): Fable's board hit expired 10 s before Opus's, so M cards walked
# M -> L -> XL onto claude-fable-5-2 on the same exhausted Anthropic account.
# (tick, fable hit ended_at, opus hit ended_at, expected retry_at); the board holds only the
# hits up to each tick, as in the replay.
_INCIDENT_9375_TICKS = (
    (1795267031, 1795266121, 1795266181, 1795267081),  # 13:17:11; hits 13:02:01, 13:03:01
    (1795269034, 1795268124, 1795268185, 1795269085),  # 13:50:34; hits 13:35:24, 13:36:25
)


def test_incident_20261121_m_cards_not_escalated_to_fable(conn, monkeypatch):
    _pool(monkeypatch)
    cfg = _cfg(tiers=_LIVE_TIERS, cooldown_seconds=900)
    for now, fable_at, opus_at, retry_at in _INCIDENT_9375_TICKS:
        _seed_hit(conn, "claude-fable-5-2", "anthropic", fable_at)
        _seed_hit(conn, "claude-opus-5-5", "anthropic", opus_at)
        m = kr.RoutingContext(conn, cfg, now=now).decide(_task(complexity="M"))
        assert m.source == "exhausted" and m.retry_at == retry_at, m.label()
        xl = kr.RoutingContext(conn, cfg, now=now).decide(_task(complexity="XL"))
        assert xl.source == "tier" and xl.candidate.model == "claude-fable-5-2"


def test_board_hit_until_takes_later_pool_bench(conn, monkeypatch):
    """A board hit's retry_at is the later of ended_at + cooldown and the pool's own bench (t_d2cb1f25)."""
    now = time.time()
    hit = int(now) - 60
    _seed_hit(conn, "claude-opus-5-5", "anthropic", hit)
    cfg = _cfg(tiers=_LIVE_TIERS, cooldown_seconds=900)
    cases = (
        (kr.Availability(False, "credential pool exhausted", until=hit + 2000), hit + 2000),
        (kr.Availability(True, "pool ok"), hit + 900),
        (kr.Availability(False, "credential pool exhausted", until=hit + 100), hit + 900),
        (kr.Availability(False, "credential pool exhausted", until=None), hit + 900),
    )
    for pool, retry_at in cases:
        monkeypatch.setattr(kr, "pool_availability", lambda provider, model, *, now=None, _p=pool: _p)
        m = kr.RoutingContext(conn, cfg, now=now).decide(_task(complexity="M"))
        assert m.source == "exhausted" and m.retry_at == retry_at, (pool, m.label())
        assert m.skipped[0]["reason"] == "rate_limited on this board"


def test_mode_gate_on_m_also_holds_xl_same_provider(conn, monkeypatch, tmp_path):
    """A provider-wide quota verdict gating M opus also gates XL Fable: no escalation onto the account."""
    cfg = _tight_gate(tmp_path, monkeypatch, tiers={"M": [_OPUS], "XL": [_FABLE]}, escalate=True)
    d = _decide(conn, cfg, _card(conn, "m", 0, complexity="M"))
    assert d.source == "quota_held", d.label()
    assert [(s["model"], s["reason"]) for s in d.skipped] == [
        ("claude-opus-5-5", f"quota: {_TIGHT}"), ("claude-fable-5-2", f"quota: {_TIGHT}")]


# ---------------------------------------------------------------------------
# In-process fallbacks arm the cooldown (provider_fallback events)
# ---------------------------------------------------------------------------


def _seed_fallback_run(conn, *, model, provider, fb, at, routed_at=None, outcome="completed", title="fb"):
    """A run routed ``provider:model`` that fell back in-process (event ``fb`` at ``at``)
    and then ended ``outcome`` — the shape the outcome query cannot see."""
    tid = kb.create_task(conn, title=title, assignee="w", complexity="M")
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "routed", {"source": "tier", "model": model, "provider": provider},
                         run_id=run_id)
        kb._append_event(conn, tid, "provider_fallback", fb, run_id=run_id)
    conn.execute("UPDATE task_events SET created_at=? WHERE run_id=? AND kind='routed'",
                 (routed_at if routed_at is not None else at, run_id))
    conn.execute("UPDATE task_events SET created_at=? WHERE run_id=? AND kind='provider_fallback'", (at, run_id))
    conn.execute("UPDATE task_runs SET outcome=?, status=?, ended_at=? WHERE id=?", (outcome, outcome, at + 60, run_id))
    conn.execute("UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?", (tid,))
    conn.commit()
    return tid


def _fb(from_model, reason="rate_limit", from_provider="custom", to_model="mid-1", to_provider="prov-b"):
    return {"from_provider": from_provider, "from_model": from_model,
            "to_provider": to_provider, "to_model": to_model, "reason": reason}


FB_TIERS = {"M": [{"model": "g", "provider": "antigravity"}, {"model": "mid-1", "provider": "prov-b"}]}


def test_fallback_event_arms_cooldown(conn, monkeypatch):
    _pool(monkeypatch)
    now = time.time()
    at = int(now) - 60
    _seed_fallback_run(conn, model="g", provider="antigravity", fb=_fb("g"), at=at)
    ctx = kr.RoutingContext(conn, _cfg(tiers=FB_TIERS, cooldown_seconds=900), now=now)
    avail = ctx.availability(kr.TierCandidate("g", "antigravity"), None)
    assert avail.available is False
    assert avail.until == at + 900
    assert avail.reason == "rate_limited on this board (fallback)"
    assert ctx.decide(_task(complexity="M")).candidate.model == "mid-1"


def test_fallback_overloaded_ignored(conn, monkeypatch):
    _pool(monkeypatch)
    now = time.time()
    for reason in ("overloaded", "timeout", "server_error", ""):
        _seed_fallback_run(conn, model="g", provider="antigravity", fb=_fb("g", reason=reason), at=int(now) - 60)
    ctx = kr.RoutingContext(conn, _cfg(tiers=FB_TIERS, cooldown_seconds=900), now=now)
    assert ctx.availability(kr.TierCandidate("g", "antigravity"), None).available is True


def test_second_hop_not_counted(conn, monkeypatch):
    """A later hop (mid-1 -> other) on a run routed to ``g`` benches neither model."""
    _pool(monkeypatch)
    now = time.time()
    _seed_fallback_run(conn, model="g", provider="antigravity",
                       fb=_fb("mid-1", from_provider="prov-b", to_model="x", to_provider="prov-x"),
                       at=int(now) - 60)
    ctx = kr.RoutingContext(conn, _cfg(tiers=FB_TIERS, cooldown_seconds=900), now=now)
    assert ctx.recently_rate_limited() == {}
    assert ctx.availability(kr.TierCandidate("g", "antigravity"), None).available is True
    assert ctx.availability(kr.TierCandidate("mid-1", "prov-b"), None).available is True


def test_fallback_outside_window_ignored(conn, monkeypatch):
    _pool(monkeypatch)
    now = time.time()
    _seed_fallback_run(conn, model="g", provider="antigravity", fb=_fb("g"), at=int(now) - 901)
    ctx = kr.RoutingContext(conn, _cfg(tiers=FB_TIERS, cooldown_seconds=900), now=now)
    assert ctx.availability(kr.TierCandidate("g", "antigravity"), None).available is True


def test_fallback_and_outcome_merge_by_max(conn, monkeypatch):
    """The newer of an outcome row and a fallback event sets ``until``; the reason names the source."""
    _pool(monkeypatch)
    now = time.time()
    first = kb.create_task(conn, title="limited", assignee="w", complexity="M")
    _seed_rate_limited_run(conn, first, model="g", provider="antigravity", ended_at=int(now) - 300)
    _seed_fallback_run(conn, model="g", provider="antigravity", fb=_fb("g"), at=int(now) - 100)
    ctx = kr.RoutingContext(conn, _cfg(tiers=FB_TIERS, cooldown_seconds=900), now=now)
    avail = ctx.availability(kr.TierCandidate("g", "antigravity"), None)
    assert avail.until == int(now) - 100 + 900 and avail.reason.endswith("(fallback)")

    later = kr.RoutingContext(conn, _cfg(tiers=FB_TIERS, cooldown_seconds=900), now=now)
    second = kb.create_task(conn, title="limited-2", assignee="w", complexity="M")
    _seed_rate_limited_run(conn, second, model="g", provider="antigravity", ended_at=int(now) - 10)
    avail = later.availability(kr.TierCandidate("g", "antigravity"), None)
    assert avail.until == int(now) - 10 + 900 and avail.reason == "rate_limited on this board"


# Incident 2026-10-06 (UTC): run 681 (t_18b37fe4) routed antigravity at 09:28:59, fell back
# to anthropic on a 429 (log has no timestamp; approximated at 09:29:30, inside the run's
# 09:28:59-09:35:08 window) and ended completed. The dispatcher kept routing antigravity.
_INCIDENT_ROUTED = 1791278939  # 2026-10-06T09:28:59Z
_INCIDENT_FALLBACK = 1791278970  # 2026-10-06T09:29:30Z
_INCIDENT_TICKS = {"09:30:00": 1791279000, "09:40:02": 1791279602, "09:45:05": 1791279905}
_INCIDENT_TIERS = {"M": [{"model": "gemini-3.8-flash-tiered", "provider": "antigravity"},
                         {"model": "claude-sonnet-5-5", "provider": "anthropic"}]}


def _incident_picks(conn) -> dict[str, str]:
    cfg = _cfg(tiers=_INCIDENT_TIERS, cooldown_seconds=900)
    return {label: kr.RoutingContext(conn, cfg, now=ts).decide(_task(complexity="M")).candidate.provider
            for label, ts in _INCIDENT_TICKS.items()}


def test_incident_681_replay(conn, monkeypatch):
    _pool(monkeypatch)
    _seed_fallback_run(
        conn, model="gemini-3.8-flash-tiered", provider="antigravity",
        fb=_fb("gemini-3.8-flash-tiered", to_model="claude-sonnet-5-5", to_provider="anthropic"),
        at=_INCIDENT_FALLBACK, routed_at=_INCIDENT_ROUTED, title="t_18b37fe4")

    picks = _incident_picks(conn)
    assert picks["09:30:00"] == "anthropic"
    assert picks["09:40:02"] == "anthropic"
    assert picks["09:45:05"] == "antigravity"  # 09:29:30 + 900 s has passed

    # On the old (outcome-only) query the cooldown never arms: all three pick antigravity.
    monkeypatch.setattr(kr.RoutingContext, "_fallback_hits", lambda self, since: [])
    assert set(_incident_picks(conn).values()) == {"antigravity"}


def _to_review(conn, tid):
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
    conn.commit()


REVIEW = [{"model": "rev-1", "provider": "prov-r", "reasoning": "high"}, {"model": "big-1", "provider": "prov-c"}]


def test_review_lane_without_review_list_ignores_tiers(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch)
    _pool(monkeypatch)
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="s", assignee="w", complexity="S")
    pinned = kb.create_task(conn, title="p", assignee="w", model_override="mine", provider_override="prov-z")
    _to_review(conn, tid)
    _to_review(conn, pinned)
    kbd.dispatch_once(conn, spawn_fn=spawn)
    assert sorted(seen) == sorted([(tid, None, None, None), (pinned, "mine", "prov-z", None)])


def test_review_list_routes_reviewer_over_card_pin(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch, review=REVIEW)
    _pool(monkeypatch)
    seen, spawn = _spawns(monkeypatch)
    tid = kb.create_task(conn, title="p", assignee="w", complexity="S",
                         model_override="mine", provider_override="prov-z")
    kb.set_reasoning_effort(conn, tid, "low")
    _to_review(conn, tid)

    res = kbd.dispatch_once(conn, spawn_fn=spawn)

    # Reviewer runs on the review model at ITS effort; the implementer's pin stays on the card.
    assert seen == [(tid, "rev-1", "prov-r", "high")]
    assert res.routed == [(tid, "review -> prov-r:rev-1")]
    card = kb.get_task(conn, tid)
    assert (card.model_override, card.provider_override, card.reasoning_effort) == ("mine", "prov-z", "low")
    ev = next(e for e in kb.list_events(conn, tid) if e.kind == "routed")
    assert ev.run_id == card.current_run_id
    assert ev.payload["lane"] == "review" and ev.payload["model"] == "rev-1"


def test_review_list_skips_unavailable_and_ready_lane_is_unaffected(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch, review=REVIEW)
    _pool(monkeypatch, {"prov-r": "credential pool exhausted"})
    seen, spawn = _spawns(monkeypatch)
    rev = kb.create_task(conn, title="r", assignee="w")
    _to_review(conn, rev)
    ready = kb.create_task(conn, title="s", assignee="w", complexity="M")
    kbd.dispatch_once(conn, spawn_fn=spawn)
    by_id = {s[0]: s[1:] for s in seen}
    assert by_id[rev] == ("big-1", "prov-c", None)
    assert by_id[ready] == ("mid-1", "prov-b", None)


def test_review_list_exhausted_holds_without_reserving_a_slot(conn, monkeypatch, all_assignees_spawnable):
    _use_routing(monkeypatch, review=REVIEW)
    _pool(monkeypatch, {"prov-r": "x", "prov-c": "x"})
    seen, spawn = _spawns(monkeypatch)
    rev = kb.create_task(conn, title="r", assignee="w")
    _to_review(conn, rev)
    ready = kb.create_task(conn, title="s", assignee="w", complexity="M")

    # Budget of one: a held review row must not hold back the ready lane's only slot.
    kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
    assert [s[0] for s in seen] == [ready]

    res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert [s[0] for s in seen] == [ready]
    assert (rev, "tier_exhausted") in res.respawn_guarded
    task = kb.get_task(conn, rev)
    assert task.status == "review" and task.consecutive_failures == 0
    hold = next(e for e in kb.list_events(conn, rev) if e.kind == "routing_held")
    assert hold.payload["lane"] == "review" and hold.payload["retry_at"]

    fb_seen, fb_spawn = _spawns(monkeypatch)
    _use_routing(monkeypatch, review=REVIEW, on_exhausted="profile")
    kbd.dispatch_once(conn, spawn_fn=fb_spawn)
    assert fb_seen == [(rev, None, None, None)]


# ---------------------------------------------------------------------------
# Auto-label
# ---------------------------------------------------------------------------


def test_auto_label_sets_and_records_failures(conn, monkeypatch):
    replies = iter([{"ok": True, "complexity": "M"}, {"ok": False, "reason": "LLM error: Timeout"}])
    monkeypatch.setattr(kr, "estimate_complexity", lambda *a, **k: next(replies))
    a = kb.create_task(conn, title="a", assignee="w")
    b = kb.create_task(conn, title="b", assignee="w")
    kb.create_task(conn, title="labelled", assignee="w", complexity="S")
    kb.create_task(conn, title="pinned", assignee="w", model_override="x")

    out = kr.auto_label(conn, 10)

    assert out == [(a, "M"), (b, None)]
    assert kb.get_task(conn, a).complexity == "M"
    assert kr.unlabeled_ready_ids(conn, 10) == []  # failures are not retried every tick


# ---------------------------------------------------------------------------
# CLI + tool surfaces
# ---------------------------------------------------------------------------


def test_cli_create_set_and_route(kanban_home, monkeypatch, capsys):
    from hermes_cli import kanban as kcli
    from hermes_cli.kanban_parser import build_parser

    monkeypatch.setattr(kcli, "_kanban_config", lambda: {"routing": {"enabled": True, "tiers": TIERS}})
    _pool(monkeypatch)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="cmd"))

    def run(*argv):
        args = parser.parse_args(["kanban", *argv])
        return kcli.kanban_command(args)

    assert run("create", "cli card", "--assignee", "w", "--complexity", "s", "--json") == 0
    tid = json.loads(capsys.readouterr().out)["id"]
    with kbc.connect_closing() as c:
        assert kb.get_task(c, tid).complexity == "S"

    assert run("set-complexity", tid, "L") == 0
    assert "Set complexity" in capsys.readouterr().out
    assert run("route", tid, "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["enabled"] is True and out["task"]["model"] == "big-1"
    assert run("set-complexity", tid, "XXL") == 2
    assert run("set-complexity", tid, "xl") == 0
    capsys.readouterr()
    with kbc.connect_closing() as c:
        assert kb.get_task(c, tid).complexity == "XL"
    assert run("set-complexity", tid, "L") == 0
    capsys.readouterr()

    monkeypatch.setattr(kcli, "_kanban_config", lambda: {"routing": {"enabled": True, "tiers": TIERS,
                                                                    "review": REVIEW}})
    with kbc.connect_closing() as c:
        _to_review(c, tid)
    assert run("route", tid, "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["model"] for r in out["review"]] == ["rev-1", "big-1"]
    assert out["task"]["lane"] == "review" and out["task"]["model"] == "rev-1"
    assert run("route", tid) == 0
    text = capsys.readouterr().out
    assert "Review lane:" in text and "-> review -> prov-r:rev-1" in text


def test_kanban_create_tool_accepts_complexity(kanban_home, monkeypatch):
    from tools import kanban_tools as kt
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA

    assert KANBAN_CREATE_SCHEMA["parameters"]["properties"]["complexity"]["enum"] == list(kb.VALID_COMPLEXITIES)
    assert "XL" in kb.VALID_COMPLEXITIES
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = json.loads(kt._handle_create({"title": "tool card", "assignee": "w", "complexity": "m"}))
    assert out.get("ok") is not False, out
    with kbc.connect_closing() as c:
        assert kb.get_task(c, out["task_id"]).complexity == "M"


# ---------------------------------------------------------------------------
# Data class policy integration
# ---------------------------------------------------------------------------


def test_confidential_card_skips_disallowed_candidates(conn, monkeypatch):
    from agent import provider_policy as pp

    monkeypatch.setattr(pp, "allowed_providers", lambda cls: ["prov-a"] if cls == "confidential" else "any")
    monkeypatch.setattr(pp, "fallback_allowed", lambda cls: False if cls == "confidential" else True)

    cfg = _cfg(tiers={
        "S": [{"model": "m-a", "provider": "prov-a"}, {"model": "m-b", "provider": "prov-b"}]
    })
    ctx = kr.RoutingContext(conn, cfg)
    # prov-a gated by quota (mode tight), prov-b available
    monkeypatch.setattr(ctx, "quota_verdict", lambda prov, prio, model=None: kr.QuotaVerdict(
        provider="prov-a", headroom=0.0, below=0.0, required=2, resets_at=time.time() + 3600, kind="mode", mode="tight"
    ) if prov == "prov-a" else None)
    _pool(monkeypatch, {})  # both pool-available

    task = _task(complexity="S", data_class="confidential", priority=1)
    d = ctx.decide(task)
    assert d.source == "quota_held"
    assert d.candidate is None
    # prov-b must be skipped with the data class reason, NOT chosen
    skipped_reasons = {s["provider"]: s["reason"] for s in d.skipped}
    assert "data class confidential: provider not in allowlist" in skipped_reasons.get("prov-b", "")
    assert "quota: " in skipped_reasons.get("prov-a", "")


def test_confidential_card_routes_to_allowed_provider(conn, monkeypatch):
    from agent import provider_policy as pp

    monkeypatch.setattr(pp, "allowed_providers", lambda cls: ["prov-a"] if cls == "confidential" else "any")
    cfg = _cfg(tiers={
        "S": [{"model": "m-b", "provider": "prov-b"}, {"model": "m-a", "provider": "prov-a"}]
    })
    ctx = kr.RoutingContext(conn, cfg)
    _pool(monkeypatch, {})
    task = _task(complexity="S", data_class="confidential", priority=1)
    d = ctx.decide(task)
    assert d.source == "tier"
    assert d.candidate.provider == "prov-a"
    assert d.candidate.model == "m-a"
    assert any("data class confidential: provider not in allowlist" in s["reason"] for s in d.skipped if s["provider"] == "prov-b")


def test_internal_card_unchanged(conn, monkeypatch):
    from agent import provider_policy as pp

    monkeypatch.setattr(pp, "allowed_providers", lambda cls: ["prov-a"] if cls == "confidential" else "any")
    cfg = _cfg(tiers={
        "S": [{"model": "m-a", "provider": "prov-a"}, {"model": "m-b", "provider": "prov-b"}]
    })
    ctx = kr.RoutingContext(conn, cfg)
    # prov-a gated by quota, prov-b available
    monkeypatch.setattr(ctx, "quota_verdict", lambda prov, prio, model=None: kr.QuotaVerdict(
        provider="prov-a", headroom=0.0, below=0.0, required=2, resets_at=time.time() + 3600, kind="mode", mode="tight"
    ) if prov == "prov-a" else None)
    _pool(monkeypatch, {})

    task = _task(complexity="S", data_class="internal", priority=1)
    d = ctx.decide(task)
    assert d.source == "tier"
    assert d.candidate.provider == "prov-b"


def test_review_lane_respects_data_class(conn, monkeypatch):
    from agent import provider_policy as pp

    monkeypatch.setattr(pp, "allowed_providers", lambda cls: ["prov-a"] if cls == "confidential" else "any")
    cfg = _cfg(review=[{"model": "m-b", "provider": "prov-b"}, {"model": "m-a", "provider": "prov-a"}])
    ctx = kr.RoutingContext(conn, cfg)
    _pool(monkeypatch, {})

    task = _task(data_class="confidential")
    d = ctx.decide_review(task)
    assert d.source == "review"
    assert d.candidate.provider == "prov-a"
    assert any("data class confidential: provider not in allowlist" in s["reason"] for s in d.skipped if s["provider"] == "prov-b")


def test_unresolvable_class_holds(conn, monkeypatch):
    from agent import provider_policy as pp

    def boom(task):
        raise pp.ProviderDenied("unknown data class")

    monkeypatch.setattr(pp, "resolve_data_class", boom)
    cfg = _cfg(tiers={"S": [{"model": "m-a", "provider": "prov-a"}]})
    ctx = kr.RoutingContext(conn, cfg)
    _pool(monkeypatch, {})

    task = _task(complexity="S", data_class="invalid")
    d = ctx.decide(task)
    assert d.source == "exhausted"
    assert any(s["reason"] == "data class unresolved" for s in d.skipped)


def test_provider_permitted_must_not_match_substring(conn, monkeypatch):
    from agent import provider_policy as pp

    monkeypatch.setattr(pp, "allowed_providers", lambda cls: ["anthropic"] if cls == "confidential" else "any")
    cfg = _cfg(tiers={
        "S": [
            {"model": "m-sub", "provider": "anthropic-proxy"},
            {"model": "m-case", "provider": "Anthropic"},
        ]
    })
    ctx = kr.RoutingContext(conn, cfg)
    _pool(monkeypatch, {})

    task = _task(complexity="S", data_class="confidential")
    d = ctx.decide(task)
    assert d.source == "tier"
    assert d.candidate.provider == "Anthropic"
    assert any(s["provider"] == "anthropic-proxy" and "not in allowlist" in s["reason"] for s in d.skipped)


def test_incident_replay_decide_and_review(conn, monkeypatch):
    """Incident replay:
    Card t_eca0bf9c / t_5a49507e: confidential, P1, anthropic gated by mode tight (needs P2),
    xai-oauth pool exhausted, antigravity pool available.
    Old code returned antigravity -> crashed.
    New code must return quota_held on both ready and review lanes.
    """
    from agent import provider_policy as pp

    monkeypatch.setattr(pp, "allowed_providers", lambda cls: ["anthropic", "openai-codex"] if cls == "confidential" else "any")
    monkeypatch.setattr(pp, "fallback_allowed", lambda cls: False if cls == "confidential" else True)

    cands = [
        {"model": "claude-opus-5-5", "provider": "anthropic"},
        {"model": "grok-4.7", "provider": "xai-oauth"},
        {"model": "gemini-3.8-flash-tiered", "provider": "antigravity"},
    ]
    cfg = _cfg(tiers={"S": cands}, review=cands)
    ctx = kr.RoutingContext(conn, cfg)

    # anthropic gated: mode tight needs P2
    monkeypatch.setattr(ctx, "quota_verdict", lambda prov, prio, model=None: kr.QuotaVerdict(
        provider="anthropic", headroom=0.0, below=0.0, required=2, resets_at=time.time() + 3600, kind="mode", mode="tight"
    ) if prov == "anthropic" else None)
    # xai-oauth exhausted
    _pool(monkeypatch, {"xai-oauth": "credential pool exhausted"})

    task = _task(complexity="S", data_class="confidential", priority=1)

    # Ready lane
    d_ready = ctx.decide(task)
    assert d_ready.source == "quota_held", f"expected quota_held, got {d_ready.source}: {d_ready.candidate}"
    assert d_ready.candidate is None

    # Review lane
    d_review = ctx.decide_review(task)
    assert d_review.source == "quota_held", f"expected quota_held, got {d_review.source}: {d_review.candidate}"
    assert d_review.candidate is None


def _write_data_policies(home: Path, confidential: list[str]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "kanban:\n  data_policies:\n    default: internal\n    classes:\n"
        "      internal:\n        providers: any\n        fallback: true\n"
        f"      confidential:\n        providers: {json.dumps(confidential)}\n        fallback: false\n",
        encoding="utf-8",
    )


def test_allowlist_resolves_under_assignee_profile_home(conn, kanban_home, tmp_path, monkeypatch):
    """The allowlist comes from the assignee profile's config, not the dispatcher's: the worker
    runs under the profile home and would deny a provider the dispatcher's policy permits."""
    _write_data_policies(kanban_home, ["prov-a"])
    profile_home = tmp_path / "profiles" / "worker"
    _write_data_policies(profile_home, ["prov-b"])

    cfg = _cfg(tiers={"S": [{"model": "m-a", "provider": "prov-a"}, {"model": "m-b", "provider": "prov-b"}]})
    ctx = kr.RoutingContext(conn, cfg)
    _pool(monkeypatch, {})

    task = _task(complexity="S", data_class="confidential")
    d = ctx.decide(task, profile_home=str(profile_home))
    assert d.source == "tier"
    assert d.candidate.provider == "prov-b"
    assert any(s["provider"] == "prov-a" and "not in allowlist" in s["reason"] for s in d.skipped)

    # Without a profile home the dispatcher's own policy applies.
    d_default = ctx.decide(task)
    assert d_default.candidate.provider == "prov-a"


# Two real homes (no allowed_providers monkeypatch): the router must read the
# allowlist under the assignee's home, where the worker enforces it (t_359863a0).

_A = {"model": "m-a", "provider": "prov-a"}
_B = {"model": "m-b", "provider": "prov-b"}


def _policy_home(home: Path, allowed: str, model: str | None = None) -> str:
    home.mkdir(exist_ok=True)
    model_block = f"model:\n  default: {model.split('/')[0]}\n  provider: {model.split('/')[1]}\n" if model else ""
    (home / "config.yaml").write_text(
        model_block
        + "kanban:\n  data_policies:\n    default: internal\n    classes:\n"
        "      internal: {providers: any, fallback: true, auxiliary: any}\n"
        f"      confidential: {{providers: [{allowed}], fallback: false, auxiliary: same_provider_or_fail}}\n")
    return str(home)


@pytest.fixture
def homes(kanban_home, tmp_path):
    """Dispatcher home allows prov-a; profile ``prof`` allows prov-b and runs m-x/prov-x
    (not permitted); profile ``ok`` allows prov-b and runs m-b/prov-b (permitted)."""
    _policy_home(kanban_home, "prov-a")
    return {"prof": _policy_home(tmp_path / "prof", "prov-b", "m-x/prov-x"),
            "ok": _policy_home(tmp_path / "ok", "prov-b", "m-b/prov-b")}


def test_on_exhausted_profile_holds_on_data_class_skip(conn, monkeypatch, homes):
    _pool(monkeypatch)
    cfg = _cfg(tiers={"S": [_A]}, on_exhausted="profile")
    task = _task(complexity="S", data_class="confidential", priority=1)
    d = kr.RoutingContext(conn, cfg).decide(task, profile_home=homes["prof"])
    assert d.source == "exhausted" and d.candidate is None
    assert "no permitted candidate" in d.note
    assert [s["provider"] for s in d.skipped] == ["prov-a", "prov-x"]
    assert "no permitted candidate" in d.label()
    # Must-not: a permitted profile model still takes the profile route.
    d = kr.RoutingContext(conn, cfg).decide(task, profile_home=homes["ok"])
    assert (d.source, d.candidate.label) == ("profile", "prov-b:m-b")
    # Must-not: an unrestricted class falls to the profile model unchanged.
    _pool(monkeypatch, {"prov-a": "credential pool exhausted"})
    d = kr.RoutingContext(conn, cfg).decide(_task(complexity="S", data_class="internal"),
                                            profile_home=homes["prof"])
    assert (d.source, d.candidate.label) == ("profile", "prov-x:m-x")


def test_review_on_exhausted_profile_holds_on_data_class_skip(conn, monkeypatch, homes):
    _pool(monkeypatch)
    cfg = _cfg(review=[_A], on_exhausted="profile")
    task = _task(data_class="confidential")
    d = kr.RoutingContext(conn, cfg).decide_review(task, profile_home=homes["prof"])
    assert (d.source, d.lane, d.candidate) == ("exhausted", "review", None)
    assert "no permitted candidate" in d.note and "no permitted candidate" in d.label()
    assert d.skipped[-1]["provider"] == "prov-x"
    # Must-not: the permitted profile model keeps the profile run.
    d = kr.RoutingContext(conn, cfg).decide_review(task, profile_home=homes["ok"])
    assert (d.source, d.lane) == ("profile", "review")
    # The card pin is what the review run keeps: a forbidden pin holds even
    # though the profile model would be permitted, and a permitted pin passes.
    pinned = _task(data_class="confidential", model_override="m-z", provider_override="prov-z")
    d = kr.RoutingContext(conn, cfg).decide_review(pinned, profile_home=homes["ok"])
    assert d.source == "exhausted" and d.skipped[-1]["provider"] == "prov-z"
    pinned_ok = _task(data_class="confidential", model_override="m-b", provider_override="prov-b")
    d = kr.RoutingContext(conn, cfg).decide_review(pinned_ok, profile_home=homes["prof"])
    assert d.source == "profile"


def test_unlabeled_profile_route_checks_data_class(conn, monkeypatch, homes):
    _pool(monkeypatch)
    task = _task(data_class="confidential")
    for cfg in (_cfg(), _cfg(tiers={})):  # unlabeled, then no candidates configured
        d = kr.RoutingContext(conn, cfg).decide(task, profile_home=homes["prof"])
        assert d.source == "exhausted" and d.candidate is None
        assert d.note == "data class confidential: no permitted candidate"
        d = kr.RoutingContext(conn, cfg).decide(task, profile_home=homes["ok"])
        assert (d.source, d.candidate.label) == ("profile", "prov-b:m-b")


def test_unresolved_class_holds_profile_routes(conn, monkeypatch, homes):
    _pool(monkeypatch, {"prov-a": "credential pool exhausted"})
    task = _task(complexity="S", data_class="bogus")
    for d in (kr.RoutingContext(conn, _cfg()).decide(_task(data_class="bogus"), profile_home=homes["ok"]),
              kr.RoutingContext(conn, _cfg(tiers={"S": [_A]}, on_exhausted="profile")).decide(
                  task, profile_home=homes["ok"]),
              kr.RoutingContext(conn, _cfg(review=[_A], on_exhausted="profile")).decide_review(
                  task, profile_home=homes["ok"])):
        assert d.source == "exhausted" and d.candidate is None
        assert d.note == "data class unresolved: no permitted candidate"
        assert d.skipped[-1]["reason"] == "data class unresolved"


# ---------------------------------------------------------------------------
# Quota gate priority follows open descendants. Stored priority is not written.
# ---------------------------------------------------------------------------

_TIGHT = "anthropic quota mode tight needs P2"
_RESET = 1795233600  # 2026-11-21T04:00:00Z, the recorded retry_at
_REPLAY_NOW = 1794860431  # 2026-11-16T20:20:31Z, when the P2 child was linked
_OPUS = {"model": "claude-opus-5-5", "provider": "anthropic"}
_REPLAY_TIERS = {"M": [_OPUS], "L": [_OPUS],
                 "XL": [{"model": "claude-fable-5-2", "provider": "anthropic"}, _OPUS]}


def _p2_headroom(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": 2,
        "written_at_epoch": time.time() - 60,
        "providers": {
            "anthropic": {"ok": True, "binding": {"name": "session", "headroom": 0.05,
                                                  "resets_at": "2026-11-21T04:00:00Z"}, "windows": []},
        },
    }))


def _tight_mode(path: Path, now: float) -> None:
    path.write_text(json.dumps({"mode": "tight", "written_at_epoch": now - 60, "reason": "test",
                                "since": "2026-11-16T19:00:00Z", "resets_at": "2026-11-21T04:00:00Z"}))


def _tight_gate(tmp_path, monkeypatch, *, now: float | None = None, tiers=None, escalate=False) -> kr.RoutingConfig:
    """Quota mode tight: anthropic needs P2. Anthropic-only tiers, so a pass is the gate, not a fallthrough."""
    mode = tmp_path / "mode.json"
    _tight_mode(mode, time.time() if now is None else now)
    _pool(monkeypatch)
    return _cfg(
        tiers=tiers or {"S": [{"model": "claude-s", "provider": "anthropic"}]},
        escalate=escalate,
        review=[{"model": "rev", "provider": "anthropic"}],
        quota_gate={"enabled": True, "headroom_file": str(tmp_path / "missing.json"), "mode_file": str(mode)},
    )


def _card(conn, title, priority, **kw) -> str:
    status = kw.pop("status", None)
    tid = kb.create_task(conn, title=title, assignee="w", priority=priority, **kw)
    if status is not None:
        _status(conn, tid, status)
    return tid


def _status(conn, tid, status) -> None:
    conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
    conn.commit()


def _link(conn, parent, child, status=None) -> None:
    """Link, then pin the child's status (``link_tasks`` demotes a ready child to todo)."""
    kb.link_tasks(conn, parent, child)
    if status is not None:
        _status(conn, child, status)


def _row(conn, tid):
    task = kb.get_task(conn, tid)
    assert task is not None
    return task


def _rename(conn, old: str, new: str) -> str:
    conn.execute("UPDATE tasks SET id=? WHERE id=?", (new, old))
    conn.execute("UPDATE task_links SET child_id=? WHERE child_id=?", (new, old))
    conn.execute("UPDATE task_links SET parent_id=? WHERE parent_id=?", (new, old))
    conn.commit()
    return new


def _decide(conn, cfg, tid, **kw):
    return kr.RoutingContext(conn, cfg, **kw).decide(_row(conn, tid))


def test_gate_priority_replay_t_3152e49e(conn, monkeypatch, tmp_path):
    """2026-11-16 20:20:31Z: P1 parent held in mode tight while its blocked P2 child waited."""
    cfg = _tight_gate(tmp_path, monkeypatch, now=_REPLAY_NOW, tiers=_REPLAY_TIERS, escalate=True)
    parent = _rename(conn, _card(conn, "parent", 1, complexity="M"), "t_3152e49e")

    before = _decide(conn, cfg, parent, now=_REPLAY_NOW)
    assert before.source == "quota_held"
    assert before.note == f"{_TIGHT} (card P1)" and before.retry_at == _RESET
    assert [(s["tier"], s["model"]) for s in before.skipped] == [
        ("M", "claude-opus-5-5"), ("L", "claude-opus-5-5"), ("XL", "claude-fable-5-2"), ("XL", "claude-opus-5-5")]

    child = _rename(conn, _card(conn, "child", 2), "t_b5815f54")
    grand = _rename(conn, _card(conn, "grand", 2), "t_fe4b8c1c")
    other = _rename(conn, _card(conn, "other", 1), "t_8fb60c0d")
    _link(conn, parent, child, status="blocked")
    _link(conn, child, grand, status="todo")
    _link(conn, parent, other, status="todo")

    after = _decide(conn, cfg, parent, now=_REPLAY_NOW)
    assert after.source == "tier"
    assert after.candidate is not None and after.candidate.model == "claude-opus-5-5"
    assert kr.RoutingContext(conn, cfg, now=_REPLAY_NOW).gate_priority(_row(conn, parent)) == \
        (1, 2, ["t_b5815f54", "t_fe4b8c1c"])
    assert _row(conn, parent).priority == 1  # stored priority is not written


def test_gate_priority_still_held_when_descendants_are_lower(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    parent = _card(conn, "parent", 1, complexity="S")
    _link(conn, parent, _card(conn, "child", 1), status="blocked")
    d = _decide(conn, cfg, parent)
    assert d.source == "quota_held"
    assert d.note.endswith("(card P1)")
    assert "gate P" not in d.note and "via" not in d.note


def test_gate_priority_raised_but_still_held(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    parent = _card(conn, "parent", 0, complexity="S")
    child = _card(conn, "child", 1)
    _link(conn, parent, child, status="blocked")
    d = _decide(conn, cfg, parent)
    assert d.source == "quota_held"
    assert d.note == f"{_TIGHT} (card P0, gate P1 via {child})"


def test_gate_priority_done_and_archived_do_not_lift(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    parent = _card(conn, "parent", 1, complexity="S")
    _link(conn, parent, _card(conn, "done", 3), status="done")
    _link(conn, parent, _card(conn, "archived", 3), status="archived")
    d = _decide(conn, cfg, parent)
    assert d.source == "quota_held" and d.note == f"{_TIGHT} (card P1)"

    # The walk goes through a done or archived node: its open child still lifts.
    for status in ("done", "archived"):
        top = _card(conn, f"top-{status}", 1, complexity="S")
        mid = _card(conn, f"mid-{status}", 3)
        grand = _card(conn, f"grand-{status}", 2)
        _link(conn, top, mid)
        _link(conn, mid, grand, status="todo")
        _status(conn, mid, status)
        assert kr.RoutingContext(conn, cfg).gate_priority(_row(conn, top)) == (1, 2, [grand])
        assert _decide(conn, cfg, top).source == "tier"


def test_gate_priority_blocked_and_scheduled_lift(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    for status in ("blocked", "scheduled", "triage", "todo"):
        parent = _card(conn, f"parent-{status}", 1, complexity="S")
        _link(conn, parent, _card(conn, status, 2), status=status)
        assert _decide(conn, cfg, parent).source == "tier", status


def test_gate_priority_transitive_via_is_the_max_only(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    card = _card(conn, "card", 1, complexity="S")
    mid = _card(conn, "mid", 1)
    grand = _card(conn, "grand", 3)
    _link(conn, card, mid, status="todo")
    _link(conn, mid, grand, status="todo")
    assert kr.RoutingContext(conn, cfg).gate_priority(_row(conn, card)) == (1, 3, [grand])

    # A higher-priority parent of the card does not lift it.
    lone = _card(conn, "lone", 1, complexity="S")
    upper = _card(conn, "upper", 3, complexity="S")
    _link(conn, upper, lone)
    _status(conn, lone, "ready")
    assert kr.RoutingContext(conn, cfg).gate_priority(_row(conn, lone)) == (1, 1, [])
    d = _decide(conn, cfg, lone)
    assert d.source == "quota_held" and d.note == f"{_TIGHT} (card P1)"

    # A link to a task row that does not exist, and a cycle back to the card, do not raise.
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (lone, "t_missing_row"))
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (grand, card))
    conn.commit()
    assert kr.RoutingContext(conn, cfg).gate_priority(_row(conn, lone)) == (1, 1, [])
    assert kr.RoutingContext(conn, cfg).gate_priority(_row(conn, card)) == (1, 3, [grand])


def test_gate_priority_via_cap(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    parent = _card(conn, "parent", 0, complexity="S")
    ids = [_card(conn, f"kid{i}", 1) for i in range(9)]
    for cid in ids:
        _link(conn, parent, cid)
    ordered = sorted(ids)
    assert kr.RoutingContext(conn, cfg).gate_priority(_row(conn, parent)) == (0, 1, ordered)
    d = _decide(conn, cfg, parent)
    assert d.source == "quota_held"
    assert d.note == f"{_TIGHT} (card P0, gate P1 via {','.join(ordered[:8])},+1)"


def test_gate_priority_pinned_and_review(conn, monkeypatch, tmp_path):
    cfg = _tight_gate(tmp_path, monkeypatch)
    pinned = _card(conn, "pin", 1, model_override="claude-x", provider_override="anthropic")
    assert _decide(conn, cfg, pinned).source == "quota_held"
    _link(conn, pinned, _card(conn, "pin-c", 2), status="blocked")
    _status(conn, pinned, "ready")
    assert _decide(conn, cfg, pinned).source == "pinned"

    review = _card(conn, "rev", 1)
    held = kr.RoutingContext(conn, cfg).decide_review(_row(conn, review))
    assert held is not None and held.source == "quota_held"
    _link(conn, review, _card(conn, "rev-c", 2), status="blocked")
    passed = kr.RoutingContext(conn, cfg).decide_review(_row(conn, review))
    assert passed is not None and passed.source == "review"


def test_gate_priority_route_view(kanban_home, monkeypatch, tmp_path, capsys):
    from hermes_cli import kanban as kcli
    from hermes_cli.kanban_parser import build_parser

    path = tmp_path / "headroom.json"
    _p2_headroom(path)
    mode = tmp_path / "mode.json"
    _tight_mode(mode, time.time())
    monkeypatch.setattr(kcli, "_kanban_config", lambda: {"routing": {
        "enabled": True,
        "tiers": {"S": [{"model": "claude-s", "provider": "anthropic"}]},
        "escalate": False,
        "quota_gate": {"enabled": True, "headroom_file": str(path), "mode_file": str(mode)},
    }})
    _pool(monkeypatch)
    with kbc.connect_closing() as c:
        parent = _card(c, "parent", 1, complexity="S")
        child = _card(c, "child", 2)
        _link(c, parent, child, status="blocked")
        alone = _card(c, "alone", 1, complexity="S")
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="cmd"))

    def run(*argv):
        return kcli.kanban_command(parser.parse_args(["kanban", *argv]))

    assert run("route", parent) == 0
    out = capsys.readouterr().out
    assert f"card P1, gate P2 via {child}:" in out
    anth = next(ln for ln in out.splitlines() if "anthropic:" in ln and "needs P2" in ln)
    assert anth.strip().startswith("ok ")

    assert run("route", alone) == 0
    out = capsys.readouterr().out
    assert "card P1:" in out and "gate P2 via" not in out
    anth = next(ln for ln in out.splitlines() if "anthropic:" in ln and "needs P2" in ln)
    assert anth.strip().startswith("-- ")

    assert run("route", parent, "--json") == 0
    raised = json.loads(capsys.readouterr().out)["quota_gate"]
    assert raised["card_priority"] == 1 and raised["gate_priority"] == 2 and raised["gate_via"] == child
    assert run("route", alone, "--json") == 0
    flat = json.loads(capsys.readouterr().out)["quota_gate"]
    assert flat["card_priority"] == 1 and flat["gate_priority"] == 1 and flat["gate_via"] == ""
