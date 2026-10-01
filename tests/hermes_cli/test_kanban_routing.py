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

    def fake(provider, model, *, now=None):
        if provider in blocked:
            return kr.Availability(False, blocked[provider], until=time.time() + 600)
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
    for empty in (None, "", "none", "-", "NULL"):
        assert kb.normalize_complexity(empty) is None
    with pytest.raises(ValueError):
        kb.normalize_complexity("XL")


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
            "XL": [{"model": "ignored"}],
            "L": {"model": "single", "reasoning": "not-a-level"},
        },
    }})
    assert [c.model for c in cfg.tiers["S"]] == ["a"]
    assert "XL" not in cfg.tiers
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


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _task(**kw):
    return kb.Task(id="t_x", title="x", body=None, assignee="w", status="ready", priority=0,
                   created_by=None, created_at=0, started_at=None, completed_at=None,
                   workspace_kind="scratch", workspace_path=None, claim_lock=None,
                   claim_expires=None, tenant=None, **kw)


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
    assert run("set-complexity", tid, "XL") == 2

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

    assert KANBAN_CREATE_SCHEMA["parameters"]["properties"]["complexity"]["enum"] == ["S", "M", "L"]
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = json.loads(kt._handle_create({"title": "tool card", "assignee": "w", "complexity": "m"}))
    assert out.get("ok") is not False, out
    with kbc.connect_closing() as c:
        assert kb.get_task(c, out["task_id"]).complexity == "M"
