"""Pace bands and pace ordering in Kanban tier routing.

``kanban.routing.quota_gate.bands`` entries may carry ``pace_below`` (compared
against headroom.json ``pace.pace_headroom`` minus ``quota_gate.reserve``);
``kanban.routing.pace_order`` sorts each tier's candidates by that pace.
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

_LOAD = kr.load_routing_config


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


@pytest.fixture
def headroom(tmp_path):
    """Write a schema-2 headroom file. ``spec`` per provider: ``(headroom, pace)``;
    ``pace=None`` writes the parent card's all-null pace block (unknown)."""
    path = tmp_path / "headroom.json"

    def write(providers: dict) -> Path:
        out = {}
        for name, (h, pace) in providers.items():
            out[name] = {
                "ok": True, "windows": [],
                "binding": {"name": "week", "headroom": h, "resets_at": "2026-10-10T03:59:59+00:00"},
                "pace": {"window_seconds": 604800 if pace is not None else None,
                         "time_left_fraction": None if pace is None else 0.8,
                         "burn_pct_per_day": None, "projected_used_pct_at_reset": None,
                         "pace_headroom": pace},
            }
        path.write_text(json.dumps({"schema": 2, "written_at_epoch": time.time() - 60, "providers": out}))
        return path

    return write


LEVEL_BANDS = {
    "anthropic": [{"below": 0.35, "min_priority": 1}, {"below": 0.20, "min_priority": 2},
                  {"below": 0.08, "min_priority": 3}],
    "openai-codex": [{"below": 0.45, "min_priority": 1}, {"below": 0.30, "min_priority": 2},
                     {"below": 0.10, "min_priority": 3}],
}
PACE = [{"pace_below": -0.10, "min_priority": 1}, {"pace_below": -0.25, "min_priority": 2},
        {"pace_below": -0.40, "min_priority": 3}]
PACE_BANDS = {p: LEVEL_BANDS[p] + PACE for p in ("anthropic", "openai-codex")}
TIERS = {
    "M": [{"model": "claude-m", "provider": "anthropic"}, {"model": "gpt-m", "provider": "openai-codex"},
          {"model": "gem-m", "provider": "antigravity"}],
    "L": [{"model": "claude-l", "provider": "anthropic"}, {"model": "gpt-l", "provider": "openai-codex"}],
}


def _routing(path, *, bands=PACE_BANDS, reserve=None, pace_order=False, **over) -> dict:
    gate = {"enabled": True, "headroom_file": str(path), "bands": bands}
    if reserve is not None:
        gate["reserve"] = reserve
    return {"routing": {"enabled": True, "tiers": TIERS, "escalate": False, "quota_gate": gate,
                        "pace_order": pace_order, **over}}


def _cfg(path, **kw) -> kr.RoutingConfig:
    return _LOAD(_routing(path, **kw))


def _pool(monkeypatch):
    monkeypatch.setattr(kr, "pool_availability",
                        lambda provider, model, *, now=None: kr.Availability(True, "pool ok"))


def _task(**kw):
    kw.setdefault("priority", 0)
    return kb.Task(id="t_x", title="x", body=None, assignee="w", status="ready",
                   created_by=None, created_at=0, started_at=None, completed_at=None,
                   workspace_kind="scratch", workspace_path=None, claim_lock=None,
                   claim_expires=None, tenant=None, **kw)


def _order(ctx, tier="M"):
    return [c.provider for c in ctx.order_tier(ctx.cfg.tiers[tier])]


# --- config -----------------------------------------------------------------------


def test_parse_pace_bands_reserve_and_pace_order():
    cfg = _LOAD({"routing": {"pace_order": True, "quota_gate": {
        "bands": {"anthropic": [{"below": 0.2, "pace_below": -0.3, "min_priority": 2},
                                {"pace_below": -0.1, "min_priority": 1},
                                {"min_priority": 3}]},  # neither kind: ignored
        "reserve": {"Anthropic": "0.2", "openai-codex": "bad"}}}})
    gate = cfg.quota_gate
    assert cfg.pace_order is True
    assert gate.bands_for("anthropic") == ((0.2, 2),)
    assert gate.pace_bands_for("anthropic") == ((-0.3, 2), (-0.1, 1))
    assert gate.reserve_for("anthropic") == 0.2
    assert gate.reserve_for("openai-codex") == 0.0  # bad value dropped -> default 0


def test_level_only_config_parses_unchanged():
    """An existing level-only config keeps its exact bands and gets no pace bands,
    no reserve and pace_order off."""
    cfg = _LOAD({"routing": {"quota_gate": {"enabled": True, "bands": LEVEL_BANDS}}})
    gate = cfg.quota_gate
    assert gate.bands_for("anthropic") == ((0.35, 1), (0.20, 2), (0.08, 3))
    assert gate.pace_bands == {} and gate.reserve == {} and cfg.pace_order is False


# --- the four acceptance cases ------------------------------------------------------


def test_level_only_config_ignores_pace_in_file(conn, headroom, monkeypatch):
    """Level-only bands: a deeply negative pace in the file changes nothing,
    neither the gate nor the walk order."""
    _pool(monkeypatch)
    path = headroom({"anthropic": (0.63, -0.90), "openai-codex": (0.52, -0.90)})
    ctx = kr.RoutingContext(conn, _cfg(path, bands=LEVEL_BANDS))
    assert ctx.quota_verdict("anthropic", 0) is None
    d = ctx.decide(_task(complexity="M", priority=0))
    assert d.source == "tier" and d.candidate.provider == "anthropic" and d.skipped == []
    assert _order(ctx) == ["anthropic", "openai-codex", "antigravity"]


def test_pace_band_holds_p0_and_passes_p2(conn, headroom, monkeypatch):
    """Anthropic at 63% left (no level band hit) but pace -0.19 - reserve 0.20 =
    -0.39 < -0.25: P0 skips it, P2 gets it."""
    _pool(monkeypatch)
    path = headroom({"anthropic": (0.63, -0.19), "openai-codex": (0.52, 0.10)})
    ctx = kr.RoutingContext(conn, _cfg(path, reserve={"anthropic": 0.20}))

    v = ctx.quota_verdict("anthropic", 0)
    assert v is not None and v.kind == "pace" and v.required == 2
    assert v.describe() == "anthropic pace -0.39 < -0.25 needs P2"

    p0 = ctx.decide(_task(complexity="M", priority=0))
    assert p0.source == "tier" and p0.candidate.provider == "openai-codex"
    assert p0.skipped[0]["reason"] == "quota: anthropic pace -0.39 < -0.25 needs P2"

    p2 = ctx.decide(_task(complexity="M", priority=2))
    assert p2.source == "tier" and p2.candidate.provider == "anthropic" and p2.skipped == []


def test_required_priority_is_max_over_level_and_pace(conn, headroom):
    path = headroom({"anthropic": (0.15, -0.12), "openai-codex": (0.40, -0.45)})
    ctx = kr.RoutingContext(conn, _cfg(path))
    level = ctx.quota_verdict("anthropic", 1)  # level < 20% -> P2 beats pace -0.12 -> P1
    assert level.kind == "level" and level.required == 2
    assert level.describe() == "anthropic headroom 15% < 20% needs P2"
    pace = ctx.quota_verdict("openai-codex", 1)  # pace -0.45 -> P3 beats level < 45% -> P1
    assert pace.kind == "pace" and pace.required == 3
    assert ctx.quota_verdict("openai-codex", 3) is None


def test_pace_order_flips_when_paces_cross(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    # No pace bands here: this isolates ordering from gating.
    ahead_a = headroom({"anthropic": (0.9, 0.30), "openai-codex": (0.9, 0.05), "antigravity": (0.9, 0.20)})
    ctx = kr.RoutingContext(conn, _cfg(ahead_a, bands=LEVEL_BANDS, pace_order=True))
    assert _order(ctx) == ["anthropic", "antigravity", "openai-codex"]
    assert ctx.decide(_task(complexity="M")).candidate.provider == "anthropic"

    ahead_c = headroom({"anthropic": (0.9, 0.05), "openai-codex": (0.9, 0.30), "antigravity": (0.9, 0.20)})
    ctx = kr.RoutingContext(conn, _cfg(ahead_c, bands=LEVEL_BANDS, pace_order=True))
    assert _order(ctx) == ["openai-codex", "antigravity", "anthropic"]
    assert ctx.decide(_task(complexity="M")).candidate.provider == "openai-codex"

    # Reserve counts: anthropic +0.30 - 0.40 = -0.10 drops below codex +0.05.
    ahead_a = headroom({"anthropic": (0.9, 0.30), "openai-codex": (0.9, 0.05), "antigravity": (0.9, 0.20)})
    ctx = kr.RoutingContext(conn, _cfg(ahead_a, bands=LEVEL_BANDS, pace_order=True, reserve={"anthropic": 0.40}))
    assert _order(ctx) == ["antigravity", "openai-codex", "anthropic"]

    # pace_order off: static order regardless of pace.
    ctx = kr.RoutingContext(conn, _cfg(ahead_a, bands=LEVEL_BANDS))
    assert _order(ctx) == ["anthropic", "openai-codex", "antigravity"]


def test_unknown_pace_falls_back(conn, headroom, monkeypatch):
    """Null pace: the gate uses level bands only and the walk sorts it as 0.0,
    with static order breaking the tie."""
    _pool(monkeypatch)
    path = headroom({"anthropic": (0.63, None), "openai-codex": (0.15, None), "antigravity": (0.9, -0.05)})
    ctx = kr.RoutingContext(conn, _cfg(path, reserve={"anthropic": 0.20}, pace_order=True))
    assert ctx.quota_verdict("anthropic", 0) is None  # 63% hits no level band; no pace data
    lv = ctx.quota_verdict("openai-codex", 0)
    assert lv is not None and lv.kind == "level" and lv.describe() == "openai-codex headroom 15% < 30% needs P2"
    # anthropic (unknown -> 0.0) and codex (unknown -> 0.0) tie, keep static order; antigravity -0.05 last.
    assert _order(ctx) == ["anthropic", "openai-codex", "antigravity"]
    assert ctx.pace_key("anthropic") == 0.0  # reserve does not apply to an unknown pace
    # Schema-1 file (no pace block at all) behaves the same.
    path.write_text(json.dumps({"schema": 1, "written_at_epoch": time.time(), "providers": {
        "anthropic": {"ok": True, "binding": {"headroom": 0.63}}}}))
    ctx = kr.RoutingContext(conn, _cfg(path, pace_order=True))
    assert ctx.quota().pace == {} and ctx.quota_verdict("anthropic", 0) is None


# --- scope of pace ordering ---------------------------------------------------------


def test_pace_order_keeps_tier_escalation_and_review_order(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    path = headroom({"anthropic": (0.9, -0.05), "openai-codex": (0.9, 0.30)})
    review = [{"model": "rev-a", "provider": "anthropic"}, {"model": "rev-c", "provider": "openai-codex"}]
    cfg = _cfg(path, bands=LEVEL_BANDS, pace_order=True, review=review, escalate=True)
    ctx = kr.RoutingContext(conn, cfg)
    entries = ctx._tier_entries(("M", "L"))
    assert [t for t, _ in entries] == ["M", "M", "M", "L", "L"]  # tiers never interleave
    assert [c.provider for t, c in entries if t == "L"] == ["openai-codex", "anthropic"]
    assert ctx.decide_review(_task()).candidate.model == "rev-a"  # review lane: static order
    pinned = ctx.decide(_task(model_override="claude-x", provider_override="anthropic"))
    assert pinned.source == "pinned"


def test_pace_order_keeps_payg_last_and_no_paid_fallthrough(conn, headroom, monkeypatch):
    """A payg provider (no quota, so unknown pace) never sorts ahead of a quota
    provider, even one far behind pace; the no-paid-fallthrough rule still holds."""
    _pool(monkeypatch)
    path = headroom({"anthropic": (0.9, -0.50)})
    tiers = {"M": [{"model": "or-m", "provider": "openrouter"}, {"model": "claude-m", "provider": "anthropic"},
                   {"model": "gem-m", "provider": "antigravity"}]}
    ctx = kr.RoutingContext(conn, _cfg(path, pace_order=True, tiers=tiers))
    assert _order(ctx) == ["antigravity", "anthropic", "openrouter"]
    # P0: anthropic gated by pace (-0.50 < -0.40 -> P3), antigravity (unknown) taken first anyway.
    assert ctx.decide(_task(complexity="M")).candidate.provider == "antigravity"
    tiers = {"M": [{"model": "or-m", "provider": "openrouter"}, {"model": "claude-m", "provider": "anthropic"}]}
    ctx = kr.RoutingContext(conn, _cfg(path, pace_order=True, tiers=tiers))
    assert _order(ctx) == ["anthropic", "openrouter"]
    d = ctx.decide(_task(complexity="M"))
    assert d.source == "quota_held" and d.skipped[-1]["reason"] == "quota: no paid fallthrough"


# --- CLI + dispatcher -------------------------------------------------------------


def _cli(monkeypatch, routing):
    from hermes_cli import kanban as kcli
    from hermes_cli.kanban_parser import build_parser

    monkeypatch.setattr(kcli, "_kanban_config", lambda: routing)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="cmd"))
    return lambda *argv: kcli.kanban_command(parser.parse_args(["kanban", *argv]))


def test_route_cli_prints_pace_reserve_and_order(kanban_home, headroom, monkeypatch, capsys):
    _pool(monkeypatch)
    path = headroom({"anthropic": (0.63, -0.19), "openai-codex": (0.52, 0.10), "antigravity": (0.9, None)})
    run = _cli(monkeypatch, _routing(path, reserve={"anthropic": 0.20}, pace_order=True))
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="low", assignee="w", complexity="M", priority=0)

    assert run("route", tid) == 0
    out = capsys.readouterr().out
    assert "pace_order=True" in out and "Tier M (pace order):" in out
    tier_m = out.split("Tier M (pace order):")[1].split("Tier L")[0]
    lines = [ln for ln in tier_m.splitlines() if ln.strip()]
    assert "openai-codex:gpt-m" in lines[0] and "pace +0.10" in lines[0]
    assert "antigravity:gem-m" in lines[1] and "pace +0.00" in lines[1]
    assert "anthropic:claude-m" in lines[2] and "pace -0.39" in lines[2]
    assert "reserve: anthropic 0.20" in out
    assert "pace -0.19 - reserve 0.20 = -0.39 (< -0.25) -> needs P2" in out
    assert "antigravity: headroom 90%" in out and "pace unknown -> level bands only" in out
    assert "-> tier M -> openai-codex:gpt-m" in out

    assert run("route", tid, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["pace_order"] is True
    by = {p["provider"]: p for p in data["quota_gate"]["providers"]}
    assert by["anthropic"]["pace_required_priority"] == 2 and by["anthropic"]["required_priority"] == 2
    assert by["anthropic"]["pace_effective"] == pytest.approx(-0.39)
    assert [c["provider"] for c in data["tiers"][1]["candidates"]] == ["openai-codex", "antigravity", "anthropic"]


def test_dispatch_spawns_on_pace_ordered_candidate(conn, headroom, monkeypatch, all_assignees_spawnable):
    path = headroom({"anthropic": (0.63, -0.19), "openai-codex": (0.52, 0.10)})
    cfg = _cfg(path, reserve={"anthropic": 0.20}, pace_order=True)
    monkeypatch.setattr(kr, "load_routing_config", lambda kanban_cfg=None: cfg)
    _pool(monkeypatch)
    seen: list = []

    def spawn(task, workspace, board=None):
        seen.append((task.id, task.model_override, task.provider_override))
        return 4242

    low = kb.create_task(conn, title="low", assignee="w", complexity="L", priority=0)
    kbd.dispatch_once(conn, spawn_fn=spawn)
    assert seen == [(low, "gpt-l", "openai-codex")]
