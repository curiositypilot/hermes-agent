"""Quota gate in Kanban tier routing (``kanban.routing.quota_gate``).

A provider whose live headroom (the quota-snapshot file) is below a band only
takes cards whose priority meets the band; others route on or wait as
``quota_hold`` with no failure counted.
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

_LOAD = kr.load_routing_config  # real parser, immune to monkeypatching


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
    path = tmp_path / "headroom.json"

    def write(providers: dict, *, age: float = 60.0) -> Path:
        out = {}
        for name, spec in providers.items():
            if spec is None:
                out[name] = {"ok": False, "binding": None, "windows": []}
            else:
                h, resets = spec if isinstance(spec, tuple) else (spec, None)
                out[name] = {"ok": True, "binding": {"name": "session", "headroom": h, "resets_at": resets},
                             "windows": []}
        path.write_text(json.dumps({"schema": 1, "written_at_epoch": time.time() - age, "providers": out}))
        return path

    return write


TIERS = {
    "S": [{"model": "claude-s", "provider": "anthropic"}, {"model": "gpt-s", "provider": "openai-codex"}],
    "M": [{"model": "claude-m", "provider": "anthropic"}, {"model": "or-m", "provider": "openrouter"}],
    "L": [{"model": "claude-l", "provider": "anthropic"}],
}
BANDS = {
    "anthropic": [{"below": 0.35, "min_priority": 1}, {"below": 0.20, "min_priority": 2},
                  {"below": 0.08, "min_priority": 3}],
    "default": [{"below": 0.25, "min_priority": 1}, {"below": 0.10, "min_priority": 2},
                {"below": 0.03, "min_priority": 3}],
}


def _cfg(path, **over) -> kr.RoutingConfig:
    gate = {"enabled": True, "headroom_file": str(path), "bands": BANDS, **over.pop("gate", {})}
    return _LOAD({"routing": {"enabled": True, "tiers": TIERS, "escalate": False,
                              "quota_gate": gate, **over}})


def _use(monkeypatch, path, **over) -> kr.RoutingConfig:
    cfg = _cfg(path, **over)
    monkeypatch.setattr(kr, "load_routing_config", lambda kanban_cfg=None: cfg)
    return cfg


def _pool(monkeypatch, blocked=()):
    def fake(provider, model, *, now=None):
        if provider in blocked:
            return kr.Availability(False, "credential pool exhausted", until=time.time() + 600)
        return kr.Availability(True, "pool ok")
    monkeypatch.setattr(kr, "pool_availability", fake)


def _task(**kw):
    kw.setdefault("priority", 0)
    return kb.Task(id="t_x", title="x", body=None, assignee="w", status="ready",
                   created_by=None, created_at=0, started_at=None, completed_at=None,
                   workspace_kind="scratch", workspace_path=None, claim_lock=None,
                   claim_expires=None, tenant=None, **kw)


def _spawns():
    seen: list = []

    def fake_spawn(task, workspace, board=None):
        seen.append((task.id, task.model_override, task.provider_override))
        return 4242
    return seen, fake_spawn


# --- config + band math -------------------------------------------------------


def test_config_defaults_and_parse():
    off = kr.load_routing_config({})
    assert off.quota_gate.enabled is False
    assert off.quota_gate.payg_providers == ("openrouter",)
    assert off.quota_gate.max_age_seconds == 1800
    assert "default" in off.quota_gate.bands

    cfg = _cfg("/x.json", gate={"max_age_seconds": "60", "payg_providers": ["OpenRouter"]})
    gate = cfg.quota_gate
    assert gate.enabled and gate.headroom_file == "/x.json" and gate.max_age_seconds == 60
    assert gate.payg_providers == ("openrouter",)
    assert gate.bands_for("anthropic")[0] == (0.35, 1)
    assert gate.bands_for("antigravity") == gate.bands["default"]


def test_config_defaults_match_config_defaults_module():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    raw = DEFAULT_CONFIG["kanban"]["routing"]["quota_gate"]
    parsed = kr.load_routing_config({"routing": {"quota_gate": raw}}).quota_gate
    assert parsed == kr.QuotaGateConfig()


def test_band_math():
    bands = kr.load_routing_config({"routing": {"quota_gate": {"bands": BANDS}}}).quota_gate.bands_for("anthropic")
    assert kr.required_priority(bands, 0.90) == (0, None)
    assert kr.required_priority(bands, 0.35) == (0, None)  # strict "below"
    assert kr.required_priority(bands, 0.30) == (1, 0.35)
    assert kr.required_priority(bands, 0.12) == (2, 0.20)
    assert kr.required_priority(bands, 0.05) == (3, 0.08)
    # max min_priority wins even when bands are listed out of order
    assert kr.required_priority(((0.08, 3), (0.35, 1)), 0.05) == (3, 0.08)


@pytest.mark.parametrize("case", ["stale", "ok_false", "missing_provider", "no_file", "garbage", "disabled"])
def test_unknown_headroom_leaves_gate_open(conn, headroom, tmp_path, case):
    path = headroom({"anthropic": 0.01})
    over = {}
    if case == "stale":
        path = headroom({"anthropic": 0.01}, age=7200)
    elif case == "ok_false":
        path = headroom({"anthropic": None})
    elif case == "missing_provider":
        path = headroom({"openai-codex": 0.01})
    elif case == "no_file":
        path = tmp_path / "nope.json"
    elif case == "garbage":
        path.write_text("{not json")
    elif case == "disabled":
        over = {"gate": {"enabled": False}}
    ctx = kr.RoutingContext(conn, _cfg(path, **over))
    assert ctx.quota_verdict("anthropic", 0) is None


def test_headroom_read_once_per_context(conn, headroom, monkeypatch):
    path = headroom({"anthropic": 0.5})
    calls = []
    real = kr.read_headroom
    monkeypatch.setattr(kr, "read_headroom", lambda gate, now=None: calls.append(1) or real(gate, now=now))
    ctx = kr.RoutingContext(conn, _cfg(path))
    for _ in range(3):
        ctx.quota_verdict("anthropic", 0)
    assert len(calls) == 1


# --- decisions ------------------------------------------------------------------


def test_p0_skips_gated_anthropic_and_routes_to_codex(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.12, "openai-codex": 0.9})))
    d = ctx.decide(_task(complexity="S", priority=0))
    assert d.source == "tier" and d.candidate.provider == "openai-codex"
    assert d.skipped[0]["reason"] == "quota: anthropic headroom 12% < 20% needs P2"


def test_p2_still_gets_anthropic(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.12, "openai-codex": 0.9})))
    d = ctx.decide(_task(complexity="S", priority=2))
    assert d.source == "tier" and d.candidate.provider == "anthropic" and d.skipped == []


def test_all_gated_is_quota_held_with_retry_and_note(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    reset_a, reset_c = "2026-10-03T14:00:00+00:00", "2026-10-03T12:00:00Z"
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": (0.12, reset_a), "openai-codex": (0.05, reset_c)})))
    d = ctx.decide(_task(complexity="S", priority=1))
    assert d.source == "quota_held"
    assert d.retry_at == kr._parse_iso_epoch(reset_c)
    assert "anthropic headroom 12% < 20% needs P2" in d.note and "(card P1)" in d.note
    assert d.label().startswith("quota held")


def test_payg_not_used_after_gate_skip(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.12})))
    d = ctx.decide(_task(complexity="M", priority=0))
    assert d.source == "quota_held"
    assert d.skipped[-1] == {"tier": "M", "model": "or-m", "provider": "openrouter",
                             "reason": "quota: no paid fallthrough"}


def test_payg_used_after_pure_rate_limit_skip(conn, headroom, monkeypatch):
    _pool(monkeypatch, blocked={"anthropic"})
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.9})))
    d = ctx.decide(_task(complexity="M", priority=0))
    assert d.source == "tier" and d.candidate.provider == "openrouter"


def test_on_exhausted_profile_does_not_bypass_quota_hold(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.12}), on_exhausted="profile"))
    assert ctx.decide(_task(complexity="L", priority=0)).source == "quota_held"
    # Without a gate skip, on_exhausted: profile still applies.
    _pool(monkeypatch, blocked={"anthropic"})
    ctx2 = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.9}), on_exhausted="profile"))
    assert ctx2.decide(_task(complexity="L", priority=0)).source == "profile"


def test_pinned_and_profile_routes_gated_on_own_provider(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    path = headroom({"anthropic": 0.05, "openai-codex": 0.9})
    ctx = kr.RoutingContext(conn, _cfg(path))
    pinned = ctx.decide(_task(model_override="claude-x", provider_override="anthropic", priority=2))
    assert pinned.source == "quota_held" and pinned.candidate.model == "claude-x"
    assert ctx.decide(_task(model_override="claude-x", provider_override="anthropic", priority=3)).source == "pinned"
    assert ctx.decide(_task(model_override="gpt", provider_override="openai-codex")).source == "pinned"

    monkeypatch.setattr(kr, "_profile_model", lambda home: kr.TierCandidate("claude-p", "anthropic"))
    unl = ctx.decide(_task(priority=0))
    assert unl.source == "quota_held"
    monkeypatch.setattr(kr, "_profile_model", lambda home: None)
    assert ctx.decide(_task(priority=0)).source == "profile"  # unknown provider = open


def test_review_lane_gated(conn, headroom, monkeypatch):
    _pool(monkeypatch)
    review = [{"model": "rev", "provider": "anthropic"}, {"model": "or-rev", "provider": "openrouter"}]
    ctx = kr.RoutingContext(conn, _cfg(headroom({"anthropic": 0.12}), review=review))
    d = ctx.decide_review(_task(priority=0))
    assert d.source == "quota_held" and d.lane == "review"
    assert d.skipped[-1]["reason"] == "quota: no paid fallthrough"
    assert ctx.decide_review(_task(priority=2)).candidate.model == "rev"


# --- dispatcher -----------------------------------------------------------------


def test_dispatch_quota_hold_one_event_no_failure(conn, headroom, monkeypatch, all_assignees_spawnable):
    _use(monkeypatch, headroom({"anthropic": (0.12, "2026-10-03T14:00:00+00:00"), "openai-codex": 0.05}))
    _pool(monkeypatch)
    seen, spawn = _spawns()
    held = kb.create_task(conn, title="low", assignee="w", complexity="S", priority=0)
    urgent = kb.create_task(conn, title="urgent", assignee="w", complexity="S", priority=3)

    for _ in range(2):
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert (held, "quota_hold") in res.respawn_guarded

    assert seen == [(urgent, "claude-s", "anthropic")]
    task = kb.get_task(conn, held)
    assert task.status == "ready" and task.consecutive_failures == 0
    holds = [e for e in kb.list_events(conn, held) if e.kind == "routing_held"]
    assert len(holds) == 1
    assert holds[0].payload["reason"] == "quota_hold" and holds[0].payload["retry_at"]
    assert "needs P2" in holds[0].payload["note"]
    assert "quota_hold=1" in kbd.describe_suppression([res])


def test_held_card_does_not_consume_spawn_slot(conn, headroom, monkeypatch, all_assignees_spawnable):
    _use(monkeypatch, headroom({"anthropic": 0.12, "openai-codex": 0.9}),
         tiers={"L": [{"model": "claude-l", "provider": "anthropic"}],
                "S": [{"model": "gpt-s", "provider": "openai-codex"}]})
    _pool(monkeypatch)
    seen, spawn = _spawns()
    kb.create_task(conn, title="held", assignee="w", complexity="L", priority=1)
    later = kb.create_task(conn, title="later", assignee="w", complexity="S", priority=0)
    kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
    assert [s[0] for s in seen] == [later]


def test_quota_held_review_does_not_reserve_slot(conn, headroom, monkeypatch, all_assignees_spawnable):
    _use(monkeypatch, headroom({"anthropic": 0.12, "openai-codex": 0.9}),
         review=[{"model": "rev", "provider": "anthropic"}])
    _pool(monkeypatch)
    seen, spawn = _spawns()
    rev = kb.create_task(conn, title="r", assignee="w", priority=0)
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (rev,))
    conn.commit()
    ready = kb.create_task(conn, title="s", assignee="w", complexity="S", priority=0)
    res = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
    assert [s[0] for s in seen] == [ready]
    res = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert (rev, "quota_hold") in res.respawn_guarded
    assert [s[0] for s in seen] == [ready]


def test_dry_run_prints_held_line(kanban_home, headroom, monkeypatch, capsys, all_assignees_spawnable):
    from hermes_cli import kanban as kcli
    from hermes_cli.kanban_parser import build_parser

    _use(monkeypatch, headroom({"anthropic": (0.12, "2026-10-03T14:00:00+00:00"), "openai-codex": 0.05}))
    _pool(monkeypatch)
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="low", assignee="w", complexity="S", priority=0)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="cmd"))

    def run(*argv):
        return kcli.kanban_command(parser.parse_args(["kanban", *argv]))

    assert run("dispatch", "--dry-run") == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("Held (quota_hold):"))
    assert tid in line and "anthropic headroom 12% < 20% needs P2" in line and "(card P0)" in line
    assert "retry ~" in line
    assert f"Guarded (quota_hold): {tid}" not in out

    assert run("dispatch", "--dry-run", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["quota_held"][0]["task_id"] == tid and data["quota_held"][0]["retry_at"]


def test_route_cli_shows_quota_gate(kanban_home, headroom, monkeypatch, capsys):
    from hermes_cli import kanban as kcli
    from hermes_cli.kanban_parser import build_parser

    path = headroom({"anthropic": 0.12, "openai-codex": 0.9, "antigravity": None})
    monkeypatch.setattr(kcli, "_kanban_config", lambda: {"routing": {
        "enabled": True, "tiers": TIERS, "escalate": False,
        "quota_gate": {"enabled": True, "headroom_file": str(path), "bands": BANDS}}})
    _pool(monkeypatch)
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="low", assignee="w", complexity="M", priority=0)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="cmd"))

    def run(*argv):
        return kcli.kanban_command(parser.parse_args(["kanban", *argv]))

    assert run("route", tid) == 0
    out = capsys.readouterr().out
    assert "Quota gate:" in out and "anthropic: headroom 12% (< 20%) -> needs P2" in out
    assert "antigravity: unknown" in out
    assert "-> quota held" in out and "no paid fallthrough" in out

    assert run("route", tid, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["task"]["source"] == "quota_held"
    by = {p["provider"]: p for p in data["quota_gate"]["providers"]}
    assert by["anthropic"]["required_priority"] == 2 and by["openai-codex"]["required_priority"] == 0
