"""Anthropic quota mode in Kanban routing (``data/quota/mode.json``, redesign slice 2).

A fresh mode file is the whole gate for ``anthropic`` (one ``min_priority``
instead of bands/pace/reserve); ``open`` puts Opus first and wakes quota parks.
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
RESET = "2026-10-17T04:00:00+00:00"
DAY = 86400


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
def files(tmp_path):
    headroom = tmp_path / "headroom.json"
    mode_file = tmp_path / "mode.json"
    # Bands that would hold every P0-P2 anthropic card: the mode must replace them.
    headroom.write_text(json.dumps({"schema": 1, "written_at_epoch": time.time() - 60, "providers": {
        "anthropic": {"ok": True, "binding": {"name": "week", "headroom": 0.05, "resets_at": RESET}, "windows": []},
        "openai-codex": {"ok": True, "binding": {"name": "week", "headroom": 0.9, "resets_at": None}, "windows": []},
    }}))

    def write(mode, *, age=60.0, resets_at=RESET):
        mode_file.write_text(json.dumps({"mode": mode, "written_at_epoch": time.time() - age, "reason": f"test {mode}",
                                         "since": "2026-10-16T00:00:00+00:00", "resets_at": resets_at}))
        return mode_file

    write.headroom = headroom
    write.mode_file = mode_file
    return write


TIERS = {
    "S": [{"model": "gpt-s", "provider": "openai-codex"}, {"model": "claude-sonnet-s", "provider": "anthropic"},
          {"model": "claude-opus-s", "provider": "anthropic"}],
    "L": [{"model": "claude-opus-l", "provider": "anthropic"}],
}
REVIEW = [{"model": "gpt-r", "provider": "openai-codex"}, {"model": "claude-opus-r", "provider": "anthropic"}]
BANDS = {"anthropic": [{"below": 0.10, "min_priority": 3}], "default": []}


def _cfg(files, *, enabled=True, **over) -> kr.RoutingConfig:
    gate = {"enabled": enabled, "headroom_file": str(files.headroom), "mode_file": str(files.mode_file),
            "bands": BANDS}
    return _LOAD({"routing": {"enabled": True, "tiers": TIERS, "review": REVIEW, "escalate": False,
                              "quota_gate": gate, **over}})


def _use(monkeypatch, files, **kw) -> kr.RoutingConfig:
    cfg = _cfg(files, **kw)
    monkeypatch.setattr(kr, "load_routing_config", lambda kanban_cfg=None: cfg)
    return cfg


@pytest.fixture(autouse=True)
def pool_ok(monkeypatch):
    monkeypatch.setattr(kr, "pool_availability", lambda provider, model, *, now=None: kr.Availability(True, "pool ok"))


def _task(**kw):
    kw.setdefault("priority", 0)
    return kb.Task(id="t_x", title="x", body=None, assignee="w", status="ready",
                   created_by=None, created_at=0, started_at=None, completed_at=None,
                   workspace_kind="scratch", workspace_path=None, claim_lock=None,
                   claim_expires=None, tenant=None, **kw)


# --- reading the file ----------------------------------------------------------------------------------


def test_read_mode_statuses(files, tmp_path):
    gate = _cfg(files).quota_gate
    assert kr.read_mode(gate).status == "missing"
    files("tight")
    m = kr.read_mode(gate)
    assert m.active and m.min_priority == 2 and m.resets_at and not m.spend_down
    files("tight", age=3600)
    assert kr.read_mode(gate).status == "stale" and not kr.read_mode(gate).active
    files("unknown")
    assert kr.read_mode(gate).status == "ok" and not kr.read_mode(gate).active
    files.mode_file.write_text("{nope")
    assert kr.read_mode(gate).status == "unreadable"
    assert kr.load_routing_config({}).quota_gate.mode_file == ""


@pytest.mark.parametrize("mode,need", [("closed", 4), ("critical", 3), ("tight", 2), ("normal", 0), ("open", 0)])
def test_mode_min_priority(files, conn, mode, need):
    files(mode)
    ctx = kr.RoutingContext(conn, _cfg(files))
    for prio in range(4):
        verdict = ctx.quota_verdict("anthropic", prio)
        assert (verdict is None) == (prio >= need)
        if verdict is not None:
            assert verdict.kind == "mode" and f"quota mode {mode} needs P{need}" in verdict.describe()


# --- gate --------------------------------------------------------------------------------------------


def test_fresh_mode_replaces_anthropic_bands(files, conn):
    files("normal")  # bands alone would need P3 at 5% headroom
    assert kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 0) is None


@pytest.mark.parametrize("state", ["missing", "stale", "unknown"])
def test_no_fresh_mode_falls_back_to_bands(files, conn, state):
    if state == "stale":
        files("normal", age=3600)
    elif state == "unknown":
        files("unknown")
    verdict = kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 0)
    assert verdict is not None and verdict.kind == "level" and verdict.required == 3


def test_gate_disabled_ignores_mode(files, conn):
    files("closed")
    ctx = kr.RoutingContext(conn, _cfg(files, enabled=False))
    assert ctx.quota_verdict("anthropic", 0) is None and not ctx.spend_down()


def test_mode_is_anthropic_only(files, conn):
    files("closed")
    assert kr.RoutingContext(conn, _cfg(files)).quota_verdict("openai-codex", 0) is None


def test_tight_routes_low_priority_off_anthropic(files, conn):
    files("tight")
    ctx = kr.RoutingContext(conn, _cfg(files))
    d = ctx.decide(_task(complexity="S", priority=1))
    assert d.source == "tier" and d.candidate.provider == "openai-codex"
    assert ctx.decide(_task(complexity="L", priority=1)).source == "quota_held"
    assert ctx.decide(_task(complexity="L", priority=2)).candidate.model == "claude-opus-l"


def test_closed_holds_pinned_opus_until_reset(files, conn):
    files("closed")
    d = kr.RoutingContext(conn, _cfg(files)).decide(
        _task(priority=3, model_override="claude-opus-l", provider_override="anthropic"))
    assert d.source == "quota_held" and d.retry_at == kr._parse_iso_epoch(RESET)
    assert "quota mode closed needs P4" in d.note


# --- spend-down ----------------------------------------------------------------------------------------


def test_open_puts_opus_first_in_tiers_and_review(files, conn):
    files("open")
    ctx = kr.RoutingContext(conn, _cfg(files))
    assert ctx.spend_down()
    assert [c.model for c in ctx.order_tier(_cfg(files).tiers["S"])] == ["claude-opus-s", "claude-sonnet-s", "gpt-s"]
    assert ctx.decide(_task(complexity="S", priority=0)).candidate.model == "claude-opus-s"
    assert ctx.decide_review(_task(priority=0)).candidate.model == "claude-opus-r"


@pytest.mark.parametrize("mode", ["normal", "tight"])
def test_static_order_outside_open(files, conn, mode):
    files(mode)
    ctx = kr.RoutingContext(conn, _cfg(files))
    assert ctx.decide(_task(complexity="S", priority=3)).candidate.model == "gpt-s"
    assert ctx.decide_review(_task(priority=3)).candidate.model == "gpt-r"


def _park(conn, title, reason, until_ts, then="ask"):
    tid = kb.create_task(conn, title=title, assignee="w", complexity="L", priority=0)
    assert kb.schedule_task(conn, tid, reason=reason, until=int(until_ts), then=then)
    return tid


def test_open_wakes_quota_parks_only(files, conn, monkeypatch, all_assignees_spawnable):
    _use(monkeypatch, files)
    reset = time.time() + 2 * DAY
    files("open", resets_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(reset)))
    quota = _park(conn, "q", "until reset: anthropic quota", reset - 3600)
    near = _park(conn, "q2", "waiting for Quota to reset", reset + 3600, then="start")
    other = _park(conn, "o", "waiting on MB", reset - 3600)
    later = _park(conn, "l", "quota re-check next week", reset + 5 * DAY)
    seen: list = []
    res = kbd.dispatch_once(conn, spawn_fn=lambda t, w, board=None: seen.append((t.id, t.model_override)) or 4242)
    assert set(dict(res.woke_scheduled)) == {quota, near}
    for tid in (other, later):
        assert kb.get_task(conn, tid).status == "scheduled"
    woke = [e for e in kb.list_events(conn, quota) if e.kind == "unblocked"]
    assert woke and woke[-1].payload["woke_by"] == "quota mode open"
    assert {tid for tid, _ in seen} == {quota, near}
    assert all(model == "claude-opus-l" for _, model in seen)


@pytest.mark.parametrize("mode", ["normal", "critical"])
def test_no_wake_outside_open(files, conn, monkeypatch, mode, all_assignees_spawnable):
    _use(monkeypatch, files)
    reset = time.time() + 2 * DAY
    files(mode, resets_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(reset)))
    tid = _park(conn, "q", "anthropic quota", reset - 3600)
    res = kbd.dispatch_once(conn, spawn_fn=lambda t, w, board=None: 4242)
    assert tid not in dict(res.woke_scheduled) and kb.get_task(conn, tid).status == "scheduled"


# --- route printer -------------------------------------------------------------------------------------


def test_route_cli_prints_mode(files, kanban_home, monkeypatch, capsys):
    from hermes_cli import kanban as kcli
    from hermes_cli.kanban_parser import build_parser

    files("tight")
    for enabled in (True, False):
        cfg = {"routing": {"enabled": True, "tiers": TIERS, "escalate": False, "quota_gate": {
            "enabled": enabled, "headroom_file": str(files.headroom), "mode_file": str(files.mode_file),
            "bands": BANDS}}}
        monkeypatch.setattr(kcli, "_kanban_config", lambda cfg=cfg: cfg)
        with kbc.connect_closing() as c:
            tid = kb.create_task(c, title="low", assignee="w", complexity="L", priority=0)
        parser = argparse.ArgumentParser()
        build_parser(parser.add_subparsers(dest="cmd"))
        assert kcli.kanban_command(parser.parse_args(["kanban", "route", tid])) == 0
        out = capsys.readouterr().out
        assert "Quota mode: anthropic tight: cards need P2" in out
        if enabled:
            assert "anthropic: headroom 5% -> quota mode tight needs P2" in out
            assert "-> quota held" in out
        else:
            assert "[advisory: gate disabled]" in out
    assert kcli.kanban_command(parser.parse_args(["kanban", "route", tid, "--json"])) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["quota_gate"]["mode"]["mode"] == "tight" and data["quota_gate"]["mode"]["min_priority"] == 2
