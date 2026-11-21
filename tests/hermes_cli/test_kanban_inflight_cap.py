"""Per-provider in-flight cap and Anthropic session floor (intake t_ca1570d0).

On 2026-11-21 one dispatcher tick started 17 Anthropic runs (13:50) and another 7
(08:58); each wave used up the 5 h session in about 20 minutes and locked MB's chats
out until the reset. ``quota_gate.max_in_flight`` caps the runs routed to a provider
(running + this tick's spawns); ``quota_gate.session_floor`` holds non-P3 Anthropic
cards while a fresh snapshot shows the session below the floor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_routing as kr

_LOAD = kr.load_routing_config  # real parser, immune to monkeypatching
WEEK_RESET = "2026-11-27T05:00:00+00:00"
SESSION_RESET = "2099-01-01T18:30:00+00:00"

TIERS = {
    "M": [{"model": "claude-opus-5-5", "provider": "anthropic"}],
    "XL": [{"model": "claude-fable-5-2", "provider": "anthropic"}],
    "S": [{"model": "grok-s", "provider": "xai-oauth"}],
}
REVIEW = [{"model": "claude-opus-5-5", "provider": "anthropic"}, {"model": "grok-r", "provider": "xai-oauth"}]


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


@pytest.fixture(autouse=True)
def pool_ok(monkeypatch):
    monkeypatch.setattr(kr, "pool_availability", lambda provider, model, *, now=None: kr.Availability(True, "pool ok"))


@pytest.fixture
def files(tmp_path):
    """``write(session_used=..., age=..., mode=...)``: headroom.json with Anthropic's week
    binding (73% left) and, unless ``session_used`` is None, its session window; plus a
    fresh mode.json (``normal`` by default)."""
    headroom = tmp_path / "headroom.json"
    mode_file = tmp_path / "mode.json"

    def write(session_used: float | None = None, *, age: float = 60.0, mode: str = "normal"):
        windows = [{"name": "week", "headroom": 0.73, "resets_at": WEEK_RESET}]
        if session_used is not None:
            windows.append({"name": "session", "headroom": round(1 - session_used, 4), "resets_at": SESSION_RESET})
        headroom.write_text(json.dumps({"schema": 2, "written_at_epoch": time.time() - age, "providers": {
            "anthropic": {"ok": True, "binding": {"name": "week", "headroom": 0.73, "resets_at": WEEK_RESET},
                          "windows": windows},
        }}))
        mode_file.write_text(json.dumps({"mode": mode, "written_at_epoch": time.time() - 60, "reason": "test",
                                         "since": "2026-11-21T00:00:00+00:00", "resets_at": WEEK_RESET}))
        return write

    write.headroom = headroom
    write.mode_file = mode_file
    write()
    return write


def _cfg(files, **gate_over) -> kr.RoutingConfig:
    gate = {"enabled": True, "headroom_file": str(files.headroom), "mode_file": str(files.mode_file), **gate_over}
    return _LOAD({"routing": {"enabled": True, "tiers": TIERS, "review": REVIEW, "escalate": False,
                              "quota_gate": gate}})


def _use(monkeypatch, files, **gate_over) -> kr.RoutingConfig:
    cfg = _cfg(files, **gate_over)
    monkeypatch.setattr(kr, "load_routing_config", lambda kanban_cfg=None: cfg)
    return cfg


def _spawns():
    seen: list = []

    def fake_spawn(task, workspace, board=None):
        seen.append((task.id, task.model_override, task.provider_override))
        return 4242
    return seen, fake_spawn


def _ready(conn, n, complexity="XL", **kw):
    return [kb.create_task(conn, title=f"c{i}", assignee="w", complexity=complexity, **kw) for i in range(n)]


def _tick(conn, spawn):
    return kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=20, max_in_progress=20)


def _task(**kw):
    kw.setdefault("priority", 0)
    return kb.Task(id="t_x", title="x", body=None, assignee="w", status="ready",
                   created_by=None, created_at=0, started_at=None, completed_at=None,
                   workspace_kind="scratch", workspace_path=None, claim_lock=None,
                   claim_expires=None, tenant=None, **kw)


# --- config ------------------------------------------------------------------------------------------


def test_defaults_cap_anthropic_at_4_and_floor_30pct():
    gate = kr.load_routing_config({}).quota_gate
    assert gate.in_flight_cap("anthropic") == 4 and gate.in_flight_cap("xai-oauth") == 0
    assert gate.session_floor == 0.30


@pytest.mark.parametrize("raw, cap", [({}, 0), (0, 0), (None, 0), ({"anthropic": 0}, 0), ({"Anthropic": "2"}, 2),
                                      ({"anthropic": "x"}, 0)])
def test_max_in_flight_parse(raw, cap):
    gate = kr.load_routing_config({"routing": {"quota_gate": {"max_in_flight": raw}}}).quota_gate
    assert gate.in_flight_cap("anthropic") == cap


# --- in-flight cap: dispatcher replays ---------------------------------------------------------------


def test_1350_wave_of_17_spawns_4_rest_held_without_failure(conn, files, monkeypatch, all_assignees_spawnable):
    """(a) 13:50: 17 ready Anthropic cards, nothing running -> 4 spawn (old code: 17)."""
    _use(monkeypatch, files)
    seen, spawn = _spawns()
    ids = _ready(conn, 10, "M") + _ready(conn, 7, "XL")
    res = _tick(conn, spawn)
    assert len(seen) == 4 and {s[2] for s in seen} == {"anthropic"}
    held = [tid for tid, _note, _retry in res.quota_held]
    assert len(held) == 13 and set(held) | {s[0] for s in seen} == set(ids)
    assert all(retry is None for _tid, _note, retry in res.quota_held)
    assert "in-flight cap anthropic 4/4" in res.quota_held[0][1]
    for tid in held:
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.consecutive_failures == 0
    # Next tick: the 4 still run, so nothing more starts.
    assert len(_tick(conn, spawn).quota_held) == 13 and len(seen) == 4


def test_0858_wave_of_7_xl_spawns_4(conn, files, monkeypatch, all_assignees_spawnable):
    """(b) 08:58: 7 ready XL cards -> 4."""
    _use(monkeypatch, files)
    seen, spawn = _spawns()
    _ready(conn, 7)
    _tick(conn, spawn)
    assert len(seen) == 4


def test_running_anthropic_runs_count_against_the_cap(conn, files, monkeypatch, all_assignees_spawnable):
    """(c) 3 running Anthropic + 5 ready -> 1; a run that ends frees its slot."""
    _use(monkeypatch, files)
    seen, spawn = _spawns()
    first = _ready(conn, 3)
    _use(monkeypatch, files, max_in_flight={"anthropic": 3})
    _tick(conn, spawn)
    assert [s[0] for s in seen] == first
    _ready(conn, 5)
    _use(monkeypatch, files)
    _tick(conn, spawn)
    assert len(seen) == 4
    kb.complete_task(conn, first[0], summary="done")
    _tick(conn, spawn)
    assert len(seen) == 5


def test_review_lane_at_cap_routes_to_next_provider(conn, files, monkeypatch, all_assignees_spawnable):
    """(d) review candidates [opus anthropic, grok xai-oauth] with Anthropic at the cap -> grok."""
    _use(monkeypatch, files)
    seen, spawn = _spawns()
    _ready(conn, 4)
    _tick(conn, spawn)
    rev = kb.create_task(conn, title="r", assignee="w", priority=0)
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (rev,))
    conn.commit()
    _tick(conn, spawn)
    assert seen[-1] == (rev, "grok-r", "xai-oauth")


def test_cap_off_spawns_the_whole_wave(conn, files, monkeypatch, all_assignees_spawnable):
    """(e) ``max_in_flight`` off (``{}`` passed to the parser, i.e. no default merge) -> 17."""
    _use(monkeypatch, files, max_in_flight={})
    seen, spawn = _spawns()
    _ready(conn, 17)
    _tick(conn, spawn)
    assert len(seen) == 17


def test_pinned_card_at_cap_is_held_not_rerouted(conn, files, monkeypatch, all_assignees_spawnable):
    """(f) a card pinned to an Anthropic model waits at the cap; it is never routed elsewhere."""
    _use(monkeypatch, files)
    seen, spawn = _spawns()
    _ready(conn, 4)
    _tick(conn, spawn)
    pinned = kb.create_task(conn, title="pin", assignee="w", model_override="claude-opus-5-5",
                            provider_override="anthropic")
    res = _tick(conn, spawn)
    assert (pinned, "quota_hold") in res.respawn_guarded and len(seen) == 4
    assert kb.get_task(conn, pinned).consecutive_failures == 0


def test_dry_run_counts_its_own_spawns(conn, files, monkeypatch, all_assignees_spawnable):
    _use(monkeypatch, files)
    _ready(conn, 6)
    res = kbd.dispatch_once(conn, dry_run=True, max_spawn=20, max_in_progress=20)
    assert len(res.spawned) == 4


# --- session floor -----------------------------------------------------------------------------------


@pytest.mark.parametrize("used", [0.47, 0.58, 0.60])
def test_session_with_headroom_passes_p0(conn, files, used):
    files(session_used=used)
    assert kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 0) is None


@pytest.mark.parametrize("used", [0.701, 1.0])
def test_session_below_floor_holds_p0_p2_passes_p3(conn, files, used):
    files(session_used=used)
    ctx = kr.RoutingContext(conn, _cfg(files))
    for prio in (0, 2):
        v = ctx.quota_verdict("anthropic", prio)
        assert v is not None and v.kind == "session" and v.required == 3
        assert v.resets_at == kr._parse_iso_epoch(SESSION_RESET)
        assert "< 30% floor" in v.describe() and "needs P3" in v.describe()
    assert ctx.quota_verdict("anthropic", 3) is None


def test_no_session_window_or_stale_snapshot_leaves_floor_inert(conn, files):
    files(session_used=None)
    assert kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 0) is None
    files(session_used=1.0, age=7200)
    assert kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 0) is None


def test_session_floor_raises_tight_mode_requirement(conn, files):
    files(session_used=0.80, mode="tight")
    v = kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 2)
    assert v is not None and v.kind == "session" and v.required == 3
    files(session_used=0.10, mode="tight")
    v = kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 0)
    assert v is not None and v.kind == "mode" and v.required == 2


@pytest.mark.parametrize("mode", ["unknown", None])
def test_floor_applies_without_a_fresh_mode(conn, files, mode):
    """Mode ``unknown`` or no mode file falls back to the level gate (week 73%: open); a
    fresh session window below the floor still holds non-P3 cards (t_16970ff4#2)."""
    files(session_used=0.90, mode=mode or "normal")
    if mode is None:
        files.mode_file.unlink()
    v = kr.RoutingContext(conn, _cfg(files)).quota_verdict("anthropic", 1)
    assert v is not None and v.kind == "session"


def test_session_window_never_feeds_the_level_gate(conn, files, monkeypatch):
    """No fresh mode: a near-empty session must not trip the level gate (floor only)."""
    files(session_used=0.99)
    files.mode_file.unlink()
    ctx = kr.RoutingContext(conn, _cfg(files, session_floor=0))
    assert ctx.quota_verdict("anthropic", 0) is None


def test_session_floor_holds_dispatch_with_session_reset_retry(conn, files, monkeypatch, all_assignees_spawnable):
    files(session_used=1.0)
    _use(monkeypatch, files)
    seen, spawn = _spawns()
    low = kb.create_task(conn, title="low", assignee="w", complexity="M", priority=1)
    urgent = kb.create_task(conn, title="urgent", assignee="w", complexity="M", priority=3)
    res = _tick(conn, spawn)
    assert [s[0] for s in seen] == [urgent]
    (tid, note, retry), = res.quota_held
    assert tid == low and "session 0% left" in note and retry == int(kr._parse_iso_epoch(SESSION_RESET))
