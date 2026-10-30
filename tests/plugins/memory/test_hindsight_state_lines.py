"""recall_state_lines: latest-status lines from the entity-state index (intake t_5958cf5b slice B).

The index here is synthetic (slice A's DDL, invented entities); no bank text."""
import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider, state_lines
from plugins.memory.hindsight.settings import _DATED_RECALL_HEADER

# Slice A's DDL (memory_system/entity_state/index.py, bffa59a).
_DDL = (
    "create table meta(key text primary key, value text)",
    "create table status(unit_id text, key text, kind text, entity text, word text, created real, trust text,"
    " source_kind text, text text, primary key(unit_id, key))",
    "create table marks(unit_id text, status_id text, key text, kind text, sim real, status_created real,"
    " primary key(unit_id, status_id))",
    "create index marks_status on marks(status_id)",
)
_PREAMBLE = ("# Hindsight Memory (persistent cross-session context)\nUse this to answer questions about the user "
             "and prior sessions. Do not call tools to look up information that is already present here.")


def _ts(day: str) -> float:
    return datetime.fromisoformat(f"{day}T12:00:00+00:00").timestamp()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# status rows: (unit_id, key, kind, entity, word, created, text)
STATUS = [
    ("s1", "alpha", "lifecycle", "Project Alpha", "cancelled", _ts("2026-10-20"), "Project Alpha was cancelled"),
    ("s2", "widget", "update", "Widget", "replaced", _ts("2026-10-18"), "Widget replaced by Gadget"),
    ("s3", "x", "update", "", "", _ts("2026-10-22"), "x moved to the new rack"),
    ("s4", "beta", "update", "Beta", "paused", _ts("2026-10-19"), "Beta paused"),
]
# marks rows: (unit_id, status_id); key/kind/created come from the status row
MARKS = [("h0", "s1"), ("h0", "s2"), ("h1", "s1"), ("h2", "s2"), ("h2", "s3"), ("h3", "s4")]
HITS = ["h0", "h1", "h2", "h3", "h4"]
TEXTS = {"h0": "alpha note", "h1": "alpha plan", "h2": "widget note", "h3": "beta note", "h4": "unrelated note"}
BLOCK = "\n".join([
    "Latest status per entity (newer than the lines marked below; prefer it when they conflict):",
    "- [2026-10-20] Project Alpha: cancelled | Project Alpha was cancelled",
    "- [2026-10-18] Widget: replaced | Widget replaced by Gadget",
    "- [2026-10-22] x | x moved to the new rack",
])
PLAIN_HITS = "- alpha note\n- alpha plan\n- widget note\n- beta note\n- unrelated note"
MARKED_HITS = ("- [older than alpha status 2026-10-20] alpha note\n"
               "- [older than alpha status 2026-10-20] alpha plan\n"
               "- [older than widget status 2026-10-18] widget note\n"
               "- beta note\n"
               "- unrelated note")
EXPECTED_LINES = [{"status_id": "s1", "key": "alpha", "kind": "lifecycle", "covers": [0, 1]},
                  {"status_id": "s2", "key": "widget", "kind": "update", "covers": [0, 2]},
                  {"status_id": "s3", "key": "x", "kind": "update", "covers": [2]}]
# Golden: `memory_system.entity_state.index.annotate(<this index>, HITS)` at slice A bffa59a (wt/t_c0782ed2,
# not yet on memory-system main), run on the index _build() writes; copied verbatim.
GOLDEN = {
    "block": BLOCK,
    "lines": [
        {"status_id": "s1", "key": "alpha", "kind": "lifecycle", "created": _ts("2026-10-20"), "covers": [0, 1],
         "line": "- [2026-10-20] Project Alpha: cancelled | Project Alpha was cancelled"},
        {"status_id": "s2", "key": "widget", "kind": "update", "created": _ts("2026-10-18"), "covers": [0, 2],
         "line": "- [2026-10-18] Widget: replaced | Widget replaced by Gadget"},
        {"status_id": "s3", "key": "x", "kind": "update", "created": _ts("2026-10-22"), "covers": [2],
         "line": "- [2026-10-22] x | x moved to the new rack"},
    ],
    "marks": [
        {"pos": 0, "unit_id": "h0", "prefix": "[older than alpha status 2026-10-20] "},
        {"pos": 1, "unit_id": "h1", "prefix": "[older than alpha status 2026-10-20] "},
        {"pos": 2, "unit_id": "h2", "prefix": "[older than widget status 2026-10-18] "},
    ],
}


def _build(path, *, status=STATUS, marks=MARKS, schema="1", age_s=3600.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    for stmt in _DDL:
        conn.execute(stmt)
    conn.executemany("insert into meta values (?, ?)",
                     [("schema_version", schema), ("built_at", _iso(time.time() - age_s)), ("bank", "test")])
    by_id = {}
    for sid, key, kind, entity, word, created, text in status:
        by_id[sid] = (key, kind, created)
        conn.execute("insert into status values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (sid, key, kind, entity, word, created, "user", "chat", text))
    for uid, sid in marks:
        key, kind, created = by_id[sid]
        conn.execute("insert into marks values (?, ?, ?, ?, ?, ?)", (uid, sid, key, kind, 0.5, created))
    conn.commit()
    conn.close()
    return path


def _annotate(hits, path, **kw):
    ann = state_lines.annotate(hits, path=str(path), **kw)
    assert ann is not None
    return ann


def _hit(uid, mentioned_at=None):
    return SimpleNamespace(id=uid, text=TEXTS.get(uid, "text"), mentioned_at=mentioned_at, occurred_start=None,
                           scores=None)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Default Hermes root for the index + JSONL (tmp); returns it."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(state_lines, "get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    return tmp_path


def _index(root):
    return root / "data" / "memory-system" / "entity-state" / "index.sqlite"


def _jsonl(root):
    path = root / "data" / "memory-system" / "state_lines.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _provider(root, **config):
    path = root / "hindsight" / "config.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"mode": "cloud", "apiKey": "k", "api_url": "http://localhost:9999", "bank_id": "b",
                                "auto_retain": False, "recall_sync": True, **config}))
    p = HindsightMemoryProvider()
    p.initialize(session_id="state-lines-test", hermes_home=str(root), platform="cli")
    return p


def _serve(p, results):
    client = MagicMock()
    client.arecall = AsyncMock(return_value=SimpleNamespace(results=results))
    p._client = p._reflect_client = client


def _prefetch(root, results, **config):
    p = _provider(root, **config)
    _serve(p, results)
    return p.prefetch("what about alpha?")


def _tool(root, results, **config):
    p = _provider(root, **config)
    _serve(p, results)
    return json.loads(p.handle_tool_call("hindsight_recall", {"query": "alpha"}))["result"]




class TestModes:
    def test_off_is_byte_identical_and_writes_nothing(self, root):
        _build(_index(root))
        rows = [_hit(u) for u in HITS]
        assert _prefetch(root, rows) == f"{_PREAMBLE}\n\n{PLAIN_HITS}"
        assert _prefetch(root, rows, recall_state_lines="off") == f"{_PREAMBLE}\n\n{PLAIN_HITS}"
        assert _tool(root, rows, recall_state_lines="off") == "\n".join(
            f"{i}. {TEXTS[u]}" for i, u in enumerate(HITS, 1))
        assert _jsonl(root) == []

    def test_shadow_serves_plain_and_logs_one_row(self, root):
        _build(_index(root))
        out = _prefetch(root, [_hit(u) for u in HITS], recall_state_lines="shadow")
        assert out == f"{_PREAMBLE}\n\n{PLAIN_HITS}"
        [row] = _jsonl(root)
        assert row["path"] == "auto" and row["hit_ids"] == HITS and row["lines"] == EXPECTED_LINES
        assert row["chars_added"] == len(f"{BLOCK}\n\n{MARKED_HITS}") - len(PLAIN_HITS)
        assert row["query_sha"] == hashlib.sha256(b"what about alpha?").hexdigest()[:16]
        assert set(row) == {"ts", "path", "profile", "query_sha", "hit_ids", "lines", "chars_added",
                            "index_built_at"}
        assert row["index_built_at"].endswith("Z") and "alpha" not in json.dumps(row["query_sha"])

    def test_shadow_logs_zero_chars_when_nothing_is_marked(self, root):
        _build(_index(root))
        assert _prefetch(root, [_hit("h4")], recall_state_lines="shadow") == f"{_PREAMBLE}\n\n- unrelated note"
        [row] = _jsonl(root)
        assert row["lines"] == [] and row["chars_added"] == 0

    def test_on_injects_header_lines_and_prefixes(self, root):
        _build(_index(root))
        out = _prefetch(root, [_hit(u) for u in HITS], recall_state_lines="on")
        assert out == f"{_PREAMBLE}\n\n{BLOCK}\n\n{MARKED_HITS}"
        assert _jsonl(root)[0]["lines"] == EXPECTED_LINES

    def test_on_with_dates_puts_prefix_after_the_date(self, root):
        _build(_index(root))
        out = _prefetch(root, [_hit("h1", "2026-10-03T08:00:00Z")], recall_state_lines="on", recall_show_dates=True)
        assert out.endswith(f"{_DATED_RECALL_HEADER}\n- [2026-10-03] [older than alpha status 2026-10-20] alpha plan")

    @pytest.mark.parametrize("value,mode", [(True, "on"), (False, "off"), ("SHADOW", "shadow"), ("bogus", "off"),
                                            (None, "off")])
    def test_mode_parsing(self, value, mode):
        # YAML 1.1 reads bare on/off as booleans.
        assert state_lines.parse_mode(value) == mode


class TestToolPath:
    def test_on_with_dates_keeps_numbering_and_prefix_after_date(self, root):
        _build(_index(root), status=[("sx", "x", "update", "X", "moved", _ts("2026-10-20"), "X moved")],
               marks=[("hx", "sx")])
        out = _tool(root, [_hit("hx", "2026-10-03T09:00:00Z"), _hit("h4")],
                    recall_state_lines="on", recall_show_dates=True)
        assert out == "\n".join([
            "Latest status per entity (newer than the lines marked below; prefer it when they conflict):",
            "- [2026-10-20] X: moved | X moved",
            "",
            _DATED_RECALL_HEADER,
            "1. [2026-10-03] [older than x status 2026-10-20] text",
            "2. unrelated note",
        ])
        [row] = _jsonl(root)
        assert row["path"] == "tool" and row["lines"] == [
            {"status_id": "sx", "key": "x", "kind": "update", "covers": [0]}]

    def test_shadow_tool_output_is_unchanged(self, root):
        _build(_index(root))
        rows = [_hit(u) for u in HITS]
        assert _tool(root, rows, recall_state_lines="shadow") == _tool(root, rows)
        assert len(_jsonl(root)) == 1


def test_kanban_first_turn_path_is_labelled(root, monkeypatch):
    _build(_index(root))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_first_turn")
    p = _provider(root, recall_state_lines="shadow", recall_kanban_workers=False, recall_kanban_first_turn=True)
    p._kanban_card_title, p._kanban_recall = "Some card", ("Some card\n\nbody", [])
    _serve(p, [_hit(u) for u in HITS])
    assert p.prefetch("work kanban task t_first_turn") == f"{_PREAMBLE}\n\n{PLAIN_HITS}"
    assert [r["path"] for r in _jsonl(root)] == ["kanban_first"]


class TestRanking:
    """(-len(covers), lifecycle first, min(covers), -status_created, status_id), max_lines=3."""

    def _lines(self, tmp_path, status, marks, hits):
        ann = _annotate(hits, _build(tmp_path / "i.sqlite", status=status, marks=marks), max_lines=10)
        return [a["status_id"] for a in ann.lines]

    def test_more_covers_first(self, tmp_path):
        st = [("a", "k1", "lifecycle", "", "", _ts("2026-10-01"), "a"), ("b", "k2", "update", "", "", 1.0, "b")]
        assert self._lines(tmp_path, st, [("h0", "a"), ("h1", "b"), ("h2", "b")], ["h0", "h1", "h2"]) == ["b", "a"]

    def test_lifecycle_before_update(self, tmp_path):
        st = [("a", "k1", "update", "", "", _ts("2026-10-09"), "a"), ("b", "k2", "lifecycle", "", "", 1.0, "b")]
        assert self._lines(tmp_path, st, [("h0", "a"), ("h0", "b")], ["h0"]) == ["b", "a"]

    def test_earliest_covered_position(self, tmp_path):
        st = [("a", "k1", "update", "", "", _ts("2026-10-09"), "a"), ("b", "k2", "update", "", "", 1.0, "b")]
        assert self._lines(tmp_path, st, [("h1", "a"), ("h0", "b")], ["h0", "h1"]) == ["b", "a"]

    def test_newer_status_first(self, tmp_path):
        st = [("a", "k1", "update", "", "", _ts("2026-10-01"), "a"),
              ("b", "k2", "update", "", "", _ts("2026-10-05"), "b")]
        assert self._lines(tmp_path, st, [("h0", "a"), ("h0", "b")], ["h0"]) == ["b", "a"]

    def test_status_id_breaks_the_last_tie(self, tmp_path):
        st = [("b", "k1", "update", "", "", _ts("2026-10-01"), "b"),
              ("a", "k2", "update", "", "", _ts("2026-10-01"), "a")]
        assert self._lines(tmp_path, st, [("h0", "a"), ("h0", "b")], ["h0"]) == ["a", "b"]

    def test_keeps_three_and_first_kept_line_owns_the_prefix(self, root):
        ann = _annotate(HITS, _build(_index(root)))
        assert [a["status_id"] for a in ann.lines] == ["s1", "s2", "s3"]  # s4 cut by max_lines
        assert ann.prefixes == {0: "[older than alpha status 2026-10-20] ",
                                1: "[older than alpha status 2026-10-20] ",
                                2: "[older than widget status 2026-10-18] "}


class TestFailOpen:
    @pytest.mark.parametrize("build", [
        None,  # missing file
        {"schema": "2"},
        {"age_s": 7 * 3600},
    ], ids=["missing", "schema_v2", "stale_7h"])
    def test_unusable_index_adds_nothing(self, root, build):
        if build is not None:
            _build(_index(root), **build)
        assert state_lines.annotate(HITS) is None
        rows = [_hit(u) for u in HITS]
        assert _prefetch(root, rows, recall_state_lines="on") == f"{_PREAMBLE}\n\n{PLAIN_HITS}"
        assert _tool(root, rows, recall_state_lines="on") == _tool(root, rows)
        assert _jsonl(root) == []

    def test_foreign_file_adds_nothing(self, root):
        _index(root).parent.mkdir(parents=True)
        _index(root).write_bytes(b"not a database")
        assert state_lines.annotate(HITS) is None

    def test_unwritable_log_is_swallowed(self, root):
        ann = _annotate(HITS, _build(_index(root)))
        blocker = root / "file"
        blocker.write_text("x")
        state_lines.log_block("auto", "q", HITS, ann, 0, path=str(blocker / "sub" / "x.jsonl"))


def test_golden_matches_slice_a_annotate(root):
    ann = _annotate(HITS, _build(_index(root)))
    assert ann.block == GOLDEN["block"]
    assert ann.lines == GOLDEN["lines"]
    assert [{"pos": p, "unit_id": HITS[p], "prefix": ann.prefixes[p]} for p in sorted(ann.prefixes)] == GOLDEN["marks"]


def test_default_paths_are_under_the_default_root(root):
    assert state_lines.index_path() == str(_index(root))
    assert state_lines.log_path() == str(root / "data" / "memory-system" / "state_lines.jsonl")
