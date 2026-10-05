"""Kanban worker context: task-scoped Hindsight directives in a ``## Principles`` block.

The network seam is ``hermes_cli.principles._get_json``; everything above it (tag
resolution, ordering, cap, rendering in ``build_worker_context``) runs for real.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import principles


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    c = kbc.connect()
    yield c
    c.close()


def _directive(name, content="rule text", priority=0, active=True, tags=("task:coding",)):
    return {"name": name, "content": content, "priority": priority, "is_active": active, "tags": list(tags)}


def _serve(monkeypatch, items):
    """Mock the Hindsight GET; returns the list of URLs requested."""
    urls: list[str] = []

    def fake(url, timeout):
        urls.append(url)
        assert timeout <= 5
        return {"items": items}

    monkeypatch.setattr(principles, "_get_json", fake)
    return urls


def _down(monkeypatch, error: Exception = ConnectionRefusedError("[Errno 111] Connection refused")):
    def fake(url, timeout):
        raise error

    monkeypatch.setattr(principles, "_get_json", fake)


def _ctx(conn, **create):
    tid = kb.create_task(conn, **create)
    return kb.build_worker_context(conn, tid)


def _principles_section(ctx: str) -> str:
    assert "## Principles" in ctx
    return ctx.split("## Principles", 1)[1]


# -- block present -----------------------------------------------------------------------

def test_block_present_priority_ordered_and_last(conn, monkeypatch):
    urls = _serve(monkeypatch, [
        _directive("b-low", "low rule", priority=0),
        _directive("a-high", "high rule", priority=9),
        _directive("c-off", "inactive rule", priority=99, active=False),
    ])
    tid = kb.create_task(conn, title="fork: do a thing", body="## Goal\nx")
    kb.add_comment(conn, tid, author="alice", body="a comment")
    ctx = kb.build_worker_context(conn, tid)

    section = _principles_section(ctx)
    assert section.index("a-high: high rule") < section.index("b-low: low rule")
    assert "inactive rule" not in ctx
    assert ctx.rindex("## Principles") > ctx.rindex("## Comment thread")  # ends the prompt
    assert "task%3Acoding" in urls[0] and "tags_match=any" in urls[0]


@pytest.mark.parametrize("title,tag", [
    ("fork: x", "task:coding"), ("scripts: x", "task:coding"), ("plugin: x", "task:coding"),
    ("research: x", "task:research"), ("review: x", "task:code-review"), ("Fork: x", "task:coding"),
])
def test_title_prefix_selects_tag(conn, monkeypatch, title, tag):
    urls = _serve(monkeypatch, [_directive("r", "body")])
    ctx = _ctx(conn, title=title)
    assert "## Principles" in ctx
    assert urls == [principles.directives_url([tag])]


def test_body_line_overrides_title_and_takes_many_tags(conn, monkeypatch):
    urls = _serve(monkeypatch, [_directive("r", "body")])
    _ctx(conn, title="fork: x", body="## Goal\nstuff\nprincipally: no\nprinciples: task:research task:card-filing\n")
    assert urls == [principles.directives_url(["task:research", "task:card-filing"])]


def test_placeholder_body_line_falls_back_to_title(conn, monkeypatch):
    urls = _serve(monkeypatch, [_directive("r", "body")])
    _ctx(conn, title="research: x", body="tag source = a body line `principles: <tag> [<tag>]`")
    assert urls == [principles.directives_url(["task:research"])]


def test_untagged_card_gets_no_block_and_no_request(conn, monkeypatch):
    urls = _serve(monkeypatch, [_directive("r", "body")])
    ctx = _ctx(conn, title="just a card")
    assert "## Principles" not in ctx
    assert urls == []


# -- cap -----------------------------------------------------------------------------------

def test_cap_honoured_in_context_and_drops_lowest_priority_first(conn, monkeypatch):
    items = [_directive(f"rule-{i:02d}", "x" * 400, priority=100 - i) for i in range(20)]  # ~8.4k chars
    _serve(monkeypatch, items)
    ctx = _ctx(conn, title="fork: big")

    lines = [ln for ln in _principles_section(ctx).splitlines() if ln.startswith("- rule-")]
    assert 0 < len(lines) < 20
    assert sum(len(ln) + 1 for ln in lines) <= 3000
    assert lines[0].startswith("- rule-00")                      # highest priority kept
    assert [ln.split(":")[0] for ln in lines] == [f"- rule-{i:02d}" for i in range(len(lines))]  # no gaps
    assert f"{20 - len(lines)} lower-priority directives omitted" in ctx


def test_fetch_directives_cap_is_a_parameter(monkeypatch):
    _serve(monkeypatch, [_directive(f"r{i}", "y" * 50, priority=i) for i in range(10)])
    got = principles.fetch_directives(["task:coding"], cap_chars=200)
    assert sum(len(principles.render_directive(d)) + 1 for d in got) <= 200
    assert [d["name"] for d in got][:2] == ["r9", "r8"]
    assert getattr(got, "omitted") == 10 - len(got)
    assert len(principles.fetch_directives(["task:coding"], cap_chars=None)) == 10


def test_single_directive_larger_than_cap_is_truncated_not_dropped(monkeypatch):
    _serve(monkeypatch, [_directive("huge", "z" * 5000)])
    got = principles.fetch_directives(["task:coding"], cap_chars=300)
    assert len(got) == 1
    assert len(principles.render_directive(got[0])) <= 300
    assert got[0]["content"].endswith("…")


# -- Hindsight down --------------------------------------------------------------------------

def test_down_path_is_one_line_and_context_still_builds(conn, monkeypatch):
    _down(monkeypatch)
    ctx = _ctx(conn, title="fork: x")
    section = _principles_section(ctx)
    assert "principles: unavailable (" in section and "Connection refused" in section
    assert len(section.strip().splitlines()) == 1  # one line, nothing else under the heading


def test_down_path_error_is_collapsed_to_one_capped_line(conn, monkeypatch):
    _down(monkeypatch, RuntimeError("boom\n" + "tail " * 500))
    section = _principles_section(_ctx(conn, title="fork: x"))
    body = section.strip().splitlines()
    assert body == ["principles: unavailable (boom)"]


def test_bad_payload_counts_as_unavailable(conn, monkeypatch):
    monkeypatch.setattr(principles, "_get_json", lambda url, timeout: ["not", "a", "dict"])
    assert "principles: unavailable (" in _ctx(conn, title="fork: x")


def test_fetch_directives_raises_with_cause_and_url(monkeypatch):
    _down(monkeypatch)
    with pytest.raises(principles.DirectivesUnavailable) as err:
        principles.fetch_directives(["task:coding"])
    assert "Connection refused" in str(err.value)
    assert "task%3Acoding" in err.value.url


def test_no_active_directives_is_said_plainly(conn, monkeypatch):
    _serve(monkeypatch, [])
    assert "principles: no active directives for task:coding" in _ctx(conn, title="fork: x")


# -- CLI shim contract ---------------------------------------------------------------------

def test_cli_output_format_and_exit_codes(monkeypatch, capsys):
    _serve(monkeypatch, [_directive("b", "two", 0), _directive("a", "one", 5)])
    assert principles.main(["task:coding"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "# principles for task:coding (2 rules, bank main)", "- a: one", "- b: two",
    ]
    assert principles.main([]) == 2
    assert "Usage: principles.py" in capsys.readouterr().err
    _down(monkeypatch)
    assert principles.main(["task:coding"]) == 1
    assert "principles.py:" in capsys.readouterr().err
