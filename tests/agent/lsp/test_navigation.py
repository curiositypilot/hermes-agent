"""``agent.lsp.navigation`` shaping helpers: positions (1-based in, UTF-16 out) and result parsing."""
from __future__ import annotations

import subprocess

import pytest

from agent.lsp import navigation
from agent.lsp.client import file_uri
from agent.lsp.navigation import (
    NavigationError, _best_matches, _declaration, _hover_text, _locations, _occurrences, _outline, _own_root_only,
    _position, _symbol_re)


def test_symbol_position_is_utf16_and_word_bounded():
    # 😀 is one Python char but two UTF-16 units; "BrainBox" must not match "Brain".
    text = "x = 1\ns = '😀'; BrainBox; Brain()\n"
    pos, line = _position(text, symbol="Brain", line=2, column=None)
    assert line == 2
    assert pos == {"line": 1, "character": len("s = '😀'; BrainBox; ".encode("utf-16-le")) // 2}


def test_symbol_off_by_one_line_is_found_nearby_and_every_occurrence_is_a_candidate_without_line():
    text = "a\nb\nclass Brain:\n    pass\n"
    assert _position(text, symbol="Brain", line=2, column=None) == ({"line": 2, "character": 6}, 3)
    # Without a line the first text match is only the fallback ``at``: a docstring mention comes first,
    # and the loop side probes the occurrences (one per line, in file order) when the outline has no match.
    doc_first = '"""Brain docs, Brain."""\nclass Brain:\n    x = Brain\n'
    assert _position(doc_first, symbol="Brain", line=None, column=None) == ({"line": 0, "character": 3}, 1)
    assert _occurrences(doc_first.split("\n"), _symbol_re("Brain")) == [
        ({"line": 0, "character": 3}, 1), ({"line": 1, "character": 6}, 2), ({"line": 2, "character": 8}, 3)]


@pytest.mark.parametrize("kwargs, needle", [
    ({"symbol": "Nope", "line": 1, "column": None}, "does not occur on line 1"),
    ({"symbol": "Nope", "line": None, "column": None}, "does not occur in the file"),
    ({"symbol": None, "line": 99, "column": None}, "out of range"),
    ({"symbol": None, "line": None, "column": None}, "give symbol"),
])
def test_bad_positions_explain_themselves(kwargs, needle):
    with pytest.raises(NavigationError, match=needle):
        _position("one\ntwo\n", **kwargs)


def test_locations_accept_location_link_and_null_shapes(tmp_path):
    a, b = str(tmp_path / "a.py"), str(tmp_path / "b.py")
    rng = {"start": {"line": 4, "character": 2}, "end": {"line": 4, "character": 3}}
    assert _locations(None) == []
    assert _locations({"uri": file_uri(a), "range": rng}) == [(a, 4, 2)]
    link = {"targetUri": file_uri(b), "targetRange": rng, "targetSelectionRange": rng}
    assert _locations([link, link, {"uri": file_uri(a), "range": rng}]) == [(b, 4, 2), (a, 4, 2)]


def test_hover_flattens_marked_strings_and_caps():
    assert _hover_text({"contents": [{"language": "python", "value": "def f()"}, "Docs."]}) == "def f()\n\nDocs."
    assert len(_hover_text({"contents": {"kind": "markdown", "value": "x" * 5000}})) == navigation._HOVER_CHARS
    assert _hover_text(None) == ""


def test_symbol_search_ranks_exact_first_and_drops_fuzzy_noise():
    rows = [("/w/t.py", 1, 0, "function test_unreadable_subtree_is_preserved"),
            ("/w/a.py", 2, 0, "class LSPServiceFactory"), ("/w/m.py", 3, 0, "class LSPService (in manager)")]
    assert [r[0] for r in _best_matches(rows, "LSPService")] == ["/w/m.py", "/w/a.py"]
    assert _best_matches(rows[:1], "LSPService") == rows[:1]  # only fuzzy hits: keep them


def test_outline_expands_classes_not_function_locals():
    def sym(name, kind, line, children=()):
        rng = {"start": {"line": line, "character": 0}}
        return {"name": name, "kind": kind, "range": rng, "selectionRange": rng, "children": list(children)}
    raw = [sym("Brain", 5, 1, [sym("tick", 6, 2, [sym("self", 13, 2)])]), sym("main", 12, 9, [sym("x", 13, 10)])]
    assert [label for _, _, label in _outline(raw)] == ["class Brain", "  method tick", "function main"]


def test_rows_from_a_sibling_worktree_the_server_serves_are_dropped(tmp_path):
    """A multi-root pyright attached to two worktrees of one repo answers for both; a query from one
    worktree keeps its own rows and third-party rows, never the sibling's copy."""
    own, sibling, vendored = tmp_path / "own", tmp_path / "sibling", tmp_path / "own" / "sub"
    for repo in (own, sibling, vendored):
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    rows = [(str(own / "a.py"), 1, 0, None), (str(sibling / "a.py"), 1, 0, None),
            (str(vendored / "c.py"), 1, 0, None), ("/usr/lib/python3/x.py", 1, 0, None)]
    kept = _own_root_only(rows, str(own), [str(own), str(sibling)])
    assert [r[0] for r in kept] == [str(own / "a.py"), str(vendored / "c.py"), "/usr/lib/python3/x.py"]


def _doc_symbol(name, kind, line, char, children=()):
    sel = {"start": {"line": line, "character": char}, "end": {"line": line, "character": char + len(name)}}
    rng = {"start": {"line": line, "character": 0}, "end": {"line": line + 1, "character": 0}}
    return {"name": name, "kind": kind, "range": rng, "selectionRange": sel, "children": list(children)}


def test_declaration_prefers_the_top_level_symbol_over_a_nested_namesake():
    lines = ["class Holder:", "    def Brain(self): ...", "class Brain:", "    pass", ""]
    outline = [_doc_symbol("Holder", 5, 0, 6, [_doc_symbol("Brain", 6, 1, 8)]), _doc_symbol("Brain", 5, 2, 6)]
    assert _declaration(outline, "Brain", lines) == ({"line": 2, "character": 6}, 3)
    # Only the nested one: found at any depth, by its selectionRange.
    assert _declaration(outline[:1], "Brain", lines) == ({"line": 1, "character": 8}, 2)


def test_declaration_from_symbol_information_finds_the_identifier_past_a_decorator():
    lines = ["import x", "@dataclass", "class Brain: pass", ""]
    info = {"name": "Brain", "kind": 5, "location": {"uri": "file:///w/b.py", "range": {
        "start": {"line": 1, "character": 0}, "end": {"line": 2, "character": 17}}}}
    assert _declaration([info], "Brain", lines) == ({"line": 2, "character": 6}, 3)
    # The identifier outside the range: not this symbol's declaration.
    narrow = {**info, "location": {"uri": "file:///w/b.py", "range": {
        "start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}}}}
    assert _declaration([narrow], "Brain", lines) is None


@pytest.mark.parametrize("outline", [
    [_doc_symbol("BrainBox", 5, 0, 6), _doc_symbol("brain", 13, 1, 0)],
    [],
    None,
])
def test_declaration_needs_the_exact_name(outline):
    assert _declaration(outline, "Brain", ["class BrainBox: pass", "brain = 1", ""]) is None
