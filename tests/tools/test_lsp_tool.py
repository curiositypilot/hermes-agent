"""``lsp`` tool end to end: registry dispatch → LSPService → a real (mock) language server process.

Covers the four actions, the result cap, and the gate the tool shares with write-time diagnostics:
an untrusted checkout never starts a server that may load project code, ``lsp.enabled: false`` hides
the tool, and a file outside any git worktree is refused with a reason.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tools.registry import registry

_MOCK = os.path.join(os.path.dirname(__file__), "..", "agent", "lsp", "_mock_lsp_server.py")
_SOURCE = "import os\nclass Brain:\n    def tick(self): s = '😀'; return Brain\n"
# The first text match is a docstring mention the server has nothing to say about.
_DOC_FIRST = '"""Brain docs."""\nimport os\nclass Brain:\n    def tick(self): return Brain\n'


@pytest.fixture
def nav_env(tmp_path, monkeypatch):
    """A git workspace holding ``brain_node.nav`` served by the mock ``nav`` server; returns a
    ``configure(trusted=True, enabled=True, refs=3, ...)`` that writes config.yaml (and, given ``source``,
    the target file) and yields paths; ``no_outline`` / ``dead_lines`` drive the mock's nav knobs."""
    import tools.lsp_tool  # registers the tool
    from agent.lsp import shutdown_service

    home, ws = tmp_path / "home", tmp_path / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    target = ws / "brain_node.nav"
    target.write_text(_SOURCE, encoding="utf-8")
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")  # no operator-trusted roots: only the config decides
    monkeypatch.setattr("agent.lsp.navigation.WORKSPACE_SETTLE", 0.0)  # the mock lists no files

    def configure(*, trusted=True, enabled=True, refs=3, index_seconds=None, source=None, no_outline=False,
                  dead_lines=()):
        if source is not None:
            target.write_text(source, encoding="utf-8")
        env = {"MOCK_LSP_SCRIPT": "nav", "MOCK_LSP_NAV_TARGET": str(target), "MOCK_LSP_REQUESTS_LOG": str(log),
               "MOCK_LSP_NAV_REFS": str(refs)}
        if index_seconds is not None:
            env["MOCK_LSP_INDEX_SECONDS"] = str(index_seconds)
        if no_outline:
            env["MOCK_LSP_NAV_NO_OUTLINE"] = "1"
        if dead_lines:
            env["MOCK_LSP_NAV_DEAD_LINES"] = ",".join(map(str, dead_lines))
        (home / "config.yaml").write_text(json.dumps({"lsp": {
            "enabled": enabled, "install_strategy": "manual", "idle_timeout": 0,
            "trusted_workspaces": [str(ws)] if trusted else [],
            "servers": {"navmock": {"command": [sys.executable, _MOCK], "extensions": [".nav"], "env": env}},
        }}), encoding="utf-8")
        shutdown_service()
        return target, log

    yield configure
    shutdown_service()


def _call(**args):
    result = registry.dispatch("lsp", args, task_id="lsp-test")
    return json.loads(result) if isinstance(result, str) else result


def _requests(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


@pytest.mark.timeout(90)
def test_definition_references_hover_and_symbols_through_a_server(nav_env):
    target, log = nav_env()
    path = str(target)

    out = _call(action="definition", path=path, symbol="Brain", line=3)
    assert out["results"] == ["brain_node.nav:2:7  class Brain:"], out
    assert out["at"] == "brain_node.nav:3" and out["server"] == "navmock"
    # The position sent is the symbol's UTF-16 offset on the 0-based line (😀 counts two units).
    sent = _requests(log)[-1]["params"]["position"]
    assert sent == {"line": 2, "character": len("    def tick(self): s = '😀'; return ".encode("utf-16-le")) // 2}

    refs = _call(action="references", path=path, symbol="Brain")
    assert refs["total"] == 3
    assert refs["results"] == ["brain_node.nav:1:1  import os", "brain_node.nav:2:1  class Brain:",
                               "brain_node.nav:3:1  def tick(self): s = '😀'; return Brain"]
    assert _requests(log)[-1]["params"]["context"] == {"includeDeclaration": True}

    assert "class Brain" in _call(action="hover", path=path, symbol="Brain", line=2)["hover"]

    found = _call(action="symbols", path=path, query="Brain")
    assert found["results"] == ["brain_node.nav:2:7  class Brain (in brain_node)"]
    assert _call(action="symbols", path=path)["results"] == ["2: class Brain", "3:   method tick"]


@pytest.mark.timeout(90)
def test_results_are_capped_with_a_count(nav_env):
    target, _ = nav_env(refs=80)
    out = _call(action="references", path=str(target), symbol="Brain")
    assert out["total"] == 80 and len(out["results"]) == 50 and "30 more" in out["note"]


@pytest.mark.timeout(90)
def test_workspace_wide_query_waits_for_the_server_to_finish_indexing(nav_env, monkeypatch):
    """References sent while the server still indexes cover only the open file; the tool waits for
    the server's work-done progress to end, so the answer is complete."""
    monkeypatch.setattr("agent.lsp.navigation.WORKSPACE_SETTLE", 1.0)  # lets the progress begin arrive
    target, _ = nav_env(index_seconds=3.0)
    assert _call(action="references", path=str(target), symbol="Brain")["total"] == 3


@pytest.mark.timeout(60)
def test_untrusted_workspace_never_starts_a_project_code_server(nav_env):
    target, log = nav_env(trusted=False)
    out = _call(action="definition", path=str(target), symbol="Brain")
    assert "not a trusted workspace" in out["error"]
    assert _requests(log) == []  # refused before any server process was spawned


@pytest.mark.timeout(60)
def test_disabled_lsp_hides_the_tool_and_refuses_calls(nav_env):
    from tools.lsp_tool import _check_lsp_reqs

    target, _ = nav_env(enabled=False)
    assert _check_lsp_reqs() is False
    assert "disabled" in _call(action="definition", path=str(target), symbol="Brain")["error"]
    nav_env(enabled=True)
    assert _check_lsp_reqs() is True


@pytest.mark.timeout(60)
def test_failures_are_one_line_reasons(nav_env, tmp_path):
    from agent.lsp.workspace import find_git_worktree

    target, _ = nav_env()
    loose = tmp_path / "loose" / "x.nav"  # not inside the test's git worktree
    loose.parent.mkdir()
    loose.write_text("Brain\n", encoding="utf-8")
    err = _call(action="definition", path=str(loose), symbol="Brain")["error"]
    # A host whose temp dir sits inside some git checkout gets that checkout's (untrusted) verdict instead.
    assert ("not inside a git worktree" if find_git_worktree(str(loose)) is None else "not a trusted workspace") in err
    other = target.with_suffix(".zzz")
    other.write_text("x\n", encoding="utf-8")
    assert "no language server handles .zzz" in _call(action="hover", path=str(other), line=1)["error"]
    assert "unknown action" in _call(action="rename", path=str(target))["error"]
    assert "does not occur" in _call(action="definition", path=str(target), symbol="Nope")["error"]


def _positions(log, method):
    return [r["params"]["position"] for r in _requests(log) if r["method"] == method]


@pytest.mark.timeout(90)
def test_symbol_without_line_resolves_to_its_declaration_not_a_docstring_mention(nav_env):
    target, log = nav_env(source=_DOC_FIRST)
    out = _call(action="references", path=str(target), symbol="Brain")
    assert out["at"] == "brain_node.nav:3" and out["total"] == 3, out
    methods = [r["method"] for r in _requests(log)]
    assert methods == ["textDocument/documentSymbol", "textDocument/references"]
    assert _positions(log, "textDocument/references") == [{"line": 2, "character": 6}]


@pytest.mark.timeout(90)
def test_symbol_without_outline_entry_probes_occurrences_until_one_answers(nav_env):
    target, log = nav_env(source=_DOC_FIRST, no_outline=True, dead_lines=[0])
    out = _call(action="references", path=str(target), symbol="Brain")
    assert out["at"] == "brain_node.nav:3" and out["total"] == 3, out
    assert [p["line"] for p in _positions(log, "textDocument/references")] == [0, 2]


@pytest.mark.timeout(90)
def test_symbol_with_no_answering_occurrence_names_the_lines_tried(nav_env):
    target, _ = nav_env(source=_DOC_FIRST, no_outline=True, dead_lines=[0, 2, 3])
    out = _call(action="references", path=str(target), symbol="Brain")
    assert out["total"] == 0 and out["at"] == "brain_node.nav:1", out
    assert out["note"] == "no results at lines 1, 3, 4; give line"
    hover = _call(action="hover", path=str(target), symbol="Brain")
    assert hover["hover"] == "(no hover information)" and "lines 1, 3, 4" in hover["note"]


@pytest.mark.timeout(90)
def test_given_line_is_used_as_is_without_outline_or_probes(nav_env):
    target, log = nav_env(source=_DOC_FIRST, dead_lines=[0])
    out = _call(action="references", path=str(target), symbol="Brain", line=1)
    assert out["total"] == 0 and out["at"] == "brain_node.nav:1", out
    requests = _requests(log)
    assert [r["method"] for r in requests] == ["textDocument/references"]
    assert requests[0]["params"]["position"]["line"] == 0
