"""Memory-provider tools named in the tool_search defer set go behind the bridge.

Memory-provider schemas (hindsight_reflect, ...) are not registry tools: they used to be
appended after ``get_tool_definitions`` ran tool_search assembly, so naming one in
``tools.tool_search.defer`` or ``kanban.worker_defer_tools`` did nothing and the bridge could
not reach it. These tests drive the real config on disk (temp ``HERMES_HOME``), the real
``inject_memory_provider_tools``, assembly, bridge dispatch and executor unwrap.
"""

import copy
import json
import types

import pytest
import hermes_yaml as yaml

import model_tools
from agent.memory_manager_tools import inject_memory_provider_tools

TOOLSETS = ["file", "memory"]


def _schema(name):
    return {"name": name, "description": f"{name} memory op",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"]}}


class FakeManager:
    providers = []

    def __init__(self):
        self.calls = []

    def get_all_tool_schemas(self):
        return [_schema("fake_reflect"), _schema("fake_recall")]

    def has_tool(self, name):
        return name in ("fake_reflect", "fake_recall")

    def handle_tool_call(self, name, args, **kw):
        self.calls.append((name, args))
        return json.dumps({"ok": name})


class FakeAgent(types.SimpleNamespace):
    def _should_emit_quiet_tool_messages(self):
        return False


@pytest.fixture(autouse=True)
def _clear_caches():
    from tools.registry import invalidate_check_fn_cache
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.delegation_context.owned_kanban_task", lambda: "")
    return home


def _config(home, defer=None, worker_defer=None, enabled="on"):
    cfg = {"tools": {"tool_search": {"enabled": enabled, "defer": defer or []}}}
    if worker_defer is not None:
        cfg["kanban"] = {"worker_defer_tools": worker_defer}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _agent(manager=None, enabled=TOOLSETS, disabled=None):
    tools = model_tools.get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True)
    return FakeAgent(_memory_manager=manager or FakeManager(), tools=tools,
                     valid_tool_names={t["function"]["name"] for t in tools},
                     enabled_toolsets=enabled, disabled_toolsets=disabled, platform="cli",
                     _context_engine_tool_names=set())


def _names(tools):
    return [t["function"]["name"] for t in tools]


def _tool_call(name, arguments):
    payload = {"calls": [{"name": name, "arguments": arguments}]}
    return types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="tool_call", arguments=json.dumps(payload)))


def _bridge(agent, name, args):
    from agent.memory_manager_tools import assembled_memory_tool_defs
    return json.loads(model_tools.handle_function_call(
        name, args, task_id="t", enabled_toolsets=agent.enabled_toolsets,
        disabled_toolsets=agent.disabled_toolsets, extra_tool_defs=assembled_memory_tool_defs(agent)))


class TestSurface:
    def test_deferred_memory_tool_moves_behind_bridge(self, home):
        _config(home, defer=["fake_reflect"])
        agent = _agent()
        assert inject_memory_provider_tools(agent) == 2
        names = _names(agent.tools)
        assert "fake_reflect" not in names and "fake_reflect" not in agent.valid_tool_names
        assert {"fake_recall", "tool_search", "tool_call"} <= set(names)
        search = next(t for t in agent.tools if t["function"]["name"] == "tool_search")
        assert "fake_reflect" in search["function"]["description"]
        assert agent._deferred_memory_tool_names == frozenset({"fake_reflect"})

    def test_kanban_worker_defer_tools_defers_in_owned_worker_only(self, home, monkeypatch):
        _config(home, worker_defer=["fake_reflect"])
        monkeypatch.setattr("agent.delegation_context.owned_kanban_task", lambda: "t_x")
        worker = _agent()
        inject_memory_provider_tools(worker)
        assert "fake_reflect" not in _names(worker.tools)
        monkeypatch.setattr("agent.delegation_context.owned_kanban_task", lambda: "")
        chat = _agent()
        inject_memory_provider_tools(chat)
        assert _names(chat.tools)[-2:] == ["fake_reflect", "fake_recall"]

    @pytest.mark.parametrize("enabled", ["on", "off"])
    def test_no_memory_name_deferred_keeps_bytes(self, home, enabled):
        _config(home, defer=["fake_reflect"] if enabled == "off" else [], enabled=enabled)
        agent = _agent()
        expected = copy.deepcopy(agent.tools) + [{"type": "function", "function": _schema(n)}
                                                 for n in ("fake_reflect", "fake_recall")]
        inject_memory_provider_tools(agent)
        assert json.dumps(agent.tools) == json.dumps(expected)
        assert not getattr(agent, "_memory_tools_assembled", False)

    def test_memory_toolset_disabled_adds_nothing(self, home):
        _config(home, defer=["fake_reflect"])
        agent = _agent(disabled=["memory"])
        before = copy.deepcopy(agent.tools)
        assert inject_memory_provider_tools(agent) == 0
        assert agent.tools == before and agent._memory_tool_defs == []
        assert not getattr(agent, "_memory_tools_assembled", False)

    def test_extras_do_not_poison_memo(self, home):
        _config(home)
        extras = [{"type": "function", "function": _schema("fake_reflect")}]
        with_extra = model_tools.get_tool_definitions(enabled_toolsets=TOOLSETS, quiet_mode=True, extra_tool_defs=extras)
        assert "fake_reflect" in _names(with_extra)
        assert "fake_reflect" not in _names(model_tools.get_tool_definitions(enabled_toolsets=TOOLSETS, quiet_mode=True))


class TestBridge:
    def test_search_and_describe_reach_deferred_memory_tool(self, home):
        _config(home, defer=["fake_reflect"])
        agent = _agent()
        inject_memory_provider_tools(agent)
        found = _bridge(agent, "tool_search", {"queries": ["fake reflect memory"]})
        assert any("fake_reflect" in g["matches"] for g in found["results"])
        described = _bridge(agent, "tool_describe", {"names": ["fake_reflect"]})
        assert described["tools"]["fake_reflect"]["parameters"]["required"] == ["query"]

    def test_tool_call_unwraps_and_dispatches_to_memory_manager(self, home):
        from agent.tool_executor import _ToolCallRef, _parse_tool_call, _resolve_sequential_dispatch
        _config(home, defer=["fake_reflect"])
        agent = _agent()
        inject_memory_provider_tools(agent)
        parsed = _parse_tool_call(agent, _tool_call("fake_reflect", {"query": "why"}))
        assert (parsed.name, parsed.args, parsed.scope_block) == ("fake_reflect", {"query": "why"}, None)
        dispatch = _resolve_sequential_dispatch(agent, _ToolCallRef(parsed.name, parsed.args, "t", "c1", []), [])
        assert json.loads(dispatch.execute(parsed.args)) == {"ok": "fake_reflect"}
        assert agent._memory_manager.calls == [("fake_reflect", {"query": "why"})]

    def test_concurrent_invoke_passes_memory_catalog(self, home, monkeypatch):
        from agent.agent_runtime_helpers import invoke_tool
        _config(home, defer=["fake_reflect"])
        agent = _agent()
        inject_memory_provider_tools(agent)
        seen = {}
        monkeypatch.setattr(model_tools, "handle_function_call",
                            lambda name, args, *a, **kw: seen.update(kw) or "{}")
        agent.session_id = "s"
        invoke_tool(agent, "tool_search", {"queries": ["fake"]}, "t", pre_tool_block_checked=True,
                    skip_tool_request_middleware=True, skip_tool_execution_middleware=True)
        assert "fake_reflect" in _names(seen.get("extra_tool_defs") or [])

    def test_must_not_bridge_eager_or_absent_memory_tool(self, home):
        from agent.tool_executor import _parse_tool_call
        _config(home, defer=["fake_reflect"])
        agent = _agent()
        inject_memory_provider_tools(agent)
        err = _bridge(agent, "tool_call", {"calls": [{"name": "fake_recall", "arguments": {"query": "x"}}]})
        assert "directly-listed tool, not a deferred one" in err["error"]
        bare = _agent()
        bare._memory_manager = None
        parsed = _parse_tool_call(bare, _tool_call("fake_reflect", {"query": "x"}))
        assert parsed.name == "tool_call" and parsed.scope_block

    def test_direct_call_of_deferred_tool_is_redirected(self):
        from agent.conversation_loop import _invalid_tool_name_error_content
        msg = _invalid_tool_name_error_content("fake_reflect", {"read_file"}, frozenset({"fake_reflect"}))
        assert "tool_call" in msg and "deferred" in msg

    def test_prompt_block_names_deferred_tools_only_when_deferring(self, home):
        from agent.system_prompt import _memory_parts
        _config(home, defer=["fake_reflect"])
        agent = _agent()
        agent._memory_manager.build_system_prompt = lambda: "Use fake_reflect to synthesize."
        agent._memory_store = None
        inject_memory_provider_tools(agent)
        assert any("Deferred memory tools (fake_reflect)" in p for p in _memory_parts(agent))
        agent._deferred_memory_tool_names = frozenset()
        assert _memory_parts(agent) == ["Use fake_reflect to synthesize."]


def test_mcp_refresh_keeps_deferred_memory_tool_off_surface(home):
    from tools import mcp_tool_agent
    _config(home, defer=["fake_reflect"])
    agent = _agent()
    inject_memory_provider_tools(agent)
    mcp_tool_agent.refresh_agent_mcp_tools(agent)
    names = _names(agent.tools)
    assert "fake_reflect" not in names and names.count("fake_recall") == 1
    assert "tool_search" in names
