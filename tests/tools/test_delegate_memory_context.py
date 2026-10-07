"""delegate_task children get the parent's memory recall in their system prompt (t_2354171e).

Children stay ``skip_memory=True`` (no provider, no writes, no memory tools); the parent recalls on
each child's goal once, concurrently, under one shared deadline, and the block rides the child's
ephemeral system prompt.
"""
import threading
import time
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _build_children

_BLOCK = "<memory-context>\n- the 10-01 auto-recall audit covered 7 of 36 turns\n</memory-context>"
_CREDS = {"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None}


def _parent(manager=None):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.enabled_toolsets = ["terminal", "file"]
    parent.disabled_toolsets = []
    parent._memory_manager = manager
    return parent


class _FakeManager:
    def __init__(self, fn, timeout=8.0):
        self._fn = fn
        self._external_prefetch_timeout = timeout
        self.calls = []

    def delegation_context(self, query, *, timeout=None):
        self.calls.append((query, timeout))
        return self._fn(query)


def _build(parent, goals):
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.side_effect = lambda **kw: MagicMock()
        children, err = _build_children(
            [{"goal": g} for g in goals], [], dict(_CREDS), top_role="leaf", max_iterations=5,
            parent_agent=parent, routing_cfg={}, live_deleg_id=None, live_writers=[],
        )
    assert err is None
    return children, [c.kwargs for c in MockAgent.call_args_list]


def test_block_reaches_child_prompt_and_child_keeps_no_provider():
    manager = _FakeManager(lambda q: _BLOCK)
    children, built = _build(_parent(manager), ["Audit what Hindsight auto-recall injects"])
    assert len(children) == 1
    prompt = built[0]["ephemeral_system_prompt"]
    assert "RECALLED MEMORY (may be stale; verify before acting):" in prompt
    assert "7 of 36 turns" in prompt
    assert built[0]["skip_memory"] is True
    # Recall keyed on the child's goal, with the manager's own deadline.
    assert manager.calls == [("Audit what Hindsight auto-recall injects", 8.0)]


def test_goal_only_task_gets_block_without_context():
    """Block placement is independent of CONTEXT: a task with no context still gets it."""
    manager = _FakeManager(lambda q: _BLOCK)
    _, built = _build(_parent(manager), ["goal only"])
    prompt = built[0]["ephemeral_system_prompt"]
    assert "CONTEXT:" not in prompt
    assert "RECALLED MEMORY" in prompt


def test_parent_without_manager_builds_child_without_block():
    _, built = _build(_parent(None), ["anything"])
    assert "RECALLED MEMORY" not in built[0]["ephemeral_system_prompt"]
    assert built[0]["skip_memory"] is True


def test_manager_error_builds_child_without_block():
    def boom(_q):
        raise RuntimeError("hindsight down")
    children, built = _build(_parent(_FakeManager(boom)), ["anything"])
    assert len(children) == 1
    assert "RECALLED MEMORY" not in built[0]["ephemeral_system_prompt"]


def test_slow_manager_past_deadline_builds_child_without_block():
    def slow(_q):
        time.sleep(1.0)
        return _BLOCK
    start = time.monotonic()
    children, built = _build(_parent(_FakeManager(slow, timeout=0.2)), ["anything"])
    assert time.monotonic() - start < 0.9
    assert len(children) == 1
    assert "RECALLED MEMORY" not in built[0]["ephemeral_system_prompt"]


def test_batch_recalls_run_concurrently():
    def slow(q):
        time.sleep(1.0)
        return f"<memory-context>\n- fact for {q}\n</memory-context>"
    start = time.monotonic()
    children, built = _build(_parent(_FakeManager(slow, timeout=5.0)), ["a", "b", "c"])
    assert time.monotonic() - start < 2.0
    assert len(children) == 3
    for goal, kwargs in zip(["a", "b", "c"], built):
        assert f"fact for {goal}" in kwargs["ephemeral_system_prompt"]
