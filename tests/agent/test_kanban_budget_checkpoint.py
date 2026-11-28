"""Checkpoint eligibility follows explicit opt-in or dispatcher ownership."""
from contextlib import nullcontext
from copy import deepcopy

import pytest
from tests.agent.test_iteration_budget_warning import _agent


@pytest.mark.parametrize("ratio,scope,expected,kanban_notice", [
    *[(ratio, "ordinary", False, False) for ratio in ("null", "0", "junk")],
    ("0.75", "ordinary", True, False),
    ("null", "owner", True, True),
    ("null", "non-owner", False, False),
    ("null", "child", False, False),
    ("0.75", "child", True, False),
    ("null", "no-completion-tool", False, False),
])
def test_checkpoint_requires_opt_in_or_dispatcher_completion_scope(
    tmp_path, monkeypatch, ratio, scope, expected, kanban_notice
):
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    from agent.turn_iteration_prep import prepare_iteration

    if scope != "ordinary":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_checkpoint")
        # The dispatcher stamps every worker's data class (kanban_db_dispatch); without it the
        # provider policy looks the fake task up on the board and fails closed.
        monkeypatch.setenv("HERMES_DATA_CLASS", "internal")
    contexts = {"non-owner": non_dispatcher_owned_context, "child": delegated_child_context}
    with contexts.get(scope, nullcontext)():
        agent = _agent(tmp_path, monkeypatch, ratio)
        try:
            if scope == "no-completion-tool":
                agent.valid_tool_names.discard("kanban_complete")
            for _ in range(3):
                agent.iteration_budget.consume()
            messages = [{"role": "user", "content": "work"},
                        {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "t", "content": "verified artifact"}]
            original = deepcopy(messages)
            prepare_iteration(agent, messages=messages, api_call_count=3)
            assert messages[:-1] == original[:-1]
            assert (messages[-1]["content"] != "verified artifact") is expected
            assert ("kanban_complete" in messages[-1]["content"]) is kanban_notice
            snapshot = deepcopy(messages)
            prepare_iteration(agent, messages=messages, api_call_count=3)
            assert messages == snapshot
            assert agent.iteration_budget.remaining == 1
        finally:
            agent._session_db.close()


def _tool_tail(content="verified artifact"):
    return [{"role": "user", "content": "work"},
            {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "execute_code", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t", "content": content}]


@pytest.fixture
def owner_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_checkpoint")
    monkeypatch.setenv("HERMES_DATA_CLASS", "internal")
    agent = _agent(tmp_path, monkeypatch, "null")
    yield agent
    agent._session_db.close()


def test_checkpoint_counts_api_calls_when_execute_code_refunds(owner_agent):
    """execute_code-only rounds refund the budget, but the loop still stops at max_iterations."""
    from agent.turn_iteration_prep import prepare_iteration

    assert owner_agent.iteration_budget.used == 0
    early = _tool_tail()
    prepare_iteration(owner_agent, messages=early, api_call_count=2)
    assert early[-1]["content"] == "verified artifact"
    messages = _tool_tail()
    prepare_iteration(owner_agent, messages=messages, api_call_count=3)
    assert "3 of 4" in messages[-1]["content"]
    assert "kanban_complete" in messages[-1]["content"]


def test_checkpoint_ignores_previous_turn_api_call_count(owner_agent):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration

    owner_agent._api_call_count = 3  # left over from the previous turn
    _reset_per_turn_agent_state(owner_agent)
    messages = _tool_tail()
    prepare_iteration(owner_agent, messages=messages, api_call_count=0)
    assert messages[-1]["content"] == "verified artifact"


# (api call, budget.used) at the last three API calls of the capped Kanban sessions of
# Nov 2026 (t_f384fd0a spec), rebuilt from state.db; eee37f9d is the control that got a notice.
_NOV_2026_CAP_HITS = [
    (135, 20), (149, 20), (150, 21),     # 714f0182
    (135, 51), (149, 51), (150, 51),     # f74098cc
    (135, 107), (149, 116), (150, 117),  # 33f31445
    (135, 135), (149, 149), (150, 150),  # eee37f9d (control)
]


@pytest.mark.parametrize("api_calls,budget_used", _NOV_2026_CAP_HITS)
def test_checkpoint_replays_nov_2026_cap_hits(monkeypatch, api_calls, budget_used):
    from types import SimpleNamespace
    from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_checkpoint")
    monkeypatch.setattr("agent.delegation_context.is_dispatcher_owned_worker_context", lambda: True)
    agent = SimpleNamespace(
        budget_warning_ratio=None, valid_tool_names={"kanban_complete"}, max_iterations=150,
        iteration_budget=SimpleNamespace(max_total=150, used=budget_used),
        _api_call_count=api_calls, _iteration_budget_warning_injected=False,
    )
    messages = _tool_tail()
    assert _maybe_inject_iteration_budget_warning(agent, messages)
    assert f"{max(api_calls, budget_used)} of 150" in messages[-1]["content"]
    assert "kanban_complete" in messages[-1]["content"]
