"""Unattended sessions drop prompt blocks nobody there can use.

The Hermes-docs help line is for a user asking about Hermes: cron, subagents, delegated children,
one-shots and dispatcher-owned Kanban workers drop it, chats keep it. The mid-turn steering briefing
is dropped for cron (no steer path reaches a cron run); workers and subagents receive steers and
keep it. ``owned_kanban_task`` is pinned in every test so the reviewer's own worker env never leaks in.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

STEER_MARKER = "OUT-OF-BAND USER MESSAGE"
HELP_MARKER = "nousresearch.com/docs"


def _agent(platform, *, depth=0, oneshot=False, tools=("terminal",)):
    return SimpleNamespace(
        load_soul_identity=False, skip_context_files=True, valid_tool_names=set(tools),
        _task_completion_guidance=False, _parallel_tool_call_guidance=False, _tool_use_enforcement=False,
        _execution_guidance=False, _environment_probe=False, _kanban_worker_guidance="",
        _memory_store=None, _memory_manager=None, model="m", provider="", platform=platform,
        pass_session_id=False, session_id="", _delegate_depth=depth, _oneshot_session=oneshot,
        _emit_status=lambda *_a, **_k: None,
    )


@pytest.fixture
def owned_task(monkeypatch):
    task = {"id": ""}
    monkeypatch.setattr("agent.delegation_context.owned_kanban_task", lambda: task["id"])
    return task


def _guidance(agent):
    from agent import system_prompt

    return "\n".join(p for p in system_prompt._guidance_parts(agent) if p)


def _stable(agent, skills_prompt=""):
    from agent import system_prompt

    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
        patch.object(system_prompt, "_skills_prompt", return_value=skills_prompt),
    ):
        return system_prompt.build_system_prompt_parts(agent)["stable"]


def test_cron_drops_steering_note(owned_task):
    assert STEER_MARKER not in _guidance(_agent("cron"))


@pytest.mark.parametrize("platform, task", [("telegram", ""), ("cli", ""), ("subagent", ""), ("", "t_x")])
def test_steerable_sessions_keep_steering_note(owned_task, platform, task):
    owned_task["id"] = task
    assert STEER_MARKER in _guidance(_agent(platform))


@pytest.mark.parametrize("platform, depth, oneshot, task", [
    ("cron", 0, False, ""),
    ("subagent", 0, False, ""),
    ("cli", 1, False, ""),
    ("cli", 0, True, ""),
    ("cli", 0, False, "t_x"),
])
def test_unattended_sessions_drop_help_line(owned_task, platform, depth, oneshot, task):
    owned_task["id"] = task
    assert HELP_MARKER not in _stable(_agent(platform, depth=depth, oneshot=oneshot))


@pytest.mark.parametrize("platform", ["telegram", "cli", "desktop"])
def test_chats_keep_help_line(owned_task, platform):
    assert HELP_MARKER in _stable(_agent(platform))


def test_chat_with_hermes_agent_skill_keeps_skill_variant(owned_task):
    from agent.prompt_builder import HERMES_AGENT_HELP_GUIDANCE

    stable = _stable(_agent("telegram", tools=("terminal", "skill_view")),
                     skills_prompt="- hermes-agent: Hermes help")
    assert HERMES_AGENT_HELP_GUIDANCE.strip() in stable
