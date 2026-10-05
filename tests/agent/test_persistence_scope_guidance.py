"""Prompt contracts for scoped persistence, not semantic write validation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import background_review
from agent.prompt_builder import MEMORY_GUIDANCE, PERSISTENCE_SCOPE_GUIDANCE
from tools.memory_tool import MEMORY_SCHEMA


@pytest.mark.parametrize("clause", [
    "Classify scope before every save",
    "stable operating context only when applicable across projects",
    "not automatically a global preference",
    "project-specific preferences and constraints",
    "hardware inventory/configuration",
    "canonical project docs; use work logs",
    "Do not save project instance facts to memory or to skills, including skill reference files",
    "save reusable HOW",
    "Split a global preference from its project evidence",
    "Link to canonical docs when needed instead of duplicating",
    "Do not create a fresh per-project memory pointer for every offload",
    "update the canonical owner and replace duplicate summaries with links rather than syncing details",
    "Indexes contain only name, link, and short purpose — not specs or progress",
])
def test_scope_policy_covers_each_destination_and_deduplication(clause):
    assert clause in PERSISTENCE_SCOPE_GUIDANCE
    assert PERSISTENCE_SCOPE_GUIDANCE in MEMORY_GUIDANCE


def test_foreground_examples_are_global_declarative_facts():
    assert "'User prefers metric units across projects' ✓" in MEMORY_GUIDANCE
    assert "Project uses pytest with xdist" not in MEMORY_GUIDANCE
    assert "Write memories as declarative facts, not instructions" in MEMORY_GUIDANCE


@pytest.mark.parametrize("review_memory,review_skills,prompt_name", [
    (True, False, "_MEMORY_REVIEW_PROMPT"),
    (False, True, "_SKILL_REVIEW_PROMPT"),
    (True, True, "_COMBINED_REVIEW_PROMPT"),
])
@pytest.mark.parametrize("focus", [None, "retain the useful lessons from this project"])
def test_all_review_and_refine_writer_paths_receive_scope_guidance(
    review_memory, review_skills, prompt_name, focus,
):
    # Use the real selection/closure, not source-text matching. The provider-facing
    # worker is intercepted here; runtime whitelist coverage lives alongside it.
    history = [{"role": "user", "content": "test conversation"}]
    agent = SimpleNamespace()
    target, prompt = background_review.spawn_background_review_thread(
        agent, history, review_memory=review_memory, review_skills=review_skills,
        focus=focus, task_cfg={},
    )
    assert prompt.startswith(getattr(background_review, prompt_name))
    assert PERSISTENCE_SCOPE_GUIDANCE in prompt
    assert "skip those project facts rather than using the wrong store" in prompt
    assert "Do not broaden permissions or attempt other tools" in prompt
    assert "Continue evaluating global user preferences and reusable skills independently" in prompt
    assert "skipped project facts are not a reason to say 'Nothing to save.'" in prompt
    assert "Never read or write secrets" in prompt
    assert "for session-specific detail" not in prompt
    assert "current situation and state of your operations" not in prompt
    if focus:
        assert focus in prompt
        assert "within the scope and tool restrictions" in prompt
        assert "does not authorize project facts in memory or skills" in prompt
    with patch.object(background_review, "_run_review_in_thread") as worker:
        target()
    worker.assert_called_once_with(agent, history, prompt, {})


def test_live_memory_schema_does_not_recommend_a_conflicting_store():
    description = MEMORY_SCHEMA["description"]
    assert "classify scope before saving" in description
    assert "stable operating context only when applicable across projects" in description
    assert "canonical project docs, not memory or skill reference files" in description
    assert "correct store is unavailable, skip those facts" in description
    assert "replace duplicate summaries with links rather than syncing details" in description
    assert "not specs/progress" in description
    assert "Do not create a fresh per-project memory pointer" in description
    assert "procedures belong in a skill, not memory" in description
    assert "Priority: user preferences & corrections > environment facts > procedures" not in description
