"""Background-review skill prompts route one-situation lessons to references/ (intake t_a195af93).

The review fork appended per-scenario recipes to the loaded SKILL.md ("PATCH that one first") until
split routers regrew past their size budget. Both skill-writing prompts now describe SKILL.md as a
short router and send situation-specific handling to a topical references/ file. The prompts stay
budget-agnostic: the number lives in the operator's own guard, not in upstream prompt text.
"""

import pytest

from agent import background_review as br


@pytest.mark.parametrize("name", ["_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT"])
def test_review_prompts_route_scenarios_to_references(name):
    prompt = getattr(br, name)
    assert "short router" in prompt
    assert "references/<topic>.md" in prompt
    assert "PATCH that one first" not in prompt
    assert "PATCH it first" not in prompt
    assert "600" not in prompt
