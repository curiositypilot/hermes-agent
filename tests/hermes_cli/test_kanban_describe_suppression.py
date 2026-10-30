"""``describe_suppression`` output survives the log redactor.

Regression: the "dispatcher stuck" warning rendered ``blocker_auth=1, quota_hold=2``
and ``agent.redact`` read ``blocker_auth=1,`` as a lowercase ``*_auth=`` env secret, so
gateway.log showed ``blocker_auth=*** quota_hold=2`` and the held-back count was lost.
"""
from __future__ import annotations

from agent.redact import redact_sensitive_text
from hermes_cli import kanban_db_dispatch as kbd

GUARD_REASONS = (
    "active_pr",
    "blocker_auth",
    "infrastructure_cooldown",
    "quota_hold",
    "rate_limit_cooldown",
    "recent_success",
    "tier_exhausted",
)


def _held_line() -> str:
    res = kbd.DispatchResult(
        respawn_guarded=[(f"t_{i:08x}", reason) for i, reason in enumerate(GUARD_REASONS)],
        rate_limited=["t_rl000001"],
        skipped_locked=True,
        memory_pressure="critical",
    )
    return kbd.describe_suppression([res])


def test_every_reason_passes_the_redactor_unchanged():
    held = _held_line()
    line = f" Last tick held back: {held}."
    assert "blocker_auth (1)" in held
    assert "memory_pressure (critical)" in held
    for reason in (*GUARD_REASONS, "rate_limited", "skipped_locked"):
        assert f"{reason} (1)" in held
    assert redact_sensitive_text(line, force=True) == line


def test_redactor_still_masks_a_real_auth_assignment():
    assert "1234" not in redact_sensitive_text("db_auth=1234 x", force=True)
