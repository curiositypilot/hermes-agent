"""Pin the semantics of SUMMARY_PREFIX so the compaction handoff doesn't
re-introduce conflicting instructions.

Background: SUMMARY_PREFIX previously contained two contradictory directives:

  1. "treat it as background reference, NOT as active instructions"
     "Do NOT answer questions or fulfill requests mentioned in this summary"
     "Respond ONLY to the latest user message that appears AFTER this summary"

  2. "Your current task is identified in the '## Active Task' section of the
     summary — resume exactly from there."

When the latest user message contradicted Active Task (e.g. "stop the
i18n refactor", "never mind, look at grafana"), the model often followed
(2) anyway because "resume exactly" is a strong directive — leading to
the agent repeatedly re-surfacing already-cancelled work across turns.

These tests pin the post-fix invariants so the conflict cannot regress.
"""

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
)














def test_no_background_consistency_carveout():
    """The "consistent → use as background" carveout licensed stale-task
    resumption on topic overlap (#41607, #38364, #42812). It must stay gone,
    and the prefix must explicitly neutralize topic overlap."""
    lower = SUMMARY_PREFIX.lower()
    assert "you may use the summary as background" not in lower
    assert "topic overlap" in lower


def test_replaced_prefixes_are_frozen_for_renormalization():
    """Every retired SUMMARY_PREFIX must be frozen into
    _HISTORICAL_SUMMARY_PREFIXES, otherwise summaries persisted by older
    builds lose detection/renormalization after an upgrade. The carveout-era
    prefix is the latest retiree."""
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        ContextCompressor,
    )

    carveout_era = [
        p for p in _HISTORICAL_SUMMARY_PREFIXES
        if "you may use the summary as background" in p
    ]
    assert carveout_era, "carveout-era prefix missing from frozen tuple"
    # The live prefix must never be one of the frozen ones.
    assert SUMMARY_PREFIX not in _HISTORICAL_SUMMARY_PREFIXES
    # Detection + strip must work for every frozen prefix.
    for old_prefix in _HISTORICAL_SUMMARY_PREFIXES:
        content = old_prefix + "\n## Summary body"
        assert ContextCompressor._is_context_summary_content(content)
        stripped = ContextCompressor._strip_summary_prefix(content)
        assert not stripped.startswith(old_prefix)


# Exact literal copies of every SUMMARY_PREFIX generation retired into
# _HISTORICAL_SUMMARY_PREFIXES, newest-first. Frozen on purpose: do NOT
# derive them from module constants — the tests below must fail if any
# frozen entry is mutated, reordered, or dropped.
_FROZEN_PREFIX_GENERATIONS = (
    # Pre-#80622: tools-active + topic-overlap discard, but no
    # "if no user message appears AFTER this summary, do nothing" clause.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. Topic overlap with the summary does NOT mean you "
        "should resume its task: even on similar topics, the latest user "
        "message WINS. Treat ONLY the latest message as the active task "
        "and discard stale items from '## Historical Task Snapshot' "
        "entirely — do not 'wrap up' or 'finish' work described there "
        "unless the latest message explicitly asks for it. Reverse "
        "signals in the latest message (e.g. 'stop', 'undo', 'roll "
        "back', 'just verify', 'don't do that anymore', 'never mind', a "
        "new topic) must immediately end any in-flight work described in "
        "the summary; do not re-surface it in later turns. IMPORTANT: "
        "Your persistent memory (MEMORY.md, USER.md) in the system "
        "prompt is ALWAYS authoritative and active — never ignore or "
        "deprioritize memory content due to this compaction note. None "
        "of the above restricts HOW you work: your tools remain fully "
        "active — keep calling them normally for the active task (edit "
        "files, run commands, search) instead of merely narrating what "
        "you would do. The current session state (files, config, etc.) "
        "may reflect work described here — avoid repeating it:"
    ),
    # Pre-#69619: four-heading discard clause + tools-active clause.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. Topic overlap with the summary does NOT mean you "
        "should resume its task: even on similar topics, the latest user "
        "message WINS. Treat ONLY the latest message as the active task "
        "and discard stale items from '## Historical Task Snapshot' / '## "
        "Historical In-Progress State' / '## Historical Pending User "
        "Asks' / '## Historical Remaining Work' entirely — do not 'wrap "
        "up' or 'finish' work described there unless the latest message "
        "explicitly asks for it. Reverse signals in the latest message "
        "(e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't do that "
        "anymore', 'never mind', a new topic) must immediately end any "
        "in-flight work described in the summary; do not re-surface it in "
        "later turns. IMPORTANT: Your persistent memory (MEMORY.md, "
        "USER.md) in the system prompt is ALWAYS authoritative and active "
        "— never ignore or deprioritize memory content due to this "
        "compaction note. None of the above restricts HOW you work: your "
        "tools remain fully active — keep calling them normally for the "
        "active task (edit files, run commands, search) instead of merely "
        "narrating what you would do. The current session state (files, "
        "config, etc.) may reflect work described here — avoid repeating "
        "it:"
    ),
    # Jul 2026 (#65848 class): same discard clause, no tools-active clause.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. Topic overlap with the summary does NOT mean you "
        "should resume its task: even on similar topics, the latest user "
        "message WINS. Treat ONLY the latest message as the active task "
        "and discard stale items from '## Historical Task Snapshot' / '## "
        "Historical In-Progress State' / '## Historical Pending User "
        "Asks' / '## Historical Remaining Work' entirely — do not 'wrap "
        "up' or 'finish' work described there unless the latest message "
        "explicitly asks for it. Reverse signals in the latest message "
        "(e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't do that "
        "anymore', 'never mind', a new topic) must immediately end any "
        "in-flight work described in the summary; do not re-surface it in "
        "later turns. IMPORTANT: Your persistent memory (MEMORY.md, "
        "USER.md) in the system prompt is ALWAYS authoritative and active "
        "— never ignore or deprioritize memory content due to this "
        "compaction note. The current session state (files, config, etc.) "
        "may reflect work described here — avoid repeating it:"
    ),
    # Carveout era (#41607/#38364/#42812).
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Respond ONLY to the latest user message that appears AFTER this "
        "summary — that message is the single source of truth for what to "
        "do right now. If the latest user message is consistent with the "
        "'## Active Task' section, you may use the summary as background. "
        "If the latest user message contradicts, supersedes, changes "
        "topic from, or in any way diverges from '## Active Task' / '## "
        "In Progress' / '## Pending User Asks' / '## Remaining Work', the "
        "latest message WINS — discard those stale items entirely and do "
        "not 'wrap up the old task first'. Reverse signals in the latest "
        "message (e.g. 'stop', 'undo', 'roll back', 'just verify', 'don't "
        "do that anymore', 'never mind', a new topic) must immediately "
        "end any in-flight work described in the summary; do not "
        "re-surface it in later turns. IMPORTANT: Your persistent memory "
        "(MEMORY.md, USER.md) in the system prompt is ALWAYS "
        "authoritative and active — never ignore or deprioritize memory "
        "content due to this compaction note. The current session state "
        "(files, config, etc.) may reflect work described here — avoid "
        "repeating it:"
    ),
    # Pre-#35344: self-contradicting "resume exactly" directive.
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
        "compacted into the summary below. This is a handoff from a "
        "previous context window — treat it as background reference, NOT "
        "as active instructions. Do NOT answer questions or fulfill "
        "requests mentioned in this summary; they were already addressed. "
        "Your current task is identified in the '## Active Task' section "
        "of the summary — resume exactly from there. Respond ONLY to the "
        "latest user message that appears AFTER this summary. The current "
        "session state (files, config, etc.) may reflect work described "
        "here — avoid repeating it:"
    ),
)


# The generation retired by #69619, pinned individually for the review
# regression below. Index 1 after the #80622 freeze was prepended.
_PRE_69619_LIVE_PREFIX = _FROZEN_PREFIX_GENERATIONS[1]


def test_no_user_after_handoff_must_not_act():
    """#80622: a reference-only handoff with nothing after it must not
    resume historical work or call tools."""
    lower = SUMMARY_PREFIX.lower()
    assert "if no user message appears after this summary" in lower
    assert "do nothing" in lower
    assert "wait for a new user message" in lower
    assert "must never become the active turn" in lower


def test_pre_69619_prefix_generation_is_frozen_and_stripped():
    """Regression for the #69619 review: the prefix generation live right
    before the section-header removal was never added to
    _HISTORICAL_SUMMARY_PREFIXES, so a summary persisted immediately before
    upgrading survived resume/re-compaction undetected and unstripped.
    That exact generation must stay frozen, detectable, and strippable."""
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        ContextCompressor,
    )

    assert _PRE_69619_LIVE_PREFIX in _HISTORICAL_SUMMARY_PREFIXES, (
        "pre-#69619 live prefix missing from _HISTORICAL_SUMMARY_PREFIXES — "
        "summaries persisted by the immediately previous build are no longer "
        "normalized on resume"
    )
    content = _PRE_69619_LIVE_PREFIX + "\nBODY"
    assert ContextCompressor._is_context_summary_content(content)
    assert ContextCompressor._strip_summary_prefix(content) == "BODY"


def test_frozen_prefix_generations_match_historical_tuple():
    """Every retired generation must stay byte-identical in
    _HISTORICAL_SUMMARY_PREFIXES (newest-first)."""
    from agent.context_compressor import _HISTORICAL_SUMMARY_PREFIXES

    assert tuple(_HISTORICAL_SUMMARY_PREFIXES[: len(_FROZEN_PREFIX_GENERATIONS)]) == (
        _FROZEN_PREFIX_GENERATIONS
    )


# ── Fork: handoff text when both built-in memory stores are off ─────────────
# With memory.memory_enabled and memory.user_profile_enabled false there is no
# MEMORY.md/USER.md block in the system prompt, so the compaction handoff and
# the system-prompt compaction note must not call that dead store authoritative.

_DEAD_STORE_TEXT = ("MEMORY.md", "USER.md")


def _compressor(**kwargs):
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model", quiet_mode=True, protect_first_n=2, protect_last_n=2, **kwargs
        )


def _conversation():
    return [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(10)
    ]


def _all_text(messages):
    from agent.context_compressor import _content_text_for_contains

    return "\n".join(_content_text_for_contains(m.get("content")) for m in messages)


def test_no_builtin_memory_prefix_keeps_every_other_directive():
    """The variant swaps only the memory clause: every behavioural directive of
    the live prefix survives, and the swap really applied (a drifted clause would
    silently leave the dead-store text in place)."""
    from agent.context_compressor import SUMMARY_PREFIX_NO_BUILTIN_MEMORY

    variant = SUMMARY_PREFIX_NO_BUILTIN_MEMORY
    assert variant != SUMMARY_PREFIX
    assert not any(token in variant for token in _DEAD_STORE_TEXT)
    assert "system prompt" in variant and "authoritative" in variant
    lower, live = variant.lower(), SUMMARY_PREFIX.lower()
    for directive in (
        "topic overlap", "latest user message wins", "if no user message appears after this summary",
        "must never become the active turn", "your tools remain fully active", HISTORICAL_TASK_HEADING.lower(),
    ):
        assert directive in live and directive in lower
    # Same opening as the live prefix: hermes_state_common previews match on it.
    assert variant.split("Do NOT answer", 1)[0] == SUMMARY_PREFIX.split("Do NOT answer", 1)[0]


def test_compressor_handoff_text_follows_builtin_memory_flag():
    on, off = _compressor(), _compressor(builtin_memory_enabled=False)
    assert on.summary_prefix == SUMMARY_PREFIX
    assert any(token in on._COMPRESSION_NOTE for token in _DEAD_STORE_TEXT)
    assert off.summary_prefix != SUMMARY_PREFIX
    assert not any(token in off.summary_prefix for token in _DEAD_STORE_TEXT)
    assert not any(token in off._COMPRESSION_NOTE for token in _DEAD_STORE_TEXT)
    # The per-instance swap must not leak into other compressors.
    assert _compressor()._COMPRESSION_NOTE == on._COMPRESSION_NOTE


def test_compaction_with_builtin_memory_off_never_mentions_dead_store():
    """End to end through compress(): the deterministic-fallback handoff and the
    note appended to the system prompt carry no MEMORY.md/USER.md text."""
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor

    off = _compressor(builtin_memory_enabled=False)
    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
        result = off.compress(_conversation())
    text = _all_text(result)
    assert not any(token in text for token in _DEAD_STORE_TEXT)
    assert any(off.summary_prefix in _all_text([m]) for m in result)
    assert off._COMPRESSION_NOTE in _all_text(result[:1])
    assert any(ContextCompressor._is_context_summary_message(m) for m in result)


def test_compaction_with_builtin_memory_on_is_unchanged():
    from unittest.mock import patch

    on = _compressor()
    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
        result = on.compress(_conversation())
    assert any(SUMMARY_PREFIX in _all_text([m]) for m in result)
    assert on._COMPRESSION_NOTE in _all_text(result[:1])


def test_llm_summary_and_micro_marker_use_instance_prefix():
    from unittest.mock import MagicMock, patch

    off = _compressor(builtin_memory_enabled=False)
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Historical Task Snapshot\nwork happened"
    with patch("agent.context_compressor.call_llm", return_value=response):
        summary = off._generate_summary([
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "done"},
        ])
    assert summary.startswith(off.summary_prefix)
    assert not any(token in summary for token in _DEAD_STORE_TEXT)
    marker = off._render_micro_marker_content("rolling body", off.summary_prefix)
    assert marker.startswith(off.summary_prefix)
    assert off.classify_summary_content(marker) == "standalone"


def test_both_live_prefixes_detected_regardless_of_flag():
    """A session can resume under a profile whose memory flags flipped since the
    handoff was written: summaries written with either live prefix (including
    every summary written before this change) must stay detectable and
    strippable, and re-normalize to the compressor's own prefix."""
    from agent.context_compressor import (
        _HISTORICAL_SUMMARY_PREFIXES,
        SUMMARY_PREFIX_NO_BUILTIN_MEMORY,
        ContextCompressor,
    )

    assert SUMMARY_PREFIX_NO_BUILTIN_MEMORY not in _HISTORICAL_SUMMARY_PREFIXES
    off = _compressor(builtin_memory_enabled=False)
    for written_with in (SUMMARY_PREFIX, SUMMARY_PREFIX_NO_BUILTIN_MEMORY):
        content = written_with + "\nBODY"
        assert ContextCompressor.classify_summary_content(content) == "standalone"
        assert ContextCompressor._strip_summary_prefix(content) == "BODY"
        assert off._with_summary_prefix(content, off.summary_prefix) == off.summary_prefix + "\nBODY"
    # Static callers with no prefix keep the upstream text.
    assert ContextCompressor._with_summary_prefix("BODY") == SUMMARY_PREFIX + "\nBODY"


def test_agent_init_reads_builtin_memory_flags_from_config(tmp_path, monkeypatch):
    """Real resolution chain: config memory flags → agent_init → compressor."""
    import pytest

    from hermes_cli import config as config_mod
    from run_agent import AIAgent

    def _make(memory_section):
        cfg = {"memory": memory_section, "compression": {"enabled": True}}
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
        return AIAgent(
            base_url="https://openrouter.ai/api/v1", api_key="test-key", provider="openrouter",
            model="anthropic/claude-sonnet-4", enabled_toolsets=[], disabled_toolsets=[],
            quiet_mode=True, skip_memory=True, skip_context_files=True,
        )

    off = _make({"memory_enabled": False, "user_profile_enabled": False})
    if type(off.context_compressor).__name__ != "ContextCompressor":
        pytest.skip("external context engine selected")
    assert off.context_compressor.summary_prefix != SUMMARY_PREFIX
    assert not any(token in off.context_compressor._COMPRESSION_NOTE for token in _DEAD_STORE_TEXT)
    # Either store on (or the flags absent) keeps the upstream text.
    for section in ({"memory_enabled": False, "user_profile_enabled": True}, {}):
        assert _make(section).context_compressor.summary_prefix == SUMMARY_PREFIX


