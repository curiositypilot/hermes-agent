"""Tests for system-prompt model-identity sync across provider failover.

The system prompt is session-stable and embeds ``Model:``/``Provider:``
identity lines.  When ``try_activate_fallback`` swaps the runtime, the
prompt must be rewritten in place (and synced into the in-flight
``api_messages``) or the agent reports the primary model's name while a
fallback model is answering — e.g. a local gemma fallback claiming to be
gpt-5.4-mini after a Codex usage-limit 429.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import rewrite_prompt_model_identity
from agent.conversation_loop import (
    _redecorate_prompt_cache_for_provider,
    _sync_failover_system_message,
)
from agent.prompt_caching import apply_anthropic_cache_control


_PROMPT = (
    "You are a helpful assistant.\n"
    "\n"
    "Memory note at line start:\n"
    "Model: decoy-from-memory\n"
    "\n"
    "Conversation started: Wednesday, June 10, 2026\n"
    "Model: gpt-5.4-mini\n"
    "Provider: openai-codex"
)


def _agent(prompt=_PROMPT, ephemeral=None):
    return SimpleNamespace(
        _cached_system_prompt=prompt,
        ephemeral_system_prompt=ephemeral,
    )


def _count_cache_markers(messages):
    count = 0
    for msg in messages:
        if "cache_control" in msg:
            count += 1
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "cache_control" in part:
                    count += 1
    return count


def _cache_agent(
    *,
    use_caching,
    native=False,
    prompt=_PROMPT,
    static=None,
    cache_ttl="5m",
    provider="openai",
    model="gpt-4o",
    tools=None,
    direct_tool_cache=False,
):
    return SimpleNamespace(
        _cached_system_prompt=prompt,
        _cached_system_prompt_static=static,
        ephemeral_system_prompt=None,
        _use_prompt_caching=use_caching,
        _use_native_cache_layout=native,
        _cache_ttl=cache_ttl,
        provider=provider,
        model=model,
        tools=tools or [],
        _direct_native_anthropic_tool_cache_capability=lambda: direct_tool_cache,
        client=None,
    )


class TestRewritePromptModelIdentity:

    def test_only_last_occurrence_is_rewritten(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        # Earlier matching lines may be user content (memory snapshots,
        # context files) and must survive untouched.
        assert "Model: decoy-from-memory" in agent._cached_system_prompt


    def test_noop_when_prompt_missing_or_empty(self):
        for prompt in (None, ""):
            agent = _agent(prompt=prompt)
            rewrite_prompt_model_identity(agent, "m", "p")
            assert agent._cached_system_prompt == prompt



class TestSyncFailoverSystemMessage:
    def test_patches_in_flight_system_message(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        api_messages = [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": "what model are you?"},
        ]
        result = _sync_failover_system_message(agent, api_messages, _PROMPT)
        assert "Model: gemma4:e2b-mlx" in api_messages[0]["content"]
        assert result == agent._cached_system_prompt


    def test_noop_without_cached_prompt(self):
        agent = _agent(prompt=None)
        api_messages = [{"role": "system", "content": "original"}]
        result = _sync_failover_system_message(agent, api_messages, "active")
        assert api_messages[0]["content"] == "original"
        assert result == "active"



class TestSyncFailoverPreservesCacheDecoration:
    """The sync must not flatten a cache-decorated system message.

    ``apply_anthropic_cache_control`` runs once per call block, before the retry
    loop, splitting the system prompt into ``[static prefix, volatile tail]``
    blocks that carry the cache_control breakpoints. A failover fires *inside*
    that retry loop, so overwriting the list with a bare string drops both
    breakpoints and the retried request re-bills the whole system prompt.
    """

    _STATIC = "You are a helpful assistant.\n\nStable brief.\n"

    def _decorated(self, prompt):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "what model are you?"},
        ]
        return apply_anthropic_cache_control(
            messages,
            cache_ttl=None,
            native_anthropic=True,
            static_system_prefix=self._STATIC,
        )

    def test_keeps_breakpoints_and_static_prefix(self):
        prompt = self._STATIC + "Model: gpt-5.4-mini\nProvider: openai-codex"
        agent = _agent(prompt=prompt)
        api_messages = self._decorated(prompt)
        assert isinstance(api_messages[0]["content"], list)

        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        _sync_failover_system_message(agent, api_messages, prompt)

        content = api_messages[0]["content"]
        assert isinstance(content, list), "cache decoration was flattened to a string"
        assert len(content) == 2
        assert all(part.get("cache_control") for part in content), (
            "the failover retry would ship zero system cache breakpoints"
        )
        # The static prefix must stay byte-identical or its cache entry misses.
        assert content[0]["text"] == self._STATIC
        # The identity refresh still lands, in the volatile tail.
        assert "Model: gemma4:e2b-mlx" in content[1]["text"]
        assert "Provider: custom" in content[1]["text"]

    def test_keeps_single_block_shape(self):
        prompt = "Model: gpt-5.4-mini\nProvider: openai-codex"
        agent = _agent(prompt=prompt)
        # No static prefix match -> the single-block fallback layout.
        api_messages = apply_anthropic_cache_control(
            [{"role": "system", "content": prompt}],
            cache_ttl=None,
            native_anthropic=True,
            static_system_prefix=None,
        )
        assert isinstance(api_messages[0]["content"], list)
        assert len(api_messages[0]["content"]) == 1

        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        _sync_failover_system_message(agent, api_messages, prompt)

        content = api_messages[0]["content"]
        assert isinstance(content, list) and len(content) == 1
        assert content[0].get("cache_control")
        assert "Model: gemma4:e2b-mlx" in content[0]["text"]




class TestRedecoratePromptCacheOnPolicyChange:
    """#72626: failover must re-render breakpoints for the *new* cache policy.

    ``TestSyncFailoverPreservesCacheDecoration`` only covers same-policy
    identity sync (native→native). These cases cover the policy-change class.
    """

    _STATIC = "You are a helpful assistant.\n\nStable brief.\n"

    @pytest.mark.parametrize(
        "use_caching,native,provider",
        [
            pytest.param(False, False, "openai", id="cache-off"),
            pytest.param(True, True, "anthropic", id="native"),
        ],
    )
    def test_optional_tools_return_a_complete_provider_plan(
        self, use_caching, native, provider
    ):
        prompt = self._STATIC + "volatile"
        messages = apply_anthropic_cache_control(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "hello"}],
            native_anthropic=True,
            static_system_prefix=self._STATIC,
        )
        registered_tools = [
            {
                "type": "function",
                "function": {"name": "registered", "parameters": {"type": "object"}},
                "cache_control": {"type": "ephemeral"},
            }
        ]
        agent = _cache_agent(
            use_caching=use_caching,
            native=native,
            prompt=prompt,
            static=self._STATIC,
            provider=provider,
            tools=registered_tools,
            direct_tool_cache=native,
        )
        before = deepcopy((messages, registered_tools))

        decorated, _prepared, planned_tools = _redecorate_prompt_cache_for_provider(
            agent, messages
        )

        assert [t["function"] for t in planned_tools] == [t["function"] for t in registered_tools]
        assert (_count_cache_markers(decorated) > 0) == use_caching
        assert any("cache_control" in t for t in planned_tools) == native
        assert (messages, registered_tools) == before

    def test_cache_off_to_cache_on_adds_breakpoints(self):
        prompt = self._STATIC + "Model: gpt-5.4-mini\nProvider: openai"
        # Primary never decorated (cache-off).
        undecorated = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        assert _count_cache_markers(undecorated) == 0

        agent = _cache_agent(
            use_caching=True,
            native=True,
            prompt=prompt,
            static=self._STATIC,
            provider="anthropic",
        )
        decorated, _, _ = _redecorate_prompt_cache_for_provider(agent, undecorated)
        assert _count_cache_markers(decorated) >= 2
        assert isinstance(decorated[0]["content"], list)
        assert decorated[0]["content"][0]["text"] == self._STATIC

    def test_replans_tools_for_the_active_destination(self):
        from agent.prompt_caching import build_prompt_cache_plan

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
        messages = [
            {"role": "system", "content": self._STATIC + "volatile"},
            {"role": "user", "content": "lookup"},
        ]
        source = build_prompt_cache_plan(
            messages,
            tools,
            native_anthropic=True,
            static_system_prefix=self._STATIC,
            direct_native_tool_cache=True,
        )
        agent = _cache_agent(
            use_caching=True,
            native=False,
            prompt=messages[0]["content"],
            static=self._STATIC,
            provider="openrouter",
            tools=tools,
            direct_tool_cache=False,
        )

        fallback_messages, _, fallback_tools = _redecorate_prompt_cache_for_provider(
            agent,
            source.messages,
            tools_for_api=source.tools,
        )

        assert "cache_control" in source.tools[-1]
        assert "cache_control" not in fallback_tools[-1]
        assert "cache_control" not in tools[-1]
        assert all(
            part.get("cache_control")
            for part in fallback_messages[0]["content"]
            if isinstance(part, dict)
        )




    def test_moa_guidance_stays_outside_last_breakpoint(self):
        prompt = "sys"
        guidance = (
            "[Mixture of Agents context — use this as private guidance for the "
            "normal Hermes agent loop.]\nAggregator: agg\n\nadvice"
        )
        base = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "task"},
            {"role": "user", "content": guidance},
        ]
        decorated = apply_anthropic_cache_control(base, native_anthropic=True)

        class _Completions:
            def rebase_prepared_request(self, prepared, messages):
                from agent.moa_loop import _attach_reference_guidance

                out = [dict(m) for m in messages]
                _attach_reference_guidance(out, prepared["guidance"])
                return {**prepared, "messages": out}

        agent = _cache_agent(use_caching=True, native=True, prompt=prompt, provider="moa")
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        prepared = {"guidance": guidance, "messages": decorated}
        out, new_prepared, planned_tools = _redecorate_prompt_cache_for_provider(
            agent, decorated, moa_prepared=prepared
        )
        assert new_prepared is not None
        assert planned_tools == []
        # Guidance must be present, but the cache marker on the last user
        # turn's content parts must terminate *before* the guidance text.
        last = out[-1]
        assert last.get("role") == "user"
        content = last["content"]
        if isinstance(content, list):
            marked = [p for p in content if isinstance(p, dict) and "cache_control" in p]
            guidance_parts = [
                p for p in content
                if isinstance(p, dict) and guidance in (p.get("text") or "")
            ]
            assert marked, "base transcript should still carry breakpoints"
            assert guidance_parts, "guidance must be re-attached"
            # Marked part should not contain the guidance body.
            assert all(guidance not in (p.get("text") or "") for p in marked)
        else:
            assert guidance in content




class TestPeelReferenceGuidanceRoundTrip:
    """peel must invert every attach shape — the two live adjacent in
    moa_loop.py precisely so this contract can't drift silently."""

    _GUIDANCE = "[Mixture of Agents reference context]\nAdvice body."

    def _round_trip(self, base):
        import copy

        from agent.moa_loop import _attach_reference_guidance, peel_reference_guidance

        attached = copy.deepcopy(base)
        _attach_reference_guidance(attached, self._GUIDANCE)
        assert attached != base, "attach must change the transcript"
        return peel_reference_guidance(attached, self._GUIDANCE)


    def test_list_part_shape(self):
        base = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "task", "cache_control": {"type": "ephemeral"}}],
            },
        ]
        assert self._round_trip(base) == base


    def test_guidance_only_part_drops_message_not_empty_residue(self):
        # If the guidance part is the only content left after peeling, the
        # whole message goes — an empty-content user turn must never remain.
        from agent.moa_loop import peel_reference_guidance

        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [{"type": "text", "text": self._GUIDANCE}],
            },
        ]
        peeled = peel_reference_guidance(messages, self._GUIDANCE)
        assert peeled == messages[:-1]


# ── Model-gated guidance follows the active model across failover (t_f58d6d0a) ──

_GOOGLE = "# Google model operational directives"
_ENFORCE = "# Tool-use enforcement"
_EXEC = "# Execution discipline"
_PAIRS = (
    ("gemini-3.8-flash-tiered", "gemini", "claude-sonnet-5-5", "anthropic"),
    ("grok-4.7", "xai", "claude-sonnet-5-5", "anthropic"),
    ("claude-opus-5-5", "anthropic", "grok-4.7", "xai"),
)


def _guidance_agent(model, **overrides):
    base = dict(
        model=model, valid_tool_names={"terminal", "read_file"}, _oneshot_session=False,
        _kanban_worker_guidance="", _tool_use_enforcement="auto", _execution_guidance="auto",
        _cached_system_prompt=None, _cached_system_prompt_static=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _built(model, provider, *, memory="Memory: user prefers short answers.", labels=True, **overrides):
    """(agent, prompt, static) with the stable tier from the real ``_guidance_parts``."""
    from agent.system_prompt import _guidance_parts, _join_tier

    agent = _guidance_agent(model, **overrides)
    static = _join_tier(["You are Hermes.", *_guidance_parts(agent)])
    tail = f"\n\nConversation started: Tuesday\nModel: {model}\nProvider: {provider}" if labels else ""
    prompt = f"{static}\n\n{memory}{tail}"
    agent._cached_system_prompt, agent._cached_system_prompt_static = prompt, static
    return agent, prompt, static


def _assert_static_is_prefix(agent):
    static = agent._cached_system_prompt_static
    assert static is None or agent._cached_system_prompt.startswith(static)


def test_fallback_swaps_model_family_guidance():
    agent, prompt, _ = _built("gemini-3.8-flash-tiered", "gemini")
    assert _ENFORCE in prompt and _GOOGLE in prompt
    rewrite_prompt_model_identity(agent, "claude-sonnet-5-5", "anthropic")
    assert _ENFORCE not in agent._cached_system_prompt
    assert _GOOGLE not in agent._cached_system_prompt

    agent, prompt, _ = _built("grok-4.7", "xai")
    assert _EXEC in prompt
    rewrite_prompt_model_identity(agent, "claude-sonnet-5-5", "anthropic")
    assert _EXEC not in agent._cached_system_prompt

    # Every pair lands on exactly what a fresh build for the fallback gives.
    for primary, p_provider, fallback, f_provider in _PAIRS:
        agent, _, _ = _built(primary, p_provider)
        rewrite_prompt_model_identity(agent, fallback, f_provider)
        _, fresh, fresh_static = _built(fallback, f_provider)
        assert agent._cached_system_prompt == fresh
        assert agent._cached_system_prompt_static == fresh_static


def test_guidance_swap_round_trip_is_byte_identical():
    for primary, p_provider, fallback, f_provider in _PAIRS:
        agent, prompt, static = _built(primary, p_provider)
        rewrite_prompt_model_identity(agent, fallback, f_provider)
        assert agent._cached_system_prompt != prompt
        _assert_static_is_prefix(agent)
        rewrite_prompt_model_identity(agent, primary, p_provider)
        assert agent._cached_system_prompt == prompt
        assert agent._cached_system_prompt_static == static
        _assert_static_is_prefix(agent)


def test_guidance_swap_noop_cases():
    from agent.prompt_builder import STEER_CHANNEL_NOTE

    def relabelled(prompt, old, new, old_p, new_p):
        return prompt.replace(f"Model: {old}\nProvider: {old_p}", f"Model: {new}\nProvider: {new_p}")

    # Same family: guidance bytes unchanged, labels rewritten.
    agent, prompt, static = _built("claude-opus-5-5", "anthropic")
    rewrite_prompt_model_identity(agent, "claude-sonnet-5-5", "anthropic")
    assert agent._cached_system_prompt == relabelled(prompt, "claude-opus-5-5", "claude-sonnet-5-5",
                                                     "anthropic", "anthropic")
    assert agent._cached_system_prompt_static == static

    # The anchor also in a memory section: ambiguous, so guidance stays; labels still move.
    agent, prompt, static = _built("gemini-3.8-flash-tiered", "gemini", memory=STEER_CHANNEL_NOTE)
    rewrite_prompt_model_identity(agent, "claude-sonnet-5-5", "anthropic")
    assert agent._cached_system_prompt == relabelled(prompt, "gemini-3.8-flash-tiered", "claude-sonnet-5-5",
                                                     "gemini", "anthropic")
    assert _GOOGLE in agent._cached_system_prompt
    assert agent._cached_system_prompt_static == static

    # No Model: line: nothing to read the old model from.
    agent, prompt, static = _built("gemini-3.8-flash-tiered", "gemini", labels=False)
    rewrite_prompt_model_identity(agent, "claude-sonnet-5-5", "anthropic")
    assert agent._cached_system_prompt == prompt
    assert agent._cached_system_prompt_static == static

    # Both gates forced off: no block is inserted for grok.
    agent, prompt, _ = _built("claude-opus-5-5", "anthropic", _tool_use_enforcement=False,
                              _execution_guidance=False)
    rewrite_prompt_model_identity(agent, "grok-4.7", "xai")
    assert agent._cached_system_prompt == relabelled(prompt, "claude-opus-5-5", "grok-4.7", "anthropic", "xai")
    assert _ENFORCE not in agent._cached_system_prompt and _EXEC not in agent._cached_system_prompt

    # No tools: guidance unchanged.
    agent, prompt, static = _built("gemini-3.8-flash-tiered", "gemini")
    agent.valid_tool_names = set()
    rewrite_prompt_model_identity(agent, "claude-sonnet-5-5", "anthropic")
    assert agent._cached_system_prompt == relabelled(prompt, "gemini-3.8-flash-tiered", "claude-sonnet-5-5",
                                                     "gemini", "anthropic")
    assert agent._cached_system_prompt_static == static
