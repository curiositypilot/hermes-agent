"""Reply turns key memory recall on the user's own words first, then the tail of the quote.

``_prepend_inbound_reply_context`` puts ``[Replying to: "<quote>"]`` AHEAD of the user's text in
the model-facing message. Memory providers head-truncate their recall query (Hindsight
``recall_max_input_chars`` = 800), so a reply to a long message keyed recall on the quote alone and
dropped what the user actually asked. The gateway now passes a ``memory_query`` (the seam cron
already uses) built user-text-first; the model-facing message is untouched.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_inbound import REPLY_MEMORY_QUERY_CHARS, discord_triggering_note, reply_memory_query
from gateway.session import SessionSource


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    return runner


def _source(platform=Platform.TELEGRAM) -> SessionSource:
    return SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1", user_name="MB")


def _reply_event(user_text: str, quote: str, **kw) -> MessageEvent:
    return MessageEvent(text=user_text, source=_source(), message_id="m2", reply_to_message_id="m1",
                        reply_to_text=quote, **kw)


# ── the helper ────────────────────────────────────────────────────────────────


def test_user_text_leads_and_the_quote_tail_fills_the_budget():
    quote = "".join(f"line {i:04d}\n" for i in range(300)).strip()  # ~3.3k chars, like a long bot reply
    user_text = "what did you mean by the third point?"
    envelope = f'[Replying to: "{quote}"]\n\n{user_text}'

    query = reply_memory_query(_reply_event(user_text, quote), envelope)

    assert query.startswith(user_text + "\n\n")
    assert len(query) <= REPLY_MEMORY_QUERY_CHARS
    # The provider head-truncates at the same budget, so the user's words always survive the cut.
    assert query[:REPLY_MEMORY_QUERY_CHARS].startswith(user_text)
    # The quote contributes its TAIL (the most recent part of the quoted message), not its head.
    assert query.endswith(quote[-(REPLY_MEMORY_QUERY_CHARS - len(user_text) - 2):])
    assert "line 0000" not in query


def test_short_quote_rides_whole_behind_the_user_text():
    quote = "Create a project plan for Q4"
    event = _reply_event("yes do that", quote)
    assert reply_memory_query(event, f'[Replying to: "{quote}"]\n\nyes do that') == f"yes do that\n\n{quote}"


def test_own_message_envelope_is_recognised():
    quote = "Here is the ledger summary you asked for."
    event = _reply_event("and the totals?", quote, reply_to_is_own_message=True)
    envelope = f'[Replying to your previous message: "{quote}"]\n\nand the totals?'
    assert reply_memory_query(event, envelope) == f"and the totals?\n\n{quote}"


def test_user_text_alone_when_it_fills_the_budget():
    quote = "q" * 50
    user_text = "u" * (REPLY_MEMORY_QUERY_CHARS + 20)
    event = _reply_event(user_text, quote)
    assert reply_memory_query(event, f'[Replying to: "{quote}"]\n\n{user_text}') == user_text


def test_empty_user_text_falls_back_to_the_quote():
    quote = "a message replied to with only a sticker"
    event = _reply_event("", quote)
    assert reply_memory_query(event, f'[Replying to: "{quote}"]\n\n') == quote


def test_non_reply_turns_and_foreign_envelopes_return_none():
    plain = MessageEvent(text="hello", source=_source(), message_id="m2")
    assert reply_memory_query(plain, "hello") is None
    # A reply whose model text was rewritten by something else (envelope not ours): keep message text.
    assert reply_memory_query(_reply_event("x", "quote"), "rewritten by a plugin") is None
    assert reply_memory_query(_reply_event("x", "quote"), ["multimodal", "parts"]) is None


def test_discord_triggering_note_is_peeled_before_matching():
    quote = "Create a project plan for Q4"
    event = MessageEvent(text="yes do that", source=_source(Platform.DISCORD), message_id="1550",
                         reply_to_message_id="1549", reply_to_text=quote)
    envelope = f'{discord_triggering_note("1550")}\n\n[Replying to: "{quote}"]\n\nyes do that'
    assert reply_memory_query(event, envelope) == f"yes do that\n\n{quote}"


# ── the real inbound prep produces an envelope the helper recognises ─────────


@pytest.mark.asyncio
async def test_real_inbound_prep_envelope_yields_user_first_query():
    runner = _runner()
    source = _source()
    quote = "Long quoted bot answer. " * 60  # > 800 chars
    event = MessageEvent(text="which one is cheapest?", source=source, message_id="m2",
                         reply_to_message_id="m1", reply_to_text=quote)

    model_text = await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    query = reply_memory_query(event, model_text)

    assert model_text.startswith('[Replying to: "Long quoted')  # model-facing message unchanged
    assert query.startswith("which one is cheapest?\n\n")
    assert len(query) <= REPLY_MEMORY_QUERY_CHARS


# ── plumbing: first turn and queued follow-up both hand memory_query to _run_agent ───────


@pytest.mark.asyncio
async def test_queued_followup_carries_memory_query():
    runner = _runner()
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:dm:c1")
    quote = "the earlier long answer " * 50
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value=f'[Replying to: "{quote}"]\n\nand what about the second option?')
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    source = _source()
    turn_ctx = SimpleNamespace(
        source=source, session_id="sid", session_key="agent:main:telegram:dm:c1", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
        result_holder=[None],
    )
    pending_event = SimpleNamespace(
        source=source, message_id="m9", channel_prompt=None, message_type=None, internal=False, metadata={},
        reply_to_message_id="m8", reply_to_text=quote,
    )

    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="and what about the second option?", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None,
    )

    kwargs = runner._run_agent.await_args.kwargs
    assert kwargs["memory_query"].startswith("and what about the second option?\n\n")
    assert kwargs["message"].startswith('[Replying to: "')


@pytest.mark.asyncio
async def test_queued_followup_without_reply_passes_no_memory_query():
    runner = _runner()
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:dm:c1")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="plain follow-up")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    source = _source()
    turn_ctx = SimpleNamespace(
        source=source, session_id="sid", session_key="agent:main:telegram:dm:c1", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
        result_holder=[None],
    )
    pending_event = SimpleNamespace(
        source=source, message_id="m9", channel_prompt=None, message_type=None, internal=False, metadata={},
    )

    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="plain follow-up", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None,
    )

    assert runner._run_agent.await_args.kwargs["memory_query"] is None


def test_turn_runner_forwards_memory_query_to_run_conversation():
    """``TurnContext.memory_query`` reaches ``run_conversation`` (the seam cron already uses);
    a non-reply turn keeps today's call shape (no key at all)."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    calls = []

    class _Agent:
        def run_conversation(self, user_message, *, conversation_history=None, task_id=None,
                             turn_author=None, memory_query=None, **kw):
            calls.append({"memory_query": memory_query, "kw": kw})
            return {"final_response": "ok", "messages": []}

    def _run(memory_query):
        ctx = TurnContext(source=_source(), message="msg", session_id="sid", session_key="k",
                          memory_query=memory_query)
        runner = TurnRunner(_runner(), ctx)
        runner._native_image_run_message = lambda: "msg"
        return runner._run_conversation_with_approval(_Agent(), [], observed_group_context=None,
                                                      persist_user_message_override=None,
                                                      persist_user_timestamp_override=None)

    _run("user words\n\nquote tail")
    _run(None)
    assert calls[0]["memory_query"] == "user words\n\nquote tail"
    assert calls[1]["memory_query"] is None
