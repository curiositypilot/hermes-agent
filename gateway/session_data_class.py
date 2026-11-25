"""Per-chat data class: ``channel_overrides.<id>.data_class`` binds a ``kanban.data_policies`` class.

A chat could pin its main model but not its data class, so its auxiliary calls (compression, titles,
vision, web_extract, the background review) followed the profile's cloud ``auxiliary.*`` routes even in
a local-only privacy topic (intake t_36fb835b). The gateway binds the class with
``agent.provider_policy.bind_data_class``, a ContextVar, so ``copy_context()`` carries it into the
agent, title and review threads; the agent package stays gateway-agnostic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional

from agent.provider_policy import bind_data_class, reset_data_class


class _DataClassToken:
    """A bound class inside a ``_set_session_env`` tokens list. ``clear_session_vars`` ignores tokens,
    so ``reset_channel_data_class`` must reset these itself."""
    __slots__ = ("token",)

    def __init__(self, token) -> None:
        self.token = token


def channel_data_class(runner: Any, source: Any) -> Optional[str]:
    """The ``data_class`` of ``source``'s channel override, or None (the common, cheap case)."""
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    override = GatewayModelCommandsMixin._channel_override_for(runner, source)
    label = getattr(override, "data_class", None)
    return str(label).strip() or None if label else None


def bind_channel_data_class(runner: Any, source: Any) -> list:
    """Bind ``source``'s channel class; returns the tokens to append to a session-env tokens list
    (empty when the chat sets none). An unknown class raises ``ProviderDenied``: the turn fails
    closed instead of running as ``internal``."""
    label = channel_data_class(runner, source)
    return [_DataClassToken(bind_data_class(label))] if label else []


def reset_channel_data_class(tokens: Optional[list]) -> None:
    """Undo every class ``bind_channel_data_class`` put in ``tokens`` (innermost first)."""
    for item in reversed(tokens or []):
        if isinstance(item, _DataClassToken):
            reset_data_class(item.token)


@contextmanager
def channel_data_class_scope(runner: Any, source: Any, inner):
    """Enter ``inner`` (the profile scope, whose home owns ``data_policies``), then bind the class."""
    with inner:
        tokens = bind_channel_data_class(runner, source)
        try:
            yield
        finally:
            reset_channel_data_class(tokens)


@asynccontextmanager
async def async_channel_data_class_scope(runner: Any, source: Any, inner):
    """``async with`` twin of :func:`channel_data_class_scope`."""
    async with inner:
        tokens = bind_channel_data_class(runner, source)
        try:
            yield
        finally:
            reset_channel_data_class(tokens)
