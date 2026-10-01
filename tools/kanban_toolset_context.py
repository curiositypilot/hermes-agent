"""Explicit Kanban selection during one model-schema build.

The registry's ordinary availability cache is profile-wide, while a gateway
can build schemas for several platforms in the same profile concurrently.
Carry only the explicit selection through a ContextVar, never process env.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Iterator, Optional

_requested: ContextVar[Optional[bool]] = ContextVar("kanban_toolset_requested", default=None)


def kanban_toolset_requested() -> Optional[bool]:
    """None outside schema assembly; otherwise whether Kanban was named explicitly."""
    return _requested.get()


@contextmanager
def scoped_kanban_toolset_selection(toolsets: Optional[Iterable[str]]) -> Iterator[None]:
    """An all/default selection is not an explicit workflow opt-in."""
    token = _requested.set("kanban" in (toolsets or ()))
    try:
        yield
    finally:
        _requested.reset(token)


# ``kanban.worker_disabled_toolsets`` when the key is absent from config entirely.
DEFAULT_WORKER_DISABLED_TOOLSETS: tuple[str, ...] = ("clarify", "browser_vault")


def worker_disabled_toolsets(config: Optional[dict] = None) -> list[str]:
    """Toolset names a dispatcher-spawned worker never gets (``kanban.worker_disabled_toolsets``).

    Read by the dispatcher when it builds the worker's ``--toolsets`` pin AND by schema assembly
    inside the worker (the pin cannot express "minus the vault tools" because they ride inside
    ``browser``). A worker is headless: ``clarify`` has no one to answer and the browser vault
    needs a user-side unlock/consent prompt. An explicit empty list disables the trim.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            config = {}
    kanban_cfg = (config or {}).get("kanban") if isinstance(config, dict) else None
    if not isinstance(kanban_cfg, dict) or "worker_disabled_toolsets" not in kanban_cfg:
        return list(DEFAULT_WORKER_DISABLED_TOOLSETS)
    try:
        from agent.skill_utils import parse_config_string_list
        names = parse_config_string_list(kanban_cfg.get("worker_disabled_toolsets"))
    except Exception:
        names = []
    return [str(n).strip() for n in names if str(n).strip()]
