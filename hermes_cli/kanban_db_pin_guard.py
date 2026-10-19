"""Fail closed when ``HERMES_KANBAN_HOME`` and a kanban env pin name different boards.

The dispatcher pins ``HERMES_KANBAN_DB`` / ``..._WORKSPACES_ROOT`` /
``..._ATTACHMENTS_ROOT`` into every worker; the pin outranks ``HERMES_KANBAN_HOME``
on purpose (the worker isolation fence, 5ec6baa). A process that also sets
``HERMES_KANBAN_HOME`` to somewhere else (a probe or test isolating its board) has
stated an intent the pin contradicts; resolving it silently in favour of the pin
wrote junk cards to the live board (intake t_29de271a).

Split out of ``hermes_cli.kanban_db``; ``kanban_home`` is read late-bound through
the facade so monkeypatching ``kanban_db.kanban_home`` keeps working.
"""

from __future__ import annotations

import os
from pathlib import Path

_PIN_VARS = (
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_BOARD",
)


class KanbanPathConflict(RuntimeError):
    """``HERMES_KANBAN_HOME`` names a kanban root the active env pin lies outside of."""


class KanbanBoardFenced(PermissionError):
    """A fenced caller (dispatched worker / delegated child) named a board other than
    its pinned one. Raised instead of silently resolving to the pin (intake t_20d642ad)."""


# Each resolver's env pin -> the (default-board parts, named-board leaf) of its natural path.
_FENCE_PINS = {
    "HERMES_KANBAN_DB": (("kanban.db",), "kanban.db"),
    "HERMES_KANBAN_WORKSPACES_ROOT": (("kanban", "workspaces"), "workspaces"),
    "HERMES_KANBAN_ATTACHMENTS_ROOT": (("kanban", "attachments"), "attachments"),
}


def natural_board_path(slug: str, default_parts: tuple[str, ...], leaf: str) -> Path:
    """A board's own path, ignoring every env pin."""
    from hermes_cli import kanban_db as _kb
    if slug == _kb.DEFAULT_BOARD:
        return _kb.kanban_home().joinpath(*default_parts)
    return _kb.board_dir(slug) / leaf


def _is_pinned_board(slug: str, pins: dict[str, str]) -> bool:
    from hermes_cli import kanban_db as _kb
    try:
        if _kb._normalize_board_slug(os.environ.get("HERMES_KANBAN_BOARD") or "") == slug:
            return True
    except ValueError:
        pass
    return any(
        natural_board_path(slug, *_FENCE_PINS[var]).resolve() == Path(pin).expanduser().resolve()
        for var, pin in pins.items()
    )


def assert_board_in_fence(slug: str) -> None:
    """Raise :class:`KanbanBoardFenced` when a fenced caller's explicit ``slug`` is not
    its pinned board. Every resolver checks against all pins, so ``attachments_root``
    is fenced even though the dispatcher pins only the DB and workspaces root. No pin
    set means no fence (a delegated child of an unpinned interactive session)."""
    pins = {var: p for var in _FENCE_PINS if (p := os.environ.get(var, "").strip())}
    if not pins or _is_pinned_board(slug, pins):
        return
    var, pin = next(iter(pins.items()))
    raise KanbanBoardFenced(
        f"board {slug!r} is outside this worker's kanban fence ({var}={pin}); "
        f"comment on a card of the pinned board instead"
    )


def checked_pin(env_var: str | None, pin: Path) -> Path:
    """Return ``pin`` unless ``HERMES_KANBAN_HOME`` is set and ``pin`` lies outside it."""
    if not os.environ.get("HERMES_KANBAN_HOME", "").strip():
        return pin
    from hermes_cli import kanban_db as _kb
    home = _kb.kanban_home().resolve()
    if pin.resolve().is_relative_to(home):
        return pin
    unset = " ".join(f"-u {name}" for name in _PIN_VARS)
    raise KanbanPathConflict(
        f"HERMES_KANBAN_HOME={home} but {env_var}={pin} pins a path outside it, so this "
        f"process would use a different kanban board than the one HERMES_KANBAN_HOME "
        f"names. Unset {' / '.join(_PIN_VARS)} (env {unset} ...) or pass "
        f"connect(db_path=...)."
    )
