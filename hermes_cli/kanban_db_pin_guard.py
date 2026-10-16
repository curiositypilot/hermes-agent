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
