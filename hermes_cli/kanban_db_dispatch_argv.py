"""Module-form worker argv: the interpreter-bound ``python -P -m hermes_cli.main``
invocation the dispatcher falls back to, and the ``PYTHONPATH`` pin it needs.

Split out of ``hermes_cli.kanban_db_dispatch`` (which imports these names and is
where ``_resolve_hermes_argv``/``_default_spawn`` read them).
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_ENTRY = ["-m", "hermes_cli.main"]


def _module_hermes_argv() -> list[str]:
    """Interpreter-bound Hermes CLI invocation (``hermes_cli.main`` is the
    console-script target — there is no top-level ``hermes`` package).

    ``-P`` (Python >= 3.11) keeps the worker's cwd off ``sys.path``: ``-m``
    otherwise puts the task workspace ahead of the ``PYTHONPATH`` pin, so a
    workspace holding ``hermes_cli/``, ``pm/`` or ``yaml.py`` (a hermes-agent
    worktree) shadows the install and the worker dies at boot (t_8eb8551f).
    """
    if sys.version_info >= (3, 11):
        return [sys.executable, "-P", *_MODULE_ENTRY]
    return [sys.executable, *_MODULE_ENTRY]


def _is_module_argv(cmd: list[str]) -> bool:
    """True for both ``_module_hermes_argv`` forms (with or without ``-P``)."""
    return cmd[1:3] == _MODULE_ENTRY or cmd[1:4] == ["-P", *_MODULE_ENTRY]


def _propagate_module_import_root(cmd: list[str], env: dict[str, str]) -> None:
    """Put the running install's package root on a module-form worker's path.

    ``_resolve_hermes_argv`` proves ``hermes_cli`` importable in THIS process,
    where a store-python shim has the repo root on ``sys.path`` in-process;
    the spawned child runs the bare ``sys.executable`` from the task workspace
    with a scrubbed ``PYTHONPATH`` and cannot import the package the parent
    just proved importable — it dies before any work and the board
    auto-blocks (#122299, #122487, #122500). Same-interpreter child, so the
    root is version-safe to propagate; ``hermes_cli.main``'s own bootstrap
    then owns dependency activation as usual. A resolved shim path owns its
    imports and is left alone. Same pin cron's external worker uses (#112729).
    With ``-P`` this pin is the worker's only route to the package.
    """
    if not _is_module_argv(cmd):
        return
    from cron.scheduler_worker_env import pin_hermes_tree_on_pythonpath

    pin_hermes_tree_on_pythonpath(env, Path(__file__).resolve().parents[1])
