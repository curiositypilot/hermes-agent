"""Keep hand-built child envs hermetic when the suite runs in a self-managed checkout.

A checkout whose ``install-stamp.json`` says ``updateMechanism: self`` makes every Hermes child
run ``venv_sync.prepare_launch``. A test's temporary ``HERMES_HOME`` only borrows that checkout,
so ``_prepare_borrowed_launch`` syncs a full dependency generation into it with an empty uv cache:
one whole-lock PyPI download per test (intake t_7b2d2135). ``tests/conftest.py`` exports the
``HERMES_DISABLE_LAZY_INSTALLS`` kill switch, but tests that rebuild the child env from an
allowlist (``PATH``, ``LANG``, ...) drop it; t_eafb4023 fixed only ``child_env()``. Patching each
file does not hold (the next test copies the allowlist), so conftest installs this guard on
``subprocess.Popen`` instead. Worktrees and CI carry no stamp and are left untouched.
"""
from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path

KILL_SWITCH = "HERMES_DISABLE_LAZY_INSTALLS"
_INSTALLED_WRAPPERS: set = set()


def self_managed_checkout(root: Path) -> bool:
    """True only when ``root/install-stamp.json`` declares ``updateMechanism: self``."""
    try:
        stamp = json.loads((Path(root) / "install-stamp.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(stamp, dict) and stamp.get("updateMechanism") == "self"


def hermetic_child_env(env):
    """Add the kill switch to an allowlist-rebuilt child env that points at a HERMES_HOME.

    An env copied from ``os.environ`` carries conftest's ``HERMES_TEST_ISOLATION`` marker; a
    test that copied it and dropped the switch did so on purpose and keeps that choice. An
    explicit value (``"0"``) always wins. Anything else, ``None`` included, passes unchanged.
    """
    if (not isinstance(env, Mapping)
            or "HERMES_HOME" not in env
            or KILL_SWITCH in env
            or "HERMES_TEST_ISOLATION" in env
            or not all(isinstance(key, str) for key in env)):
        return env
    return {**env, KILL_SWITCH: "1"}


def wrap_popen_init(init):
    """Return a ``Popen.__init__`` that passes ``env=`` through :func:`hermetic_child_env`."""
    if init in _INSTALLED_WRAPPERS:
        return init  # xdist re-executes conftest; never stack the wrapper

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        if "env" in kwargs:
            kwargs["env"] = hermetic_child_env(kwargs["env"])
        return init(self, *args, **kwargs)

    _INSTALLED_WRAPPERS.add(__init__)
    return __init__
