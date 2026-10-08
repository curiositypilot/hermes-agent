"""``hermes_compose_env`` reads the environment of the checkout it is given, whatever the cwd.

``python -m`` puts the cwd ahead of ``PYTHONPATH``. Run from inside another
checkout (a linked worktree of hermes-agent), that checkout's own ``pm`` would
compose the environment and key an install that was never built.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.pm.activation_support import (
    CANARY, bash, bash_env, fake_store, isolated_checkout, posix,
)


@pytest.mark.platforms("posix")
def test_compose_env_ignores_a_pm_package_in_the_cwd(tmp_path: Path):
    root = isolated_checkout(tmp_path)
    store, _ = fake_store(tmp_path)
    decoy = tmp_path / "decoy"
    (decoy / "pm").mkdir(parents=True)
    (decoy / "pm" / "__init__.py").write_text("", encoding="utf-8")
    (decoy / "pm" / "environments.py").write_text(
        "print('export HERMES_DECOY_PM=1')\n", encoding="utf-8"
    )
    script = (
        f'. "{posix(root / "scripts" / "_activation.sh")}" && '
        f'hermes_compose_env "{posix(root)}" sh'
    )
    result = subprocess.run(
        [bash(), "-c", script],
        capture_output=True, text=True, cwd=posix(decoy), env=bash_env(store), timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "HERMES_DECOY_PM" not in result.stdout
    assert CANARY in result.stdout
