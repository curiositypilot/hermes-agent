"""run_tests.sh keeps a real-HOME checkout on other pins off the account's tool store.

The account store's facts.json holds one entry per package, so a worktree or a
renovate branch provisioning its pins into it re-points the live install
(t_4f98fc8f). Such a checkout gets a private store keyed by its lock; the
installed checkout (install-stamp.json), a checkout on the account's pins and
every non-default layout keep today's choice.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL_STORE = ROOT / "scripts" / "_tool_store.sh"


def _home(tmp_path: Path, name: str = "home", facts: bool = True) -> Path:
    home = tmp_path / name
    tools = home / ".hermes" / "tools"
    tools.mkdir(parents=True)
    if facts:
        (tools / "facts.json").write_text('{"packages": {}}')
    return home


def _checkout(tmp_path: Path, name: str = "checkout", lock: str = '{"pins": 1}') -> Path:
    repo = tmp_path / name
    (repo / "pm").mkdir(parents=True)
    (repo / "pm" / "lock.json").write_text(lock)
    return repo


def _run(func: str, repo: Path, home: Path, account: Path, pins: int = 1, **env: str):
    script = (
        f'REPO_ROOT="$1"; . "{TOOL_STORE}"; '
        '_account_home() { printf "%s\\n" "$FAKE_ACCOUNT"; }; '
        '_store_has_pins() { [ "$PINS" -eq 0 ] && echo 3.14; return "$PINS"; }; '
        f"{func}"
    )
    base = {k: v for k, v in os.environ.items() if k not in ("HERMES_RUNTIME_DIR", "HERMES_HOME")}
    return subprocess.run(
        ["bash", "-c", script, "bash", str(repo)],
        env={**base, "HOME": str(home), "FAKE_ACCOUNT": str(account), "PINS": str(pins), **env},
        capture_output=True, text=True, timeout=30, check=False,
    )


@pytest.mark.platforms("posix")
def test_real_home_checkout_on_other_pins_gets_a_private_store_per_lock(tmp_path):
    home = _home(tmp_path)
    first = _run("_private_tool_store", _checkout(tmp_path, "a"), home, home)
    assert first.returncode == 0, first.stderr
    store = first.stdout.strip()
    assert re.fullmatch(re.escape(f"{home}/.hermes/cache/tool-stores/") + "[0-9a-f]{16}", store)

    same = _run("_private_tool_store", _checkout(tmp_path, "b"), home, home)
    assert same.stdout.strip() == store, "identical pins must share one private store"

    other = _run("_private_tool_store", _checkout(tmp_path, "c", lock='{"pins": 2}'), home, home)
    assert other.returncode == 0, other.stderr
    assert other.stdout.strip() != store, "a lock change must get its own store"


def _stamped(tmp_path: Path) -> dict:
    repo = _checkout(tmp_path)
    (repo / "install-stamp.json").write_text("{}")
    home = _home(tmp_path)
    return {"repo": repo, "home": home, "account": home}


def _base(tmp_path: Path, **overrides) -> dict:
    home = _home(tmp_path)
    return {"repo": _checkout(tmp_path), "home": home, "account": home, **overrides}


NO_PRIVATE_STORE = {
    "installed_checkout": _stamped,
    "pins_match": lambda tmp: _base(tmp, pins=0),
    "runtime_dir_set": lambda tmp: _base(tmp, HERMES_RUNTIME_DIR=str(tmp / "explicit")),
    "custom_hermes_home": lambda tmp: _base(tmp, HERMES_HOME="/elsewhere"),
    "no_account_store": lambda tmp: {"repo": _checkout(tmp), "home": (h := _home(tmp, facts=False)),
                                     "account": h},
    "faked_home": lambda tmp: _base(tmp, account=_home(tmp, "account")),
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", sorted(NO_PRIVATE_STORE))
def test_other_layouts_keep_todays_store(tmp_path, case):
    result = _run("_private_tool_store", **NO_PRIVATE_STORE[case](tmp_path))
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == ""


@pytest.mark.platforms("posix")
def test_faked_home_still_borrows_the_account_store_on_matching_pins(tmp_path):
    account = _home(tmp_path, "account")
    faked = tmp_path / "faked"
    faked.mkdir()
    result = _run("_account_tool_store", _checkout(tmp_path), faked, account, pins=0)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{account}\n3.14\n"
