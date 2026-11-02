"""Tests for retained generation re-adoption when inputs stamp matches (t_92660d96)."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
import pytest

from pm.package import InstallError
from pm.plugin_inputs import Members
from tests.pm._fixtures import (
    _wheel,
    build_worker as build_worker,  # noqa: PLC0414
    client as client,  # noqa: PLC0414
    isolated_python as isolated_python,  # noqa: PLC0414
)
from tests.pm.test_environment_build import locked_project


def _setup_sync_env(locked_project, monkeypatch):
    import importlib
    from pm import paths

    source, uv, _env = locked_project
    monkeypatch.setattr(paths, "repo_root", lambda: source)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (uv, Path(sys.executable)))
    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(engine, "lazy_installs_allowed", lambda: True)
    home = Path(os.environ["HERMES_HOME"])
    config = home / "config.yaml"
    config.write_text("plugins: {enabled: []}\nsecurity: {allow_lazy_installs: true}\n", encoding="utf-8")
    return source, engine


def _sync_exact_extras(engine, source, extras):
    """Sync setting fact extras to exactly *extras* so extras union does not accumulate."""
    from pm import paths
    from pm.lock import Facts
    facts = Facts(paths.runtime_facts_path())
    fact = facts.get("venv")
    if fact is not None:
        fact["extras"] = list(extras)
        facts._merge_and_write("venv", fact)
    engine.sync_venv(extras, plugins=Members([]), explicit=True)


def test_aba_change_readopts_first_generation(locked_project, monkeypatch):
    """Sync S0, then S1, then S0 again. Third sync re-adopts first generation."""
    from pm import receipt
    from pm.environments import install_state_dir, selected_venv
    from pm.packages import Venv

    source, engine = _setup_sync_env(locked_project, monkeypatch)

    # 1. Sync S0 (extras=["chosen"])
    _sync_exact_extras(engine, source, ["chosen"])
    g1 = selected_venv(source).parent
    assert (g1 / ".venv-stamp").is_file()
    assert (g1 / ".lease-managed").is_file()

    # 2. Sync S1 (extras=["other"])
    _sync_exact_extras(engine, source, ["other"])
    g2 = selected_venv(source).parent
    assert g2 != g1
    assert (g2 / ".venv-stamp").is_file()

    # 3. Re-sync S0 (extras=["chosen"])
    apply_called = False
    orig_apply = Venv.apply

    def fake_apply(*args, **kwargs):
        nonlocal apply_called
        apply_called = True
        return orig_apply(*args, **kwargs)

    monkeypatch.setattr(Venv, "apply", fake_apply)

    _sync_exact_extras(engine, source, ["chosen"])
    assert not apply_called, "Venv.apply should not be called when re-adopting generation"

    # Environments dir holds 2 generations
    gen_dirs = [d for d in (install_state_dir(source) / "environments").iterdir() if d.is_dir() and not d.is_symlink()]
    assert len(gen_dirs) == 2

    # Facts environment equals first generation's venv
    assert selected_venv(source) == g1 / "venv"

    # Receipt reason starts with "re-adopted generation"
    last_receipt = receipt.latest()
    assert last_receipt is not None
    assert last_receipt["venv_rebuild"]["ok"] is False
    assert last_receipt["venv_rebuild"]["reason"].startswith("re-adopted generation")
    assert g1.name in last_receipt["venv_rebuild"]["reason"]


def test_failed_build_writes_no_marker(locked_project, monkeypatch):
    """If apply raises or stamp re-check fails, no .venv-stamp is written anywhere."""
    from pm.environments import install_state_dir
    from pm.packages import Venv

    source, engine = _setup_sync_env(locked_project, monkeypatch)

    def failing_apply(*args, **kwargs):
        raise InstallError("venv", "simulated build failure")

    monkeypatch.setattr(Venv, "apply", failing_apply)

    with pytest.raises(InstallError, match="simulated build failure"):
        engine.sync_venv(["chosen"], plugins=Members([]), explicit=True)

    gen_dir = install_state_dir(source) / "environments"
    if gen_dir.exists():
        stamps = list(gen_dir.glob("*/.venv-stamp"))
        assert len(stamps) == 0


@pytest.mark.parametrize("case", [
    "diff_stamp",
    "missing_stamp",
    "missing_cfg",
    "missing_lock",
    "missing_pyproject",
    "dangling_python",
    "check_raises",
    "repair_true",
])
def test_must_not_reuse_cases(locked_project, monkeypatch, case):
    """Candidate must not be reused when marker differs, missing, broken cfg, broken lock,
    dangling python symlink, check() raises InstallError, or repair=True."""
    from pm.environments import selected_venv
    from pm.environment import PythonEnvironment
    from pm.packages import Venv

    source, engine = _setup_sync_env(locked_project, monkeypatch)

    # Build initial S0
    _sync_exact_extras(engine, source, ["chosen"])
    g1 = selected_venv(source).parent
    assert (g1 / ".venv-stamp").is_file()

    # Move to S1 so g1 is retained but not selected
    _sync_exact_extras(engine, source, ["other"])
    g2 = selected_venv(source).parent
    assert g2 != g1

    pkg = engine.get_package("venv")
    s0_stamp = pkg.expected_stamp(["chosen"], plugin_dirs=[])
    (g1 / ".venv-stamp").write_text(s0_stamp + "\n", encoding="utf-8")
    assert (g1 / "venv" / "pyvenv.cfg").is_file()
    assert (g1 / "workspace" / "uv.lock").is_file()
    assert (g1 / "workspace" / "pyproject.toml").is_file()

    # Invalidate according to case
    orig_check = PythonEnvironment.check
    if case == "diff_stamp":
        (g1 / ".venv-stamp").write_text("bogus_stamp\n", encoding="utf-8")
    elif case == "missing_stamp":
        (g1 / ".venv-stamp").unlink()
    elif case == "missing_cfg":
        (g1 / "venv" / "pyvenv.cfg").unlink()
    elif case == "missing_lock":
        (g1 / "workspace" / "uv.lock").unlink()
    elif case == "missing_pyproject":
        (g1 / "workspace" / "pyproject.toml").unlink()
    elif case == "dangling_python":
        from pm.environments import venv_python
        py_bin = venv_python(g1 / "venv")
        py_bin.unlink()
    elif case == "check_raises":
        def bad_check(self, lock=None):
            if self.destination.resolve() == (g1 / "venv").resolve():
                raise InstallError("venv", "broken dependencies")
            return orig_check(self, lock)

        monkeypatch.setattr(PythonEnvironment, "check", bad_check)
    elif case == "repair_true":
        pass

    apply_called = False
    orig_apply = Venv.apply

    def tracked_apply(*args, **kwargs):
        nonlocal apply_called
        apply_called = True
        return orig_apply(*args, **kwargs)

    monkeypatch.setattr(Venv, "apply", tracked_apply)

    if case == "repair_true":
        engine.sync_venv(explicit=True, repair=True)
    else:
        _sync_exact_extras(engine, source, ["chosen"])

    assert apply_called, f"Case {case} should have called Venv.apply to build a new generation"
    new_sel = selected_venv(source).parent
    assert new_sel != g1, f"Case {case} should not have re-adopted g1"


def test_newest_matching_generation_wins(locked_project, monkeypatch):
    """Two generations match stamp. The one with newer .lease-managed mtime is picked and refreshed."""
    from pm.environments import install_state_dir, selected_venv
    from pm.packages import Venv

    source, engine = _setup_sync_env(locked_project, monkeypatch)

    # 1. Sync S0
    _sync_exact_extras(engine, source, ["chosen"])
    g1 = selected_venv(source).parent

    # 2. Sync S1
    _sync_exact_extras(engine, source, ["other"])

    # 3. Simulate another generation g3 created for S0 by copying g1
    g3 = install_state_dir(source) / "environments" / "g3_copy"
    shutil.copytree(g1, g3, symlinks=True)

    # Set g3 .lease-managed mtime to be newer than g1
    t1 = 1000.0
    t3 = 2000.0
    os.utime(g1 / ".lease-managed", (t1, t1))
    os.utime(g3 / ".lease-managed", (t3, t3))

    apply_called = False
    orig_apply = Venv.apply

    def tracked_apply(*args, **kwargs):
        nonlocal apply_called
        apply_called = True
        return orig_apply(*args, **kwargs)

    monkeypatch.setattr(Venv, "apply", tracked_apply)

    before_sync = time.time()
    _sync_exact_extras(engine, source, ["chosen"])
    assert not apply_called

    # Selected must be g3 because g3 had the newer .lease-managed mtime
    assert selected_venv(source).parent == g3
    assert (g3 / ".lease-managed").stat().st_mtime >= before_sync


def test_replay_sequence_4a36_pattern(locked_project, monkeypatch):
    """Replay sequence as stamps S0, S1, S0, S2, S0. Expect 2 builds instead of 4,
    and final selection is the first S0 generation."""
    from pm.environments import selected_venv
    from pm.packages import Venv

    source, engine = _setup_sync_env(locked_project, monkeypatch)

    build_count = 0
    orig_apply = Venv.apply

    def counted_apply(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        return orig_apply(*args, **kwargs)

    monkeypatch.setattr(Venv, "apply", counted_apply)

    # 1. S0 (build 1)
    _sync_exact_extras(engine, source, ["chosen"])
    g0 = selected_venv(source).parent

    # 2. S1 (build 2)
    _sync_exact_extras(engine, source, ["other"])

    # 3. S0 (re-adopt g0, no build)
    _sync_exact_extras(engine, source, ["chosen"])
    assert selected_venv(source).parent == g0

    # 4. S2 (build 3: extras=["chosen", "other"])
    _sync_exact_extras(engine, source, ["chosen", "other"])

    # 5. S0 (re-adopt g0, no build)
    _sync_exact_extras(engine, source, ["chosen"])
    assert selected_venv(source).parent == g0

    # Total builds across all syncs: exactly 3 builds for 5 syncs (instead of 5).
    assert build_count == 3
