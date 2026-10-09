"""A target gap is refused in the caller, before a PM worker is spawned.

Regression for NousResearch/hermes-agent#129792: on POSIX every
``selected_git_env()`` call (each checkpoint git command) spawned
``pm/worker.py`` only for it to raise "POSIX uses system git by choice".
"""
import subprocess

import pytest

from hermes_cli._subprocess_compat import selected_git_env
import pm.install
import pm.store
from pm import client
from pm.package import InstallError
from pm.registry import get_package

GAPPED_TARGET, GAP_REASON = next(iter(get_package("git").gaps.items()))
UNGAPPED_TARGET = next(target for target in ("win32-x64", "win32-arm64")
                       if get_package("git").missing_reason(target) is None)


@pytest.fixture
def worker_calls(monkeypatch):
    """Nothing installed, lazy installs allowed, every worker spawn recorded instead of run."""
    calls = []
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(client, "is_runtime", lambda: False)
    monkeypatch.setattr(pm.install, "is_installed", lambda name: False)
    monkeypatch.setattr(pm.install, "env_for", lambda name, base_env=None: dict(base_env or {}))
    monkeypatch.setattr(client, "_request", lambda operation, arguments, **kw: calls.append((operation, arguments)))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: calls.append(("Popen", a)))
    return calls


@pytest.mark.parametrize("explicit", [False, True])
def test_gapped_package_raises_without_a_worker(monkeypatch, worker_calls, explicit):
    monkeypatch.setattr(pm.store, "current_target", lambda: GAPPED_TARGET)

    with pytest.raises(InstallError) as caught:
        client.ensure("git", explicit=explicit)

    assert caught.value.package == "git"
    assert f"unavailable on {GAPPED_TARGET}" in caught.value.cause
    assert GAP_REASON in caught.value.cause
    assert caught.value.remedy == "none"
    assert worker_calls == []


def test_ungapped_missing_package_still_reaches_the_worker(monkeypatch, worker_calls):
    monkeypatch.setattr(pm.store, "current_target", lambda: UNGAPPED_TARGET)

    runner = client.ensure("git", base_env={"KEEP": "1"})

    assert [operation for operation, _ in worker_calls] == ["ensure"]
    assert runner.env == {"KEEP": "1"}


def test_lazy_install_refusal_wins_over_the_gap(monkeypatch, worker_calls):
    monkeypatch.setattr(pm.store, "current_target", lambda: GAPPED_TARGET)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

    with pytest.raises(InstallError) as caught:
        client.ensure("git")

    assert "lazy installs are disabled" in caught.value.cause
    assert worker_calls == []


def test_selected_git_env_falls_back_to_base_without_a_worker(monkeypatch, worker_calls):
    monkeypatch.setattr(pm.store, "current_target", lambda: GAPPED_TARGET)
    base = {"PATH": "/usr/bin", "KEEP": "1"}

    assert selected_git_env(base) == base
    assert worker_calls == []
