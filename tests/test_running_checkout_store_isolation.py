"""The checkout running pytest resolves its PM store inside the test home, as on CI.

A checkout installed inside a real home (``~/.hermes/hermes-agent``, or a kanban worktree
beneath it) finds install stamps and facts there; code under test that asks for the running
install's store then read the developer's store or tripped the home I/O guard (t_dc0c7aef).
"""
from __future__ import annotations

import json

import pm.environments
import pm.paths
from hermes_cli import _launchers
from hermes_constants import get_default_hermes_root


def test_running_checkout_store_is_the_test_homes(monkeypatch):
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    tools = get_default_hermes_root() / "tools"
    assert pm.paths.store_root() == tools
    assert pm.environments.store_root(pm.paths.repo_root(), honor_runtime_override=False) == tools
    assert _launchers.resolve_store_python(pm.paths.repo_root(), publication=True) is None


def test_other_trees_and_the_runtime_override_resolve_as_in_production(tmp_path, monkeypatch):
    store = tmp_path / "store"
    tree = tmp_path / "payload" / "hermes-agent"
    tree.mkdir(parents=True)
    (tree.parent / "install-stamp.json").write_text(json.dumps({"runtimeDir": str(store)}))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    assert pm.environments.store_root(tree) == store.resolve()

    override = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(override))
    assert pm.paths.store_root() == override.resolve()
