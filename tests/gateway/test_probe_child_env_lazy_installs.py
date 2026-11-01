"""Daemon test children stay hermetic in a self-managed checkout.

A checkout whose install-stamp.json says ``updateMechanism: self`` makes every Hermes child run
``venv_sync.prepare_launch``; a borrowed temporary HERMES_HOME then relaunches the child with
``-I``, which drops the test's PYTHONPATH witnesses (intake t_eafb4023). ``child_env()`` carries
the kill switch so those children skip that path.
"""
import json
import os

import pytest

from hermes_cli import steward, venv_sync
from tests.gateway.fixtures.local_recovery_probe import child_env


class StampRead(Exception):
    pass


@pytest.fixture
def self_managed_root(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    (root / '.git').mkdir(parents=True)
    (root / 'pyproject.toml').write_text('[project]\nname = "fake"\n')
    (root / 'install-stamp.json').write_text(json.dumps({'updateMechanism': 'self'}))

    def sentinel(_root):
        raise StampRead(str(_root))

    monkeypatch.setattr(steward, 'read_install_stamp', sentinel)
    monkeypatch.setattr(venv_sync, '_is_tail_script', lambda *_: False)
    return root


def test_child_env_carries_lazy_install_kill_switch():
    assert child_env()['HERMES_DISABLE_LAZY_INSTALLS'] == '1'


def test_child_env_skips_self_managed_launch(self_managed_root, monkeypatch):
    monkeypatch.setattr(os, 'environ', dict(child_env()))
    assert venv_sync.prepare_launch(self_managed_root, ['gateway', 'run']) is None


def test_env_without_kill_switch_reaches_install_stamp(self_managed_root, monkeypatch):
    env = dict(child_env())
    env.pop('HERMES_DISABLE_LAZY_INSTALLS')
    monkeypatch.setattr(os, 'environ', env)
    with pytest.raises(StampRead):
        venv_sync.prepare_launch(self_managed_root, ['gateway', 'run'])
