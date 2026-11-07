"""Allowlist-rebuilt child envs keep the lazy-install kill switch (intake t_7b2d2135).

In a self-managed checkout every Hermes child a test starts with a hand-built env that carries
HERMES_HOME used to sync a full dependency generation into the test's temp home, downloading the
whole lock per test. ``tests/conftest.py`` wraps ``subprocess.Popen`` there so such envs gain
``HERMES_DISABLE_LAZY_INSTALLS=1``.
"""
import asyncio
import json
import os
import subprocess
import sys

import pytest

from hermes_cli import steward, venv_sync
from tests._fixtures.lazy_install_child_env import (
    hermetic_child_env,
    self_managed_checkout,
    wrap_popen_init,
)

PRINT_SWITCH = 'import os;print(os.environ.get("HERMES_DISABLE_LAZY_INSTALLS"))'


class StampRead(Exception):
    pass


def test_allowlist_env_with_home_gains_switch_without_mutation(tmp_path):
    env = {'PATH': os.environ['PATH'], 'HERMES_HOME': str(tmp_path)}
    out = hermetic_child_env(env)
    assert out['HERMES_DISABLE_LAZY_INSTALLS'] == '1'
    assert 'HERMES_DISABLE_LAZY_INSTALLS' not in env


@pytest.mark.parametrize('env', [
    {'PATH': '/bin'},
    {'PATH': '/bin', 'HERMES_HOME': '/h', 'HERMES_DISABLE_LAZY_INSTALLS': '0'},
    {'PATH': '/bin', 'HERMES_HOME': '/h', 'HERMES_TEST_ISOLATION': '1'},
    None,
    {b'PATH': b'/bin', b'HERMES_HOME': b'/h'},
], ids=['no-home', 'explicit-zero', 'copied-from-os-environ', 'none', 'bytes-keys'])
def test_env_left_as_the_test_built_it(env):
    before = dict(env) if env is not None else None
    assert hermetic_child_env(env) == before


@pytest.mark.parametrize('content,expected', [
    (json.dumps({'updateMechanism': 'self'}), True),
    (json.dumps({'updateMechanism': 'git'}), False),
    (None, False),
    ('not json', False),
])
def test_self_managed_checkout_reads_the_stamp(tmp_path, content, expected):
    if content is not None:
        (tmp_path / 'install-stamp.json').write_text(content)
    assert self_managed_checkout(tmp_path) is expected


@pytest.fixture
def guarded_popen(monkeypatch):
    monkeypatch.setattr(subprocess.Popen, '__init__', wrap_popen_init(subprocess.Popen.__init__))


def test_real_child_sees_switch_through_subprocess_run(guarded_popen, tmp_path):
    env = {'PATH': os.environ['PATH'], 'HERMES_HOME': str(tmp_path)}
    out = subprocess.run([sys.executable, '-c', PRINT_SWITCH], env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '1'


def test_real_child_sees_switch_through_asyncio(guarded_popen, tmp_path):
    env = {'PATH': os.environ['PATH'], 'HERMES_HOME': str(tmp_path)}

    async def run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, '-c', PRINT_SWITCH, env=env, stdout=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    assert asyncio.run(run()) == '1'


def test_real_child_without_home_gets_no_switch(guarded_popen):
    out = subprocess.run([sys.executable, '-c', PRINT_SWITCH], env={'PATH': os.environ['PATH']},
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'None'


def test_wrapping_twice_does_not_stack():
    once = wrap_popen_init(subprocess.Popen.__init__)
    assert wrap_popen_init(once) is once


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


def test_guarded_env_skips_self_managed_launch(self_managed_root, tmp_path, monkeypatch):
    env = {'PATH': os.environ['PATH'], 'HERMES_HOME': str(tmp_path / 'home')}
    monkeypatch.setattr(os, 'environ', hermetic_child_env(env))
    assert venv_sync.prepare_launch(self_managed_root, ['gateway', 'run']) is None


def test_unguarded_allowlist_env_reaches_install_stamp(self_managed_root, tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'environ', {'PATH': os.environ['PATH'], 'HERMES_HOME': str(tmp_path / 'home')})
    with pytest.raises(StampRead):
        venv_sync.prepare_launch(self_managed_root, ['gateway', 'run'])
