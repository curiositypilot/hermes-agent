"""``kanban.worker_memory_max_mb`` bounds the owner-run turn, not only the thin client.

Intake t_94026b0d: since the upstream owner-run model the card's agent runs in the gateway's
``python -m agent.managed_worker``, so the cap on the dispatcher's thin-client scope bounded
nothing a turn runs. Under a systemd-supervised gateway a Kanban card's turn now spawns in its
run's ``hermes-worker-kanban-<task>-run-<run>-turn.scope`` with that MemoryMax; chat and cron
spawns, and gateways outside systemd, stay byte-identical.

The tests drive the real ``_spawn_registered_worker`` with a capturing Popen.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError

ROOT = Path(__file__).resolve().parents[2]
WORKER_ARGV = [sys.executable, '-m', 'agent.managed_worker']
MiB = 1024 * 1024
CONTEXT = {'task_id': 't_ab12', 'run_id': 12, 'tenant': 'pilot', 'skills': [], 'board': 'default'}
UNIT = 'hermes-worker-kanban-t_ab12-run-12-turn'


class FakePlan:
    """The sandbox plan's surface the spawn reads, holding two real fds as its memfds."""

    def __init__(self):
        read_fd, write_fd = os.pipe()
        self.pass_fds = (read_fd, write_fd)
        self.held = self.pass_fds
        self.argv = ('/usr/bin/python3', '-P', 'kanban_sandbox.py', 'exec', '/usr/bin/bwrap', '--', *WORKER_ARGV)
        self.outer_env = {'PATH': '/usr/bin:/bin', 'HERMES_SANDBOX': 'bwrap'}

    def close(self):
        for fd in self.pass_fds:
            os.close(fd)
        self.pass_fds = ()

    def log_line(self):
        return b'[kanban sandbox] bwrap argv: fake\n'

    def fds_open(self):
        def is_open(fd):
            try:
                os.fstat(fd)
            except OSError:
                return False
            return True
        return [fd for fd in self.held if is_open(fd)]


@pytest.fixture
def spawn(tmp_path, monkeypatch):
    from gateway import session_kanban
    from gateway import session_managed_worker as managed
    from gateway import session_runtime_workers
    from hermes_cli import config
    import tools.process_registry as pr

    root = tmp_path / '.hermes'
    root.mkdir()
    state = SimpleNamespace(calls=[], losses=[], cfg={'worker_memory_max_mb': 8192}, plan=None,
                            scope_available=True, supervised=True)
    original = subprocess.Popen

    def popen(args, **kw):
        state.calls.append((list(args), dict(kw)))
        kw.pop('pass_fds', None)
        return original([sys.executable, '-c', 'import sys; sys.stdin.read()'], **kw)

    def config_read():
        if isinstance(state.cfg, Exception):
            raise state.cfg
        return {'kanban': state.cfg}

    async def sandbox(authority, policy, command, env, cwd):
        assert command == WORKER_ARGV
        return state.plan, b''

    real_which = shutil.which
    monkeypatch.setattr(managed.subprocess, 'Popen', popen)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    monkeypatch.setattr(managed, '_kanban_sandbox', sandbox)
    monkeypatch.setattr(config, 'load_config_readonly', config_read)
    monkeypatch.setattr(pr, '_is_supervised_gateway_process', lambda: state.supervised)
    monkeypatch.setattr(pr, '_systemd_run_user_scope_available', lambda: state.scope_available)
    monkeypatch.setattr(shutil, 'which',
                        lambda name, *a, **k: '/usr/bin/systemd-run' if name == 'systemd-run' else real_which(name, *a, **k))
    monkeypatch.setattr(session_kanban, 'watch_submitter_from_owner', lambda context, closed, kill: None)
    monkeypatch.setattr(session_kanban, 'record_worker_loss',
                        lambda policy, reason, **kw: state.losses.append(reason))
    monkeypatch.setattr(session_runtime_workers, 'persist_work_count', lambda authority: None)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('INVOCATION_ID', 'abc123')

    def run(kanban_context):
        authority = SimpleNamespace(profile_id=str(root), pending_results={}, waiters={})
        policy = SimpleNamespace(kanban_json=json.dumps(kanban_context) if kanban_context is not None else None)
        row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 2, 'payload': {'text': 'go'}}

        async def go():
            try:
                return await managed._spawn_registered_worker(authority, SessionRef('p', 's'), row, policy)
            finally:
                for task in list(getattr(authority, '_mutation_tasks', ())):
                    await task
        spawned = asyncio.run(go())
        if spawned is not None:
            spawned[0].close()
        return spawned
    state.run = run
    return state


def _scope_parts(argv):
    """``(systemd-run options, wrapped command)`` of a scoped argv."""
    split = argv.index('--')
    return argv[:split], argv[split + 1:]


def _property(options, name):
    values = [options[i + 1] for i, a in enumerate(options) if a == '--property' and options[i + 1].startswith(name + '=')]
    assert len(values) == 1, options
    return int(values[0].split('=', 1)[1])


@pytest.mark.platforms('linux')
def test_kanban_turn_runs_in_its_run_turn_scope_with_the_configured_cap(spawn):
    import tools.process_registry as pr
    spawn.run(CONTEXT)
    (argv, kw), = spawn.calls
    options, command = _scope_parts(argv)
    assert options[:4] == ['/usr/bin/systemd-run', '--user', '--scope', '--quiet']
    assert options[options.index('--unit') + 1] == UNIT
    assert _property(options, 'MemoryMax') == pr.explicit_worker_memory_max_bytes(8192 * MiB)
    assert command == WORKER_ARGV
    # Exec'd in place: same pipes and session as the bare worker; the bus env reaches systemd-run.
    assert kw['stdin'] == subprocess.PIPE and kw['stdout'] == subprocess.PIPE and kw['start_new_session'] is True
    assert isinstance(kw['env'], dict) and 'PATH' in kw['env'] and 'pass_fds' not in kw


@pytest.mark.platforms('linux')
def test_sandboxed_turn_wraps_the_plan_and_keeps_its_memfds(spawn):
    spawn.plan = plan = FakePlan()
    held = plan.pass_fds
    spawn.run(CONTEXT)
    (argv, kw), = spawn.calls
    options, command = _scope_parts(argv)
    assert options[options.index('--unit') + 1] == UNIT
    assert command == list(plan.argv)
    assert kw['pass_fds'] == held  # the plan's memfds still ride along
    assert kw['env']['HERMES_SANDBOX'] == 'bwrap'
    assert plan.fds_open() == []  # the spawn closed them once the child held them


@pytest.mark.platforms('linux')
def test_chat_turn_is_never_scoped_and_never_reads_the_cap(spawn):
    spawn.cfg = RuntimeError('a chat turn must not read kanban.worker_memory_max_mb')
    spawn.run(None)
    (argv, kw), = spawn.calls
    assert argv == WORKER_ARGV and kw['env'] is None


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('topology', ['no-invocation-id', 'not-the-supervised-gateway'])
def test_kanban_turn_outside_a_supervised_gateway_is_unchanged(spawn, topology, monkeypatch):
    if topology == 'no-invocation-id':
        monkeypatch.delenv('INVOCATION_ID')
    else:
        spawn.supervised = False
    spawn.run(CONTEXT)
    (argv, kw), = spawn.calls
    assert argv == WORKER_ARGV and kw['env'] is None


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('sandboxed', [False, True])
def test_unavailable_turn_scope_fails_the_admission_closed(spawn, sandboxed):
    spawn.scope_available = False
    spawn.plan = plan = FakePlan() if sandboxed else None
    with pytest.raises(RuntimeStoreError) as raised:
        spawn.run(CONTEXT)
    assert raised.value.reason == 'scope_unavailable'
    assert spawn.calls == []  # never an uncapped turn
    assert spawn.losses == ['scope_unavailable']  # the card's run learns why
    if plan is not None:
        assert plan.fds_open() == []  # the unspawned plan's memfds are not leaked


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('cfg', [{}, {'worker_memory_max_mb': 0}, {'worker_memory_max_mb': 'lots'}])
def test_unset_cap_keeps_the_default_bound(spawn, cfg):
    import tools.process_registry as pr
    spawn.cfg = cfg
    spawn.run(CONTEXT)
    (argv, _kw), = spawn.calls
    options, _command = _scope_parts(argv)
    assert _property(options, 'MemoryMax') == pr._worker_memory_max_bytes()


@pytest.mark.platforms('linux')
def test_unreadable_config_keeps_the_default_bound(spawn):
    """``ConfigBackendUnavailable`` is a SystemExit: it must not escape the spawn thread."""
    import tools.process_registry as pr
    from hermes_cli.config_backend import ConfigBackendUnavailable
    spawn.cfg = ConfigBackendUnavailable('config.yaml unreadable')
    spawn.run(CONTEXT)
    (argv, _kw), = spawn.calls
    assert _property(_scope_parts(argv)[0], 'MemoryMax') == pr._worker_memory_max_bytes()


@pytest.mark.platforms('linux')
def test_managed_worker_exits_when_its_owner_closes_stdin():
    """Outside the gateway cgroup a gateway stop no longer SIGKILLs the turn: it must end on EOF."""
    proc = subprocess.Popen(WORKER_ARGV, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        assert json.loads(proc.stdout.readline())['type'] == 'hello'
        proc.stdin.close()
        assert proc.wait(timeout=60) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
