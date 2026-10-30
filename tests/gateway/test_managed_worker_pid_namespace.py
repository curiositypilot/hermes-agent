"""A managed worker inside its own pid namespace (bwrap ``--unshare-pid``) is identified by its owner.

The worker reports its namespace-local pid (2 under bwrap) in ``hello``. The owner maps it to the
host pid of the one descendant of its handle whose innermost ``NSpid`` and birth match, mints that
host pid as ``scope['pid']`` (fences, psutil, the adopt claim) and the reported one as
``scope['ns_pid']`` (what the worker checks against ``getpid()``). Regression for intake t_dd989123.
"""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from agent.managed_worker import adoption_scope
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

ROOT = str(Path(__file__).resolve().parents[2])
BWRAP = ['bwrap', '--unshare-user', '--unshare-pid', '--die-with-parent',
         '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev']
# The real worker's own hello(), run inside the namespace; it then blocks on stdin like a worker.
INNER = ('import json, sys\n'
         f'sys.path.insert(0, {ROOT!r})\n'
         'from agent.managed_worker import hello\n'
         'print(json.dumps(hello()), flush=True)\n'
         'sys.stdin.read()\n')
# A launcher between the owner's handle and bwrap: handle -> bwrap monitor -> ns init -> python.
LAUNCHER = ('import subprocess, sys\n'
            'children = [subprocess.Popen(sys.argv[2:], stdin=sys.stdin) for _ in range(int(sys.argv[1]))]\n'
            'sys.exit(max(c.wait() for c in children))\n')


def _bwrap_usable():
    if shutil.which('bwrap') is None:
        return False
    probe = subprocess.run([*BWRAP, sys.executable, '-c', 'pass'], capture_output=True, timeout=30, check=False)
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(not _bwrap_usable(), reason='bwrap missing or unprivileged user namespaces off')


def _inner_argv():
    return [*BWRAP, sys.executable, '-c', INNER]


def _spawn(argv):
    return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _hello(process):
    return {'type': 'hello', **json.loads(process.stdout.readline())}


def _ns_chain(pid):
    with open(f'/proc/{pid}/status', encoding='ascii') as status:
        return next(line.split()[1:] for line in status if line.startswith('NSpid:'))


def _stop(*processes):
    # Innermost first, while each is still inside this test's subtree (the conftest kill guard).
    for process in processes:
        for child in reversed(psutil.Process(process.pid).children(recursive=True)):
            try:
                child.kill()
            except psutil.NoSuchProcess:  # bwrap's init exits with its pid 2
                pass
    for process in processes:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


@pytest.mark.platforms('linux')
@needs_bwrap
@pytest.mark.asyncio
async def test_sandboxed_worker_maps_to_its_host_pid_and_adopts(tmp_path):
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker, verify_worker_identity
    from hermes_cli.process_identity import worker_birth
    process = _spawn(_inner_argv())
    try:
        hello = _hello(process)
        host_pid, birth = verify_worker_identity(process, hello)
        assert host_pid != hello['pid'] and host_pid in [p.pid for p in psutil.Process(process.pid).children(recursive=True)]
        assert _ns_chain(host_pid) == [str(host_pid), str(hello['pid'])]
        assert birth == hello['birth'] == worker_birth(psutil.Process(host_pid))
        with closing(SessionDB(tmp_path / 'state.db')) as db:
            db.create_session('owned', 'cli')
            epoch = begin_runtime_epoch(db, instance_id='owner')
            authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
            admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
            claim_session_input(db, epoch=epoch, session_id='owned')
            scope = reserve_admission_worker(authority, admission_id=admitted['admission_id'], process=process,
                                             principal_id='human', hello=hello)
            assert (scope['pid'], scope['ns_pid'], scope['birth']) == (host_pid, hello['pid'], birth)
            # The worker presents the host pid its owner verified; the claim binds exactly that.
            actor = SimpleNamespace(subject='human', profile_id=str(tmp_path), capabilities={'worker:adopt'})
            connection = SimpleNamespace(authority=authority, actor=actor)
            identity = adoption_scope({k: v for k, v in scope.items() if k != 'epoch'})
            receipt = await worker_request(connection, None, identity, operation='adopt')
            assert receipt['execution_id'] == scope['execution_id']
    finally:
        _stop(process)


@pytest.mark.platforms('linux')
@needs_bwrap
def test_namespace_pid_outside_the_handle_tree_never_matches():
    from gateway.session_worker_reservation import verify_worker_identity
    handle = _spawn([sys.executable, '-c', 'import sys; sys.stdin.read()'])
    stranger = _spawn(_inner_argv())
    try:
        # Alive, truthful birth, the reported pid is its innermost NSpid: but not under the handle.
        with pytest.raises(Exception, match='permission_denied'):
            verify_worker_identity(handle, _hello(stranger))
    finally:
        _stop(handle, stranger)


@pytest.mark.platforms('linux')
@needs_bwrap
def test_namespaced_descendant_with_a_wrong_birth_never_matches():
    from gateway.session_worker_reservation import verify_worker_identity
    process = _spawn(_inner_argv())
    try:
        hello = _hello(process)
        with pytest.raises(Exception, match='permission_denied'):
            verify_worker_identity(process, {**hello, 'birth': hello['birth'] + 1})
    finally:
        _stop(process)


@pytest.mark.platforms('linux')
@needs_bwrap
def test_two_matching_namespaced_descendants_are_refused(monkeypatch):
    from gateway.session_worker_reservation import verify_worker_identity
    import hermes_cli.process_identity as identity
    # Two namespaces each hold a pid 2; equal births need the same start tick, so pin the birth.
    monkeypatch.setattr(identity, 'worker_birth', lambda proc: 1.0)
    single = _spawn([sys.executable, '-c', LAUNCHER, '1', *_inner_argv()])
    double = _spawn([sys.executable, '-c', LAUNCHER, '2', *_inner_argv()])
    try:
        # One launcher hop above bwrap still resolves (handle -> monitor -> ns init -> python).
        hello = _hello(single) | {'birth': 1.0}
        host_pid, _ = verify_worker_identity(single, hello)
        assert _ns_chain(host_pid)[-1] == str(hello['pid'])
        first, second = _hello(double), _hello(double)
        assert first['pid'] == second['pid']
        with pytest.raises(Exception, match='permission_denied'):
            verify_worker_identity(double, first | {'birth': 1.0})
    finally:
        _stop(single, double)


def _frame(tmp_path, monkeypatch, scope):
    from gateway import session_managed_worker as smw
    from gateway.session_policy import build_policy
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'},
                          {'gateway': {'managed_workers': True}})
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='route')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    frame = smw._bootstrap(authority, SimpleNamespace(session_id='sid'), {'payload': {'text': 'hi'}}, policy, scope)
    return json.loads(json.dumps(frame))


def test_bootstrap_checks_ns_pid_against_getpid_not_the_host_pid(tmp_path, monkeypatch):
    from agent.managed_worker import validate_bootstrap
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid() + 100000, 'ns_pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    wire = _frame(tmp_path, monkeypatch, scope)
    assert validate_bootstrap(wire) is wire
    assert adoption_scope(wire['scope']) == {k: v for k, v in scope.items() if k != 'ns_pid'}
    with pytest.raises(ValueError):
        validate_bootstrap(_frame(tmp_path, monkeypatch, scope | {'ns_pid': os.getpid() + 1}))
    no_ns_pid = {k: v for k, v in scope.items() if k != 'ns_pid'} | {'pid': os.getpid()}
    with pytest.raises(ValueError):
        validate_bootstrap(_frame(tmp_path, monkeypatch, no_ns_pid))


def test_adopted_worker_in_a_pid_namespace_binds_its_delegation_ledger(tmp_path, monkeypatch):
    # After adopt, a namespaced worker's scope carries its host pid, which getpid() never returns.
    from agent import runtime_session_store
    from agent.runtime_session_store import RuntimeSessionStore
    from tools import async_delegation_worker as worker
    host_pid = os.getpid() + 100000
    monkeypatch.setattr(runtime_session_store, '_owner_view_pid', None)
    runtime_session_store.bind_owner_view_pid(host_pid)
    monkeypatch.setattr(runtime_session_store, '_worker_process', True)
    monkeypatch.setattr(worker, '_store', None)
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': host_pid, 'birth': 0, 'secret': 's', 'epoch': 1}
    store = RuntimeSessionStore(lambda method, **params: pytest.fail(method), scope, tmp_path / 'outbox')
    try:
        assert worker.worker_ledger().store is store
    finally:
        store._outbox_owner.close()
