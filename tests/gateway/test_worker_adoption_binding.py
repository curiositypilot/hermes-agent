"""A worker-adoption ticket minted for a sandboxed managed worker binds the connection to that
worker's host ``(pid, birth)`` all the way to ``session_worker._claim`` (t_49c2353e): the grant
carries it, ``bootstrap_auth_identity`` passes it, ``AuthorityConnection`` exposes it, and a bound
connection can neither register nor claim another process. An unbound connection is unchanged.
Real TicketStore, real SessionAuthority over SQLite, real processes.
"""
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from gateway.runtime_bootstrap import TicketStore
from hermes_cli.process_identity import worker_birth


def _store():
    return TicketStore("owner", frozenset({"p"}))


def test_only_a_worker_adoption_ticket_may_carry_a_binding():
    for purpose in ("interactive", "native-http", "native-profile-create", "native-profile-registry", "exposure"):
        with pytest.raises(PermissionError):
            _store().mint(profile_id="p", subject="uid:1", purpose=purpose, worker_binding=(10, 1.0))


def test_an_unbound_grant_is_exactly_what_it_was():
    store = _store()
    ticket = store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption")
    assert "worker_binding" not in store.redeem(ticket, profile_id=None, purpose="worker-adoption")


def test_one_worker_cannot_exhaust_the_shared_ticket_store():
    store = _store()
    for _ in range(TicketStore.MAX_PER_WORKER):
        store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption", worker_binding=(10, 1.0))
    with pytest.raises(PermissionError, match="for this worker"):
        store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption", worker_binding=(10, 1.0))
    # Other workers and every other client still mint.
    store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption", worker_binding=(11, 1.0))
    store.mint(profile_id="p", subject="uid:1", purpose="interactive")
    store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption")


def test_a_redeemed_ticket_frees_its_workers_slot():
    store = _store()
    tickets = [store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption", worker_binding=(10, 1.0))
               for _ in range(TicketStore.MAX_PER_WORKER)]
    store.redeem(tickets[0], profile_id=None, purpose="worker-adoption")
    store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption", worker_binding=(10, 1.0))


# --- find_managed_worker ------------------------------------------------------------------------

def _managed(worker=None, closed=False):
    from gateway.session_managed_worker import ManagedWorker
    managed = ManagedWorker(SimpleNamespace(pid=1))
    managed.worker = worker
    if closed:
        managed.closed.set()
    return managed


def _runner(*authorities):
    return SimpleNamespace(session_authorities=list(authorities))


def _authority(profile_id, **workers):
    return SimpleNamespace(profile_id=profile_id, _managed_workers=workers)


def test_find_managed_worker_returns_the_live_reserved_worker_and_its_profile():
    from gateway.session_managed_worker import find_managed_worker
    runner = _runner(_authority("a", s1=_managed((100, 1.5))), _authority("b", s2=_managed((200, 2.5))))
    assert find_managed_worker(runner, 200) == ("b", (200, 2.5))
    assert find_managed_worker(runner, 100) == ("a", (100, 1.5))


@pytest.mark.parametrize("case", ["wrong-pid", "closed", "not-reserved-yet", "no-workers"])
def test_find_managed_worker_finds_nothing_else(case):
    from gateway.session_managed_worker import find_managed_worker
    authority = {
        "wrong-pid": _authority("a", s=_managed((100, 1.5))),
        "closed": _authority("a", s=_managed((200, 1.5), closed=True)),
        "not-reserved-yet": _authority("a", s=_managed(None)),  # still saying hello
        "no-workers": SimpleNamespace(profile_id="a"),
    }[case]
    assert find_managed_worker(_runner(authority), 200) is None


def test_find_managed_worker_reads_a_standalone_runners_single_authority():
    from gateway.session_managed_worker import find_managed_worker
    runner = SimpleNamespace(session_authority=_authority("solo", s=_managed((300, 3.0))))
    assert find_managed_worker(runner, 300) == ("solo", (300, 3.0))


# --- the redeem -> identity -> AuthorityConnection -> _claim chain ------------------------------

@pytest.fixture()
def owner(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("owned", "telegram")
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id="p", instance_id="owner", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="owner"))
    authority.sessions["owned"] = LiveSession(SimpleNamespace(platform=Platform.TELEGRAM, user_id="human"), "route")
    children = [subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(2)]
    try:
        yield authority, db, children
    finally:
        for child in children:
            child.stdin.close()
            child.wait(timeout=5)
        db.close()


def _connection(authority, *, binding):
    """What run_api builds for a redeemed worker-adoption ticket."""
    from gateway.run_api import bootstrap_auth_identity
    from gateway.session_controls import AuthorityConnection
    store = TicketStore("owner", frozenset({"p"}))
    ticket = store.mint(profile_id="p", subject="uid:1", purpose="worker-adoption", worker_binding=binding)
    grant = store.redeem(ticket, profile_id=None, purpose="worker-adoption")
    return AuthorityConnection(authority, SimpleNamespace(), bootstrap_auth_identity(grant))


def _scope(pid, **extra):
    return {"profile_id": "p", "session_id": "owned", "execution_id": f"compute-{pid}", "generation": 0,
            "pid": pid, "birth": worker_birth(psutil.Process(pid)), "secret": "w" * 32, **extra}


def _identity(pid):
    return pid, worker_birth(psutil.Process(pid))


@pytest.mark.asyncio
async def test_a_bound_connection_carries_its_binding_and_cannot_register(owner):
    from gateway.session_contract import SessionRef
    from gateway.session_worker import worker_request
    from hermes_state_runtime import RuntimeStoreError
    authority, db, (child, _other) = owner
    connection = _connection(authority, binding=_identity(child.pid))
    assert connection.worker_binding == _identity(child.pid)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await worker_request(connection, SessionRef("p", "owned"), _scope(child.pid, kind="compute"), operation="register")
    assert db._read_one("SELECT COUNT(*) FROM worker_executions")[0] == 0


@pytest.mark.asyncio
async def test_a_bound_connection_claims_only_its_own_process(owner):
    from gateway.session_contract import SessionRef
    from gateway.session_worker import worker_request
    from hermes_state_runtime import RuntimeStoreError
    authority, _db, (child, other) = owner
    connection = _connection(authority, binding=_identity(child.pid))
    ref = SessionRef("p", "owned")
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await worker_request(connection, ref, _scope(other.pid), operation="adopt")
    reborn = dict(_scope(child.pid), birth=_identity(child.pid)[1] + 1)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await worker_request(connection, ref, reborn, operation="adopt")
    # Its own process passes the binding and reaches the reservation check (none exists here).
    with pytest.raises(RuntimeStoreError, match="not_found"):
        await worker_request(connection, ref, _scope(child.pid), operation="adopt")


@pytest.mark.asyncio
async def test_an_unbound_connection_takes_todays_path(owner):
    from gateway.session_contract import SessionRef
    from gateway.session_worker import worker_request
    authority, _db, (child, _other) = owner
    connection = _connection(authority, binding=None)
    assert connection.worker_binding is None
    registered = await worker_request(connection, SessionRef("p", "owned"), _scope(child.pid, kind="compute"),
                                      operation="register")
    assert registered["status"] == "registered"

