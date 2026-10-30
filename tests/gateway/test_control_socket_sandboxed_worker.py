"""The control socket admits a bwrap-sandboxed managed worker (another user namespace) only as
itself: ``identify`` plus a worker-adoption ticket bound to its host ``(pid, birth)``. Every other
foreign-namespace peer, verb, ticket purpose or selector stays refused (t_49c2353e, slice B of
t_dd989123): a sandboxed agent minting any other ticket could run an unsandboxed turn.

Two layers. The monkeypatched cases drive the real socket from a same-namespace client while
``classify_peer`` reports it as a foreign-namespace peer, so every rule runs on any Linux host. The
bwrap cases run the real ``bwrap --unshare-user`` client and the real WorkerRPC client functions
(``query_identify``, ``_session_ticket``).
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from gateway import control_socket
from gateway.control_socket import GatewayControlServer, resolve_client_socket_path
from gateway.runtime_bootstrap import TicketStore
from hermes_cli.process_identity import worker_birth

pytestmark = pytest.mark.platforms("linux")  # SO_PEERCRED + user namespaces

REPO = Path(__file__).resolve().parents[2]
INSTANCE = "inst-1"


def _bwrap_userns_available() -> bool:
    import shutil
    if not shutil.which("bwrap"):
        return False
    try:
        return subprocess.run(["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
                              capture_output=True, timeout=10, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bwrap = pytest.mark.skipif(not _bwrap_userns_available(), reason="needs bwrap + unprivileged user namespaces")


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "h" / ".hermes"
    d.mkdir(parents=True, mode=0o700)
    return d.resolve()


def _server(home: Path, lookup, handlers=None) -> GatewayControlServer:
    server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 11}, **(handlers or {})})
    server.ticket_store = TicketStore(INSTANCE, frozenset({str(home), str(home.parent)}))
    server.managed_worker_lookup = lookup
    return server


def _ask(home: Path, payload: dict) -> bytes:
    """The raw answer; a refused peer reads ``b""`` (the server closes unread: EOF or reset)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5)
        sock.connect(str(resolve_client_socket_path(home)))
        chunks = []
        try:
            sock.sendall(json.dumps(payload).encode() + b"\n")
            while chunk := sock.recv(65536):
                chunks.append(chunk)
        except (ConnectionResetError, BrokenPipeError):
            pass
        return b"".join(chunks)


def _ticket_request(home: Path, **overrides) -> dict:
    params = {"profile_id": str(home), "instance_id": INSTANCE, "purpose": "worker-adoption", **overrides}
    return {"protocol": 1, "id": 1, "verb": "session-ticket", "params": params}


def _exchange(server: GatewayControlServer, home: Path, payloads: list[dict]) -> list[bytes]:
    async def scenario():
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return [await loop.run_in_executor(None, _ask, home, p) for p in payloads]
        finally:
            await server.stop()
    return asyncio.run(scenario())


def _registered_self(home: Path, *, birth_offset: float = 0.0):
    me = os.getpid()
    birth = worker_birth(psutil.Process(me)) + birth_offset
    return lambda pid: (str(home), (me, birth)) if pid == me else None


@pytest.fixture()
def as_foreign_namespace(monkeypatch):
    """The in-process client reads as a peer from another user namespace."""
    monkeypatch.setattr(control_socket, "classify_peer", lambda _sock: ("worker", os.getpid()))


def test_registered_worker_identifies_and_gets_a_ticket(home, as_foreign_namespace):
    server = _server(home, _registered_self(home))
    identify, ticket = _exchange(server, home, [{"protocol": 1, "id": 1, "verb": "identify"}, _ticket_request(home)])
    assert json.loads(identify)["result"] == {"pid": 11}
    answer = json.loads(ticket)
    assert answer["ok"] is True, answer
    assert answer["result"]["profile_id"] == str(home)


def _mint_and_redeem(home: Path, lookup):
    """The grant a foreign-namespace peer's ticket redeems to, read before the server stops."""
    server = _server(home, lookup)

    async def scenario():
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, _ask, home, _ticket_request(home))
            answer = json.loads(raw)
            assert answer["ok"] is True, answer
            return server.ticket_store.redeem(answer["result"]["ticket"], profile_id=None, purpose="worker-adoption")
        finally:
            await server.stop()
    return asyncio.run(scenario())


def test_the_redeemed_grant_carries_the_workers_host_pid_and_birth(home, as_foreign_namespace):
    grant = _mint_and_redeem(home, _registered_self(home))
    me = os.getpid()
    assert grant["worker_binding"] == (me, worker_birth(psutil.Process(me)))
    assert grant["capabilities"] == frozenset({"worker:adopt"})


def test_an_unregistered_foreign_peer_gets_no_answer(home, as_foreign_namespace):
    assert _exchange(_server(home, lambda pid: None), home, [{"protocol": 1, "id": 1, "verb": "identify"}]) == [b""]


def test_a_standalone_server_without_a_lookup_refuses_foreign_peers(home, as_foreign_namespace):
    assert _exchange(_server(home, None), home, [{"protocol": 1, "id": 1, "verb": "identify"}]) == [b""]


def test_a_registered_pid_with_another_birth_gets_no_answer(home, as_foreign_namespace):
    server = _server(home, _registered_self(home, birth_offset=1.0))
    assert _exchange(server, home, [{"protocol": 1, "id": 1, "verb": "identify"}, _ticket_request(home)]) == [b"", b""]


@pytest.mark.parametrize("purpose", ["interactive", "native-http", "native-profile-create",
                                     "native-profile-registry", "exposure"])
def test_a_registered_worker_cannot_mint_any_other_purpose(home, as_foreign_namespace, purpose):
    answer, = _exchange(_server(home, _registered_self(home)), home, [_ticket_request(home, purpose=purpose)])
    assert json.loads(answer) == {"ok": False, "error": "PermissionError: sandboxed worker: ticket refused",
                                  "protocol": 1, "id": 1}


@pytest.mark.parametrize("selector", ["subject", "host-scope", "other-profile"])
def test_a_registered_worker_cannot_name_a_subject_host_scope_or_another_profile(home, as_foreign_namespace, selector):
    override = {"subject": {"subject": 'auth:v1:["basic","","someone"]'}, "host-scope": {"scope": "host"},
                # served by this store, but not the worker's own profile
                "other-profile": {"profile_id": str(home.parent)}}[selector]
    answer, = _exchange(_server(home, _registered_self(home)), home, [_ticket_request(home, **override)])
    assert json.loads(answer)["error"] == "PermissionError: sandboxed worker: ticket refused"


def test_a_registered_worker_cannot_call_any_other_verb(home, as_foreign_namespace):
    calls = []
    server = _server(home, _registered_self(home), handlers={
        "pause-for-update": lambda: calls.append("pause") or {"pausing": True},
        "status": lambda: calls.append("status") or {}})
    server.private_handlers["hosted-attest"] = lambda params, subject: calls.append("private") or {}
    answers = _exchange(server, home, [
        {"protocol": 1, "id": 1, "verb": verb, "params": {}} for verb in ("status", "pause-for-update", "hosted-attest")])
    assert [json.loads(a)["error"] for a in answers] == ["PermissionError: sandboxed worker: verb refused"] * 3
    assert calls == []


def test_same_namespace_clients_are_unchanged(home):
    """The real classifier: an ordinary client never consults the lookup and keeps every verb."""
    def lookup(pid):
        raise AssertionError("a same-namespace peer must not reach the worker lookup")
    identify, status, ticket = _exchange(_server(home, lookup, handlers={"status": lambda: {"ok": 1}}), home, [
        {"protocol": 1, "id": 1, "verb": "identify"}, {"protocol": 1, "id": 1, "verb": "status"},
        _ticket_request(home, purpose="interactive")])
    assert json.loads(identify)["result"] == {"pid": 11}
    assert json.loads(status)["result"] == {"ok": 1}
    assert json.loads(ticket)["ok"] is True


# --- the real sandbox ---------------------------------------------------------------------------

_SANDBOXED_CLIENT = """
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {repo!r})
home = Path({home!r})
out = {{"pid": os.getpid()}}
try:
    from hermes_cli.gateway_runtime_discovery import query_identify
    out["identify"] = query_identify(home, timeout=5)
    from hermes_cli.gateway_client import _session_ticket
    endpoint = SimpleNamespace(profile_id=str(home), instance_id={instance!r}, control_home=None)
    out["ticket"] = _session_ticket(home, endpoint, purpose="worker-adoption")
except Exception as exc:
    out["error"] = type(exc).__name__
sys.stdout.write(json.dumps(out))
"""

_RAW_CLIENT = (
    "import socket,sys;s=socket.socket(socket.AF_UNIX);s.settimeout(5);s.connect({path!r});"
    "s.sendall({line!r});sys.stdout.write(s.recv(65536).decode())")


async def _in_bwrap(code: str) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "bwrap", "--unshare-user", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        sys.executable, "-c", code, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _err = await asyncio.wait_for(proc.communicate(), timeout=60)
    return out


def _host_birth_lookup(home: Path, seen: list, *, birth_offset: float = 0.0):
    """The managed-worker table as the owner would hold it for the connecting client: its host
    pid (what SO_PEERCRED reports) and its kernel birth, read from the proc handle."""
    def lookup(pid):
        seen.append(pid)
        return str(home), (pid, worker_birth(psutil.Process(pid)) + birth_offset)
    return lookup


@needs_bwrap
def test_a_registered_bwrap_worker_runs_the_real_worker_rpc_handshake(home):
    seen = []
    server = _server(home, _host_birth_lookup(home, seen))

    async def scenario():
        assert await server.start()
        try:
            out = json.loads(await _in_bwrap(_SANDBOXED_CLIENT.format(repo=str(REPO), home=str(home), instance=INSTANCE)))
            grant = (server.ticket_store.redeem(out["ticket"], profile_id=None, purpose="worker-adoption")
                     if "ticket" in out else None)
            return out, grant
        finally:
            await server.stop()

    out, grant = asyncio.run(scenario())
    assert "error" not in out, out
    assert out["identify"] == {"pid": 11}
    assert seen and set(seen) == {out["pid"]}  # no pid namespace: the worker's own pid is its host pid
    assert grant["worker_binding"][0] == out["pid"]
    assert grant["purpose"] == "worker-adoption"


@needs_bwrap
@pytest.mark.parametrize("registered", ["absent", "wrong-birth"])
def test_an_unregistered_or_reborn_bwrap_peer_gets_no_answer(home, registered):
    lookup = (lambda pid: None) if registered == "absent" else _host_birth_lookup(home, [], birth_offset=1.0)
    server = _server(home, lookup)

    async def scenario():
        assert await server.start()
        try:
            raw = await _in_bwrap(_RAW_CLIENT.format(path=str(resolve_client_socket_path(home)),
                                                     line=b'{"protocol":1,"verb":"identify","id":1}\n'))
            handshake = json.loads(await _in_bwrap(
                _SANDBOXED_CLIENT.format(repo=str(REPO), home=str(home), instance=INSTANCE)))
            return raw, handshake
        finally:
            await server.stop()

    raw, handshake = asyncio.run(scenario())
    assert raw == b""
    assert "identify" not in handshake and handshake.get("error"), handshake


@needs_bwrap
def test_a_registered_bwrap_worker_is_refused_every_other_verb(home):
    server = _server(home, _host_birth_lookup(home, []), handlers={"status": lambda: {"ok": 1}})

    async def scenario():
        assert await server.start()
        try:
            return await _in_bwrap(_RAW_CLIENT.format(path=str(resolve_client_socket_path(home)),
                                                      line=b'{"protocol":1,"verb":"status","id":1}\n'))
        finally:
            await server.stop()

    assert json.loads(asyncio.run(scenario()))["error"] == "PermissionError: sandboxed worker: verb refused"


def test_classify_peer_reports_a_same_namespace_socketpair_as_same():
    ours, theirs = socket.socketpair(socket.AF_UNIX)
    with ours, theirs:
        assert control_socket.classify_peer(ours) == ("same", None)
        assert control_socket.classify_peer(None) == ("same", None)
