"""A canonical fire that a restart drain refuses is verified, booked as never admitted and joins
the unreachable-model ladder instead of pausing the job.

Regression (intake t_0ad97f36): a gateway restart drain refused ``cron.submit`` AND the
``cron.recover`` that verifies the refusal on the worker socket. The firer raised
``CronExecutionUnknown``, paused the job, the restart-safe worker exited 1 and the next slot was
lost; reconcile later sent a second notice for the same fire.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import threading
from types import SimpleNamespace

import pytest

from cron import jobs, scheduler_authority
from cron.unreachable_retry import is_owner_draining_refusal, will_retry
from gateway import session_cron
from gateway.drain_control import ws_method_refused_while_draining
from hermes_cli import gateway_client
from hermes_cli.gateway_client import GatewayRPCError, GatewayUnavailableError
from hermes_state_runtime import RuntimeStoreError

OBSERVE_VERBS = ("cron.recover", "cron.status", "cron.cancel")
OTHER_REFUSALS = ("permission_denied", "runtime_retired", "admission_conflict")


@pytest.mark.parametrize("method", OBSERVE_VERBS)
def test_drain_gate_lets_cron_observe_verbs_through(method):
    assert ws_method_refused_while_draining(SimpleNamespace(_draining=True), method) is False


@pytest.mark.parametrize("method", ("cron.submit", "session.create"))
def test_drain_gate_still_fences_new_work(method):
    assert ws_method_refused_while_draining(SimpleNamespace(_draining=True), method) is True


@pytest.mark.parametrize("method", ("cron.submit", "session.create", *OBSERVE_VERBS))
def test_drain_gate_refuses_nothing_when_not_draining(method):
    assert ws_method_refused_while_draining(SimpleNamespace(_draining=False), method) is False


def socket_owner(monkeypatch, submit_exc, recover):
    """Patch the gateway socket: ``cron.submit`` raises *submit_exc*; ``cron.recover`` calls *recover*."""
    frames = []

    class Peer:
        async def rpc(self, method, **params):
            frames.append(method)
            if method == "cron.submit":
                raise submit_exc
            if method == "cron.recover":
                return recover()
            raise AssertionError(f"unexpected frame {method}")

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, "connect_gateway", connected)
    return frames


@pytest.fixture
def owner_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def in_process_owner(monkeypatch, loop, submit_exc):
    """Patch an in-process owner whose ``operation`` refuses submit and verifies ``missing``."""
    calls = []

    async def operation(authority, name, params, actor=None):
        calls.append(name)
        if name == "submit":
            raise submit_exc
        if name == "recover":
            return {"status": "missing"}
        raise AssertionError(f"unexpected operation {name}")

    monkeypatch.setattr(session_cron, "owner_for_home", lambda home: (object(), loop))
    monkeypatch.setattr(session_cron, "operation", operation)
    return calls


def journal_record(job, execution_id):
    return json.loads(scheduler_authority.journal_path(job["id"], execution_id).read_text())


def assert_drain_refusal_joins_ladder(job, result, execution_id):
    success, output, _, error = result
    assert success is False and "runtime_draining" in error and "(FAILED)" in output
    assert job.get("_model_unreachable") is True
    record = journal_record(job, execution_id)
    assert record["refusal"][0] is False
    assert record["job"]["_model_unreachable"] is True
    # The flag is what the bookkeeping tail reads: the ladder re-runs and holds the notice.
    assert will_retry(job)


def test_socket_drain_refusal_is_verified_and_joins_the_ladder(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        frames = socket_owner(monkeypatch, GatewayRPCError("runtime_draining"),
                              lambda: {"status": "missing"})
        result = scheduler_authority.run_canonical_job(job, execution_id="fire-1")
        assert frames == ["cron.submit", "cron.recover"]
        assert_drain_refusal_joins_ladder(job, result, "fire-1")


def test_in_process_drain_refusal_is_verified_and_joins_the_ladder(tmp_path, monkeypatch, owner_loop):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        calls = in_process_owner(monkeypatch, owner_loop, RuntimeStoreError("runtime_draining"))
        result = scheduler_authority.run_canonical_job(job, execution_id="fire-1")
        assert calls == ["submit", "recover"]
        assert_drain_refusal_joins_ladder(job, result, "fire-1")


@pytest.mark.parametrize("reason", OTHER_REFUSALS)
def test_other_verified_refusals_fail_without_the_flag(tmp_path, monkeypatch, reason):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        socket_owner(monkeypatch, GatewayRPCError(reason), lambda: {"status": "missing"})
        success, _, _, error = scheduler_authority.run_canonical_job(job, execution_id="fire-1")
        assert success is False and reason in error
        assert "_model_unreachable" not in job
        assert "_model_unreachable" not in journal_record(job, "fire-1")["job"]


def test_drain_refusal_whose_recover_fails_stays_unknown(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")

        def recover():
            raise GatewayRPCError("runtime_draining")

        socket_owner(monkeypatch, GatewayRPCError("runtime_draining"), recover)
        with pytest.raises(scheduler_authority.CronExecutionUnknown):
            scheduler_authority.run_canonical_job(job, execution_id="fire-1")
        assert "_model_unreachable" not in job


@pytest.mark.parametrize("exc, expected", [
    (GatewayRPCError("runtime_draining"), True),
    (RuntimeStoreError("runtime_draining"), True),
    *[(GatewayRPCError(r), False) for r in OTHER_REFUSALS],
    *[(RuntimeStoreError(r), False) for r in OTHER_REFUSALS],
    (gateway_client.GatewayClientError("runtime_draining"), False),
    (GatewayUnavailableError("Gateway draining", "draining", "update_paused"), False),
    (RuntimeError("runtime_draining"), False),
])
def test_owner_draining_refusal_predicate(exc, expected):
    assert is_owner_draining_refusal(exc) is expected
