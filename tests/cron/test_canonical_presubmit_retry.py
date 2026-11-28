"""A canonical fire that finds the gateway draining or starting (nothing submitted) joins the
unreachable-model ladder instead of failing loudly on the first occurrence.

Regression: a sibling cron worker's launch-time dependency sync held the checkout lock for ~18 s,
``ensure_gateway_runtime`` answered ``draining/update_paused`` and the fire was booked as a plain
FAILED run whose notice went straight out.
"""
from contextlib import asynccontextmanager
import json

import pytest

from cron import jobs, scheduler_authority
from cron.unreachable_retry import is_gateway_unavailable_presubmit, will_retry
from hermes_cli import gateway_client
from hermes_cli.gateway_client import GatewayUnavailableError

TRANSIENT = [("Gateway draining: update_paused", "draining", "update_paused"),
             ("Gateway starting: deadline", "starting", "deadline")]
PERMANENT = [("Gateway incompatible", "incompatible", "runtime_protocol"),
             ("Gateway inaccessible", "inaccessible", "control_timeout"),
             ("Gateway absent", "absent", None)]


def refuse_with(monkeypatch, exc):
    @asynccontextmanager
    async def connect():
        raise exc
        yield  # pragma: no cover - never reached
    monkeypatch.setattr(gateway_client, "connect_gateway", connect)


@pytest.mark.parametrize("args", TRANSIENT, ids=lambda a: a[1])
def test_transient_presubmit_unavailable_joins_the_ladder(tmp_path, monkeypatch, args):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        refuse_with(monkeypatch, GatewayUnavailableError(*args))
        success, output, _, error = scheduler_authority.run_canonical_job(job)
        assert success is False and args[0] in error and "(FAILED)" in output
        assert job.get("_model_unreachable") is True
        # The flag is what the bookkeeping tail reads: the ladder will re-run and hold the notice.
        assert will_retry(job)


@pytest.mark.parametrize("exc", [GatewayUnavailableError(*a) for a in PERMANENT]
                         + [RuntimeError("boom")], ids=lambda e: getattr(e, "state", None) or "runtime")
def test_permanent_or_foreign_failures_still_alert_at_once(tmp_path, monkeypatch, exc):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        refuse_with(monkeypatch, exc)
        success, *_ = scheduler_authority.run_canonical_job(job)
        assert success is False
        assert "_model_unreachable" not in job


def test_failure_after_an_attempt_stays_unknown_and_unflagged(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        params = {"job_id": job["id"], "request_id": "fire-1", "extra_prompt": None}
        journal = scheduler_authority.journal_path(job["id"], "fire-1")
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({"params": params, "receipt": None}))
        refuse_with(monkeypatch, GatewayUnavailableError(*TRANSIENT[0]))
        with pytest.raises(scheduler_authority.CronExecutionUnknown):
            scheduler_authority.run_canonical_job(job, execution_id="fire-1")
        assert "_model_unreachable" not in job


@pytest.mark.parametrize("exc, expected", [
    *[(GatewayUnavailableError(*a), True) for a in TRANSIENT],
    *[(GatewayUnavailableError(*a), False) for a in PERMANENT],
    (gateway_client.GatewayClientError("disconnected"), False),
    (RuntimeError("draining"), False),
])
def test_presubmit_predicate(exc, expected):
    assert is_gateway_unavailable_presubmit(exc) is expected


# A restart's after-turn wait (incident t_dc001b38): the owner refuses new work, so a fire that
# meets it must re-run on the ladder instead of being paused as unverified and later booked FAILED
# with "admission missing".

def test_fire_dials_outside_restart_wait(tmp_path, monkeypatch):
    seen = []

    @asynccontextmanager
    async def connect():
        seen.append(gateway_client._REFUSE_RESTART_WAIT.get())
        raise GatewayUnavailableError("Gateway draining: restart_wait", "draining", "restart_wait")
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        success, *_ = scheduler_authority.run_canonical_job(job)
        assert seen == [True]
        assert success is False and job.get("_model_unreachable") is True
        assert will_retry(job)


def test_handshake_refusal_before_submit_joins_ladder(tmp_path, monkeypatch):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        refuse_with(monkeypatch, InvalidStatus(Response(403, "Forbidden", Headers())))
        success, *_ = scheduler_authority.run_canonical_job(job)
        assert success is False and job.get("_model_unreachable") is True


class _Client:
    """A firer's socket whose owner answers every verb from *answers* (an exception is raised)."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def rpc(self, method, **params):
        self.calls.append(method)
        answer = self.answers[method]
        if isinstance(answer, Exception):
            raise answer
        return answer


def serve(monkeypatch, client):
    @asynccontextmanager
    async def connect():
        yield client
    monkeypatch.setattr(gateway_client, "connect_gateway", connect)


@pytest.mark.parametrize("reason, ladder", [("runtime_draining", True), ("permission_denied", False)])
def test_draining_submit_refusal_with_missing_recover_joins_ladder(tmp_path, monkeypatch, reason, ladder):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="inbox", schedule="every 1h", deliver="local")
        client = _Client({"cron.submit": gateway_client.GatewayRPCError(reason),
                          "cron.recover": {"status": "missing"}})
        serve(monkeypatch, client)
        success, _, _, error = scheduler_authority.run_canonical_job(job)
        assert success is False and reason in error
        assert client.calls == ["cron.submit", "cron.recover"]
        # Any other refusal stays the loud FAILED run it was.
        assert (job.get("_model_unreachable") is True) is ladder


def test_incident_287b3be7_restart_wait_fire_is_not_paused(tmp_path, monkeypatch):
    """The 00:01 fire of kanban-orchestrator: its socket reached an owner in restart wait as an
    operator connection. The owner refuses the submit and must still answer the recover."""
    import uuid
    from types import SimpleNamespace

    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    job_id, request_id = "580e25b75ad8", "287b3be72e2c473398d97a6f3661cfe4"

    class Owner:
        async def rpc(self, method, **params):
            reply = await self.conn.dispatch({"id": 1, "method": method, "params": params})
            if "error" in reply:
                raise gateway_client.rpc_error(reply["error"])
            return reply["result"]

    owner = Owner()

    @asynccontextmanager
    async def connect():
        db = SessionDB(tmp_path / "state.db")
        runner = SimpleNamespace(_draining=True, _running=True,
                                 session_runtime_descriptor={"state": "draining", "drain_reason": "restart_wait"})
        authority = SessionAuthority(runner, profile_id="p", instance_id="owner", db=db,
                                     epoch=begin_runtime_epoch(db, instance_id="owner"))
        owner.conn = AuthorityConnection(authority, object(), {"user_id": "human"}, operator=True)
        try:
            yield owner
        finally:
            await owner.conn.close()
            db.close()

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    with jobs.use_cron_store(tmp_path / "cron"):
        with monkeypatch.context() as m:
            m.setattr(jobs, "uuid", SimpleNamespace(uuid4=lambda: uuid.UUID(job_id + "0" * 20)))
            job = jobs.create_job(prompt="brief", schedule="every 15m", deliver="local")
        assert job["id"] == job_id
        success, _, _, error = scheduler_authority.run_canonical_job(job, execution_id=request_id)
        assert success is False and "runtime_draining" in error
        assert job.get("_model_unreachable") is True and will_retry(job)
        assert jobs.get_job(job_id)["state"] != "paused"
