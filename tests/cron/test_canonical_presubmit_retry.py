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
