"""A cron job's ``memory_query`` is settable through ``hermes cron edit`` and the CLI-only
``cronjob()`` lane, never through the model-facing tool schema (t_05957ba8).

Kanban workers may not import ``cron.jobs`` (live_state_probe_guard), so without these write
paths a worker that had to set a job's recall query could only ask a human to run Python.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return home


def _job(**extra):
    from cron.jobs import create_job, update_job

    job = create_job(prompt="run the fixed weekly procedure", schedule="every 1h")
    if extra:
        update_job(job["id"], extra)
    return job


def test_cli_edit_sets_memory_query_and_prints_it(capsys):
    from cron.jobs import get_job
    from hermes_cli.cron import cron_edit

    job = _job()
    assert cron_edit(SimpleNamespace(job_id=job["id"], memory_query="/no-recall")) == 0
    assert get_job(job["id"])["memory_query"] == "/no-recall"
    assert "Memory query: /no-recall" in capsys.readouterr().out


def test_cli_edit_trims_and_empty_clears_back_to_prompt():
    from cron.jobs import get_job
    from cron.scheduler_prompt import _cron_memory_query
    from hermes_cli.cron import cron_edit

    job = _job()
    assert cron_edit(SimpleNamespace(job_id=job["id"], memory_query="  topic  ")) == 0
    assert get_job(job["id"])["memory_query"] == "topic"

    assert cron_edit(SimpleNamespace(job_id=job["id"], memory_query="")) == 0
    stored = get_job(job["id"])
    assert stored["memory_query"] is None
    assert _cron_memory_query(stored) == "run the fixed weekly procedure"


def test_edit_without_memory_query_keeps_existing_value():
    from cron.jobs import get_job
    from hermes_cli.cron import cron_edit

    job = _job(memory_query="weekly topic")
    assert cron_edit(SimpleNamespace(job_id=job["id"], name="renamed")) == 0
    stored = get_job(job["id"])
    assert stored["name"] == "renamed"
    assert stored["memory_query"] == "weekly topic"


def test_cronjob_update_lane_sets_memory_query():
    from tools.cronjob_tools import cronjob

    job = _job()
    result = json.loads(cronjob(action="update", job_id=job["id"], memory_query="/no-recall"))
    assert result["success"] is True
    assert result["job"]["memory_query"] == "/no-recall"


def test_memory_query_stays_out_of_the_model_schema():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    assert "memory_query" not in CRONJOB_SCHEMA["parameters"]["properties"]


def test_cron_edit_parser_accepts_memory_query_flag():
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=lambda args: 0)
    args = parser.parse_args(["cron", "edit", "abc123", "--memory-query", "/no-recall"])
    assert args.memory_query == "/no-recall"
