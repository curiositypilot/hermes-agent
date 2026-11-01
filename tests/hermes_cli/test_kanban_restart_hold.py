"""The scoped Kanban dispatch hold behind ``hermes gateway restart --when-idle`` (t_50ffb4fd).

A hold counts only for the gateway pid it names and only until it expires; anything unreadable is no
hold, so a crashed CLI or a replacement gateway can never stay frozen.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from hermes_cli import kanban_restart_hold as hold


@pytest.fixture(autouse=True)
def kanban_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    return tmp_path


def test_hold_naming_this_pid_with_future_expiry_is_active(kanban_home):
    hold.engage(os.getpid(), 60, "restart test")

    record = hold.active_hold()
    assert hold.active() is True
    assert record is not None and record["reason"] == "restart test"
    assert hold.hold_path() == kanban_home / "restart_hold.json"


def test_expired_hold_is_not_active():
    hold.engage(os.getpid(), -1, "expired")

    assert hold.active() is False


def test_hold_for_another_gateway_pid_does_not_hold_this_one():
    other = os.getpid() + 1
    hold.engage(other, 60, "other gateway")

    assert hold.active() is False
    assert hold.active(other) is True


def test_missing_file_is_no_hold():
    assert not hold.hold_path().exists()
    assert hold.active() is False


@pytest.mark.parametrize("text", ["", "{not json", "[]", "null", '{"gateway_pid": "x", "expires_at": 1}',
                                  json.dumps({"gateway_pid": os.getpid()})])
def test_corrupt_or_partial_file_is_no_hold(kanban_home, text):
    (kanban_home / "restart_hold.json").write_text(text, encoding="utf-8")

    assert hold.active() is False


def test_release_is_idempotent():
    hold.engage(os.getpid(), 60, "release me")
    hold.release()
    hold.release()

    assert hold.active() is False
    assert not hold.hold_path().exists()


def test_engage_writes_the_documented_record():
    before = time.time()
    record = hold.engage(4242, 30, "why")

    on_disk = json.loads(hold.hold_path().read_text(encoding="utf-8"))
    assert on_disk == record
    assert on_disk["gateway_pid"] == 4242 and on_disk["reason"] == "why"
    assert before <= on_disk["engaged_at"] < on_disk["expires_at"] <= time.time() + 30
