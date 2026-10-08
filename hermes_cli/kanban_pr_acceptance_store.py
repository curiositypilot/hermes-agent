"""Persist acceptance with the same ownership snapshot as the terminal write.

Two contract kinds gate a terminal transition here: PR contracts
(``OWNER/REPO`` / PR URL, :mod:`hermes_cli.kanban_pr_acceptance`) and local
test contracts (``test:<cmd>``, :mod:`hermes_cli.kanban_test_gate`). Evidence
is collected outside SQLite transactions; ``record_acceptance`` re-checks the
captured run/status/contract under the caller's write lock.
"""
from __future__ import annotations

import json

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance
from hermes_cli.kanban_test_gate import is_test_contract, refusal_detail, run_test_contract

_GATED_STATUSES = {"running", "ready", "blocked", "review"}


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def _run_test_gate(conn, task_id, snapshot):
    row = conn.execute("SELECT workspace_path FROM tasks WHERE id=?", (task_id,)).fetchone()
    return snapshot, run_test_contract(snapshot[2], row["workspace_path"] if row else None)


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract = snapshot
    if not contract or contract == "local-only":
        return None
    if status not in _GATED_STATUSES or (expected_run_id is not None and run_id != expected_run_id):
        return False
    if is_test_contract(contract):
        return _run_test_gate(conn, task_id, snapshot)
    published_pr = metadata.get("published_pr") if isinstance(metadata, dict) else None
    match = _PR.fullmatch(published_pr) if isinstance(published_pr, str) else None
    # Publication binds once. Retrying cannot replace the task's PR with a green sibling.
    if match and contract == match[1]:
        with write_txn(conn):
            if _snapshot(conn, task_id) != snapshot:
                return False
            conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (published_pr, task_id))
        snapshot = (run_id, status, published_pr)
        contract = published_pr
    # The assignee profile's gh login owns the repo: acceptance must not run as
    # the ambient login of whichever process completes the card (#122689).
    assignee = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()["assignee"]
    return snapshot, collect_acceptance(contract, published_pr, assignee=assignee)


def prepare_test_gate(conn, task_id, expected_run_id):
    """``request_review`` gate: only ``test:`` contracts apply (PR contracts
    gate the final ``done``). None = no gate; False = stale ownership."""
    snapshot = _snapshot(conn, task_id)
    if snapshot is None or not is_test_contract(snapshot[2]):
        return None
    if snapshot[1] not in _GATED_STATUSES or (expected_run_id is not None and snapshot[0] != expected_run_id):
        return False
    return _run_test_gate(conn, task_id, snapshot)


def record_acceptance(conn, task_id, acceptance):
    """Called under the terminal transition's write_txn, before its UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    test_gate = receipt.get("kind") == "test_gate"
    _append_event(conn, task_id, "test_gate" if test_gate else "pr_acceptance", receipt, run_id=snapshot[0])
    if test_gate and snapshot[0] is not None:
        # The open run carries the receipt even when the transition is refused.
        row = conn.execute("SELECT metadata FROM task_runs WHERE id=?", (snapshot[0],)).fetchone()
        try:
            current = json.loads(row["metadata"]) if row and row["metadata"] else {}
        except (TypeError, ValueError):
            current = {}
        if not isinstance(current, dict):
            current = {}
        conn.execute("UPDATE task_runs SET metadata=? WHERE id=?",
                     (json.dumps({**current, "tests": receipt}, ensure_ascii=False), snapshot[0]))
    if not receipt["ok"]:
        if test_gate:
            detail = refusal_detail(receipt)
        else:
            detail = f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]


def with_test_receipt(metadata, acceptance):
    """Handoff metadata with the passing test receipt under ``tests``."""
    if not acceptance or acceptance[1].get("kind") != "test_gate":
        return metadata
    return {**(metadata if isinstance(metadata, dict) else {}), "tests": acceptance[1]}
