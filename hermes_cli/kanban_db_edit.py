"""Field stages of :func:`hermes_cli.kanban_db.edit_task`: the tenant move and the result backfill."""
from __future__ import annotations

import json
import sqlite3
from typing import Optional


def normalize_tenant(tenant: str) -> str:
    """A retenant needs a real slug: clearing a card back to NULL is not an edit."""
    tenant = tenant.strip()
    if not tenant:
        raise ValueError("tenant must be non-empty")
    return tenant


def stage_tenant_edit(conn: sqlite3.Connection, task_id: str, status: str, tenant: str):
    """``kanban edit --tenant``: ``(fields, event_payload)``, None when unchanged, False on a
    running card (its worker's HERMES_TENANT, sandbox and route were fixed at spawn).

    The move never loosens policy: a card with no explicit data class whose old tenant
    resolved it to a restricted class keeps that class as an explicit pin. A
    ``ProviderDenied`` from an unreadable policy propagates (fail closed)."""
    from agent.provider_policy import is_restricted, resolve_data_class

    if status == "running":
        return False
    row = conn.execute("SELECT tenant, data_class FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row["tenant"] == tenant:
        return None
    fields: tuple = (("tenant", tenant),)
    payload: dict = {"tenant": {"from": row["tenant"], "to": tenant}}
    if row["data_class"] is None:
        old = resolve_data_class({"id": task_id, "data_class": None, "tenant": row["tenant"]})
        new = resolve_data_class({"id": task_id, "data_class": None, "tenant": tenant})
        if old != new and is_restricted(old):
            fields += (("data_class", old),)
            payload["data_class_pinned"] = old
    return fields, payload


def backfill_result(
    conn: sqlite3.Connection, task_id: str, result: str,
    summary: Optional[str], metadata: Optional[dict],
) -> None:
    """Write a done task's backfilled result onto its latest completed run (one is
    synthesized when none exists) and record the ``edited`` event."""
    from hermes_cli.kanban_db import _append_event, _first_line, _synthesize_ended_run

    handoff_summary = summary if summary is not None else result
    run = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND outcome = 'completed' "
        "ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if run is None:
        run_id = _synthesize_ended_run(
            conn, task_id, outcome="completed", summary=handoff_summary, metadata=metadata,
        )
    else:
        run_id = int(run["id"])
        conn.execute("UPDATE task_runs SET summary = ? WHERE id = ?", (handoff_summary, run_id))
        if metadata is not None:
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), run_id),
            )
    _append_event(
        conn, task_id, "edited",
        {
            "fields": ["result", "summary"] + (["metadata"] if metadata is not None else []),
            "result_len": len(result) if result else 0,
            "summary": _first_line(handoff_summary, 400) or None,
        },
        run_id=run_id,
    )
