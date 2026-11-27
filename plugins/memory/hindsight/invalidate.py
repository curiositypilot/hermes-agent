"""hindsight_invalidate and ``hindsight_retain(supersedes=...)``: retire a recalled fact the agent
knows is false, in the turn it learns so (t_337039ce).

Before this the agent could only retain: it wrote "this supersedes ..." into the note and reported
the stale fact as replaced while it stayed live. Recall rows now carry an 8-char id (see
``settings._format_tool_recall``), remembered here per provider so the model can pass it back.

Ordering is the point of ``supersedes``: a replacement retained while its stale twin is live can
be merged INTO that twin by the server's cross-document dedup, so the retain runs only when every
named id is invalidated (an already-invalidated fact counts, so a retry is safe).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

# Server parity: overrides orchestrator ``_XDOC_CORR_INVALIDATE_LABELS``. Any other status label
# (decision, lesson, ...) is a choice the user made; it changes through ``memory-system retire``.
_INVALIDATABLE_STATUS = frozenset({"finding", "hypothesis"})
_REMEMBERED_IDS_MAX = 200
_REASON_MAX = 300
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
_REVERSIBLE = "reversible: PATCH state=valid"

_IDS_DESCRIPTION = "The 8-char ids shown by hindsight_recall ({id8} before the text), or full UUIDs."

INVALIDATE_SCHEMA = {
    "name": "hindsight_invalidate",
    "description": (
        "Retire recalled memories that are false: use it when the user corrects a fact or a check "
        "shows a recalled fact is wrong, in the same turn, passing every recalled id that states "
        "the stale claim, BEFORE retaining the replacement (or use hindsight_retain's supersedes). "
        "Never claim a memory was replaced unless this tool returned `invalidated` for it."
    ),
    "parameters": {"type": "object", "required": ["ids", "reason"], "properties": {
        "ids": {"type": "array", "items": {"type": "string"}, "description": _IDS_DESCRIPTION},
        "reason": {"type": "string", "description": "What is wrong with the fact, e.g. the correction."},
    }},
}

SUPERSEDES_PARAM = {"type": "array", "items": {"type": "string"}, "description": (
    "Ids of recalled memories this content corrects (every recalled id stating the stale claim). "
    "They are invalidated first; the content is stored only if all of them were invalidated. "
    + _IDS_DESCRIPTION
)}


def new_id_store() -> OrderedDict:
    return OrderedDict()


def remember_ids(store: OrderedDict, results: list) -> None:
    """Record ``id8 -> (full id, fact type)`` for the recall rows the tool just served."""
    for r in results:
        full = str(getattr(r, "id", None) or "")
        if not full:
            continue
        store[full[:8]] = (full, getattr(r, "type", None))
        store.move_to_end(full[:8])
    while len(store) > _REMEMBERED_IDS_MAX:
        store.popitem(last=False)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    return [str(v).strip() for v in value or [] if str(v).strip()]


def _field(fact: Any, name: str) -> Any:
    return fact.get(name) if isinstance(fact, dict) else getattr(fact, name, None)


def _bank_errors() -> tuple:
    """What a bank call raises for a bad id or an unreachable server: the SDK's ApiException
    (404, 400, ...), aiohttp transport errors, timeouts. Anything else is a bug and propagates."""
    import aiohttp
    from hindsight_client_api.exceptions import ApiException

    return ApiException, aiohttp.ClientError, OSError, TimeoutError


def _error(exc: Exception) -> str:
    """One line: the SDK's ApiException text carries headers and body on later lines."""
    return f"{type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0][:200]}"


def _blocking_status(entities: Any) -> str:
    """The first ``status:<x>`` label on the fact that makes it non-invalidatable, else ''."""
    for entity in entities or []:
        name = entity if isinstance(entity, str) else str(_field(entity, "name") or "")
        key, _, label = name.partition(":")
        if key.strip().lower() == "status" and label.strip().lower() not in _INVALIDATABLE_STATUS:
            return label.strip().lower()
    return ""


def _invalidate_one(provider, ref: str, reason: str) -> tuple[str, str]:
    """``(outcome, line)`` for one id; outcome is invalidated / refused / failed."""
    known = provider._recall_ids.get(ref)
    full = known[0] if known else (ref if _UUID_RE.match(ref) else "")
    short = ref[:8]
    if not full:
        return "refused", f"refused {short}: unknown id; run hindsight_recall first"
    bank, errors = provider._bank_id, _bank_errors()
    try:
        fact = provider._run_hindsight_operation(lambda c: c.memory.get_memory(bank, full))
    except errors as exc:  # reported per id: one bad id must not hide the others' outcomes
        return "failed", f"failed {short}: {_error(exc)}"
    text = str(_field(fact, "text") or "")[:80]
    if _field(fact, "state") == "invalidated":
        return "invalidated", f"invalidated {short}: {text} (already)"
    if (_field(fact, "type") or (known[1] if known else None)) == "observation":
        return "refused", f"refused {short}: observation; invalidate its source facts instead"
    if status := _blocking_status(_field(fact, "entities")):
        return "refused", (f"refused {short}: status:{status} fact; change it with "
                           f"`memory-system retire {full} --new <replacement full UUID> --why <reason>` "
                           "in the terminal (it takes full UUIDs, not 8-char ids)")
    request = {"state": "invalidated",
               "reason": f"agent-correction: {reason[:_REASON_MAX]} (session {provider._session_id})"}
    try:
        provider._run_hindsight_operation(lambda c: c.memory.update_memory(bank, full, request))
    except errors as exc:
        return "failed", f"failed {short}: {_error(exc)}"
    return "invalidated", f"invalidated {short}: {text}"


def invalidate_ids(provider, ids: Any, reason: str) -> list[tuple[str, str]]:
    refs = _as_list(ids)
    if not refs:
        raise ValueError("no ids given; pass the 8-char ids from hindsight_recall")
    return [_invalidate_one(provider, ref, reason) for ref in refs]


def tool_invalidate(provider, args: dict) -> str:
    """``hindsight_invalidate`` handler (wired in the provider's ``_TOOL_HANDLERS``)."""
    rows = invalidate_ids(provider, args.get("ids"), str(args["reason"]))
    return "\n".join([*(line for _, line in rows), _REVERSIBLE])


def supersede(provider, supersedes: Any, content: str) -> tuple[list[str], str | None]:
    """Invalidate *supersedes* ahead of a retain. Returns ``(notes, blocked)``: *blocked* is the
    tool result to return instead of retaining when any id was not invalidated."""
    if not _as_list(supersedes):
        return [], None
    rows = invalidate_ids(provider, supersedes, content)
    lines = [line for _, line in rows]
    if all(outcome == "invalidated" for outcome, _ in rows):
        return [f"Superseded: {'; '.join(lines)} ({_REVERSIBLE})."], None
    return [], "\n".join([
        "Memory NOT stored: not every supersedes id was invalidated, and a replacement stored "
        "while the stale fact is live can be merged into it.", *lines,
        "Retry once the failed ids invalidate (already-invalidated ones are fine to resend). A "
        "refused status: fact changes through the `memory-system retire` command shown, an "
        "observation through its source facts. Report nothing as replaced until then.", _REVERSIBLE])
