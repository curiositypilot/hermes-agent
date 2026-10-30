"""Latest-status lines over a recall block, read from the memory-system entity-state index.

Setting ``recall_state_lines`` (intake t_5958cf5b slice B; spec Change 5):

    off     default; output byte-identical to a plugin without this module, nothing read or written
    shadow  compute the lines, serve the unchanged block, append one JSONL row per block
    on      serve the lines above the hits and prefix each covered hit; the JSONL row is still written,
            so the shadow-week numbers can be re-measured after the flip

The index is written by memory-system (slice A, ``memory_system/entity_state/index.py``, which holds
the shared contract) at ``<default hermes root>/data/memory-system/entity-state/index.sqlite``:

    meta(key, value)                  schema_version=1, built_at (ISO UTC)
    status(unit_id, key, kind, entity, word, created, trust, source_kind, text)
    marks(unit_id, status_id, key, kind, sim, status_created)   (status_id, key) joins status(unit_id, key)

Read rule (must match ``memory-system state annotate --ids``): aggregate the marks rows of the served
hit ids per status (covers = hit positions), rank by (-len(covers), lifecycle first, min(covers),
-status_created, status_id), keep MAX_LINES.  A covered hit takes the prefix of the first kept line
that covers it, with that hit's own mark-row key.

Fail-open: a missing or locked file, schema_version != 1, or built_at older than MAX_AGE_S gives no
lines (debug log) and no JSONL row.  A block with no hit ids gives no row either: the audit's mean
``chars_added`` is over the blocks that had an index to read.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")
HEADER = "Latest status per entity (newer than the lines marked below; prefer it when they conflict):"
SCHEMA_VERSION = "1"
MAX_AGE_S = 6 * 3600
MAX_LINES = 3
BUDGET_MS = 20.0
INDEX_RELATIVE = ("data", "memory-system", "entity-state", "index.sqlite")
LOG_RELATIVE = ("data", "memory-system", "state_lines.jsonl")
_SQL = ("select m.unit_id, m.status_id, m.key, m.kind, m.status_created, s.entity, s.word, s.text"
        " from marks m join status s on s.unit_id = m.status_id and s.key = m.key"
        " where m.unit_id in ({}) order by m.unit_id, m.status_id")


def index_path() -> str:
    return os.path.join(str(get_default_hermes_root()), *INDEX_RELATIVE)


def log_path() -> str:
    return os.path.join(str(get_default_hermes_root()), *LOG_RELATIVE)


def parse_mode(value: Any) -> str:
    """off|shadow|on; YAML 1.1 reads bare ``off``/``on`` as booleans, so those map back. Unknown -> off."""
    if isinstance(value, bool):
        return "on" if value else "off"
    mode = str(value or "").strip().lower()
    return mode if mode in MODES else "off"


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


def line_text(created: float, entity: str, word: str, key: str, text: str) -> str:
    lead = f"{entity}: {word}" if entity and word else key
    return f"- [{_date(created)}] {lead} | {text}"


def prefix_text(key: str, created: float) -> str:
    return f"[older than {key} status {_date(created)}] "


def _rank(a: dict) -> tuple:
    return (-len(a["covers"]), a["kind"] != "lifecycle", min(a["covers"]), -a["created"], a["status_id"])


@dataclass
class Annotation:
    built_at: str
    lines: list[dict]  # {status_id, key, kind, created, covers, line}, ranked
    prefixes: dict[int, str]  # hit position -> "[older than ...] "

    @property
    def block(self) -> str:
        return "\n".join([HEADER, *(a["line"] for a in self.lines)]) if self.lines else ""


def _fresh_built_at(conn: sqlite3.Connection, now: float) -> str | None:
    meta = dict(conn.execute("select key, value from meta where key in ('schema_version', 'built_at')"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        logger.debug("state lines skipped: index schema_version %r", meta.get("schema_version"))
        return None
    built_at = meta.get("built_at") or ""
    try:
        age = now - datetime.fromisoformat(built_at).timestamp()
    except ValueError:
        age = None
    if age is None or age > MAX_AGE_S:
        logger.debug("state lines skipped: index built_at %r is stale or unreadable", built_at)
        return None
    return built_at


def annotate(ids: list, *, path: str | None = None, now: float | None = None,
             max_lines: int = MAX_LINES) -> Annotation | None:
    """The read rule for one block (hit ids in served order); None when the index is unusable."""
    target = Path(path or index_path()).absolute()
    try:
        conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, timeout=0.05)
    except sqlite3.Error as exc:
        logger.debug("state lines skipped: cannot open %s: %s", target, exc)
        return None
    try:
        built_at = _fresh_built_at(conn, time.time() if now is None else now)
        if built_at is None:
            return None
        wanted = sorted({i for i in ids if i})
        rows: list = []
        for at in range(0, len(wanted), 500):
            chunk = wanted[at:at + 500]
            rows += conn.execute(_SQL.format(",".join("?" * len(chunk))), chunk).fetchall()
    except sqlite3.Error as exc:  # locked, missing tables, corrupt
        logger.debug("state lines skipped: index read failed: %s", exc)
        return None
    finally:
        conn.close()
    by_unit: dict[str, list] = {}
    for r in rows:
        by_unit.setdefault(r[0], []).append(r)
    agg: dict[str, dict] = {}
    for pos, uid in enumerate(ids):
        for _, sid, key, kind, created, entity, word, text in by_unit.get(uid, ()):
            a = agg.setdefault(sid, {"status_id": sid, "key": key, "kind": kind, "created": created,
                                     "line": line_text(created, entity, word, key, text), "covers": [], "keys": {}})
            a["covers"].append(pos)
            a["keys"][pos] = key
    lines = sorted(agg.values(), key=_rank)[:max_lines]
    prefixes: dict[int, str] = {}
    for a in lines:
        for p in a["covers"]:
            prefixes.setdefault(p, prefix_text(a["keys"][p], a["created"]))
    return Annotation(built_at, [{k: v for k, v in a.items() if k != "keys"} for a in lines], prefixes)


def _hit_ids(results: Iterable[Any]) -> list[str]:
    return [i if isinstance(i := getattr(r, "id", None), str) else "" for r in results]


def log_block(where: str, query: str, ids: list[str], ann: Annotation, chars_added: int,
              *, path: str | None = None) -> None:
    """Append one JSONL row (single O_APPEND write); every failure is swallowed, as in recall_log."""
    try:
        row = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "path": where,
            "profile": profile_name_for_home(get_hermes_home()) or "",
            "query_sha": hashlib.sha256(query.encode("utf-8", "replace")).hexdigest()[:16],
            "hit_ids": ids,
            "lines": [{k: a[k] for k in ("status_id", "key", "kind", "covers")} for a in ann.lines],
            "chars_added": chars_added,
            "index_built_at": ann.built_at,
        }
        target = path or log_path()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        try:
            fd = os.open(target, flags, 0o600)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            fd = os.open(target, flags, 0o600)
        try:
            os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)
    except Exception as exc:  # fail-open: logging must never break a recall
        logger.debug("state lines log skipped: %s", exc, exc_info=True)


def decorate(mode: str, where: str, query: str, results: list, render: Callable[[dict], str]) -> str:
    """The served text for one recall block. *results* are the served hits in order; *render(prefixes)*
    renders them with ``{position: prefix}`` applied. ``off`` calls render once and nothing else."""
    plain = render({})
    if mode == "off" or not plain:
        return plain
    try:
        started = time.perf_counter()
        ids = _hit_ids(results)
        ann = annotate(ids) if any(ids) else None
        if ann is None:
            return plain
        marked = f"{ann.block}\n\n{render(ann.prefixes)}" if ann.lines else plain
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms > BUDGET_MS:
            logger.debug("state lines took %.1f ms (budget %.0f ms)", elapsed_ms, BUDGET_MS)
        log_block(where, query, ids, ann, len(marked) - len(plain))
        return marked if mode == "on" else plain
    except Exception as exc:  # an annotation bug must never cost the recall itself
        logger.debug("state lines skipped: %s", exc, exc_info=True)
        return plain
