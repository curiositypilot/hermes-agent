"""Append-only log of the memory ids a Hindsight recall surfaced (fail-open, no network).

One JSON line per recalled id in ``<hermes root>/data/memory-system/recalled.jsonl``::

    {"ts": "2026-10-05T22:15:03Z", "memory_id": "<uuid>", "query_sha": "<16 hex>"}

``query_sha`` is the first 16 hex chars of the SHA-256 of the query text (the query itself is never
stored).  The nightly memory-system dream reads the last 7 days of this file to review first the facts
recall actually surfaced (``memory_system/dream/recall_log.py`` is the reader; keep both in step).

The root is the *default* Hermes root (``~/.hermes``), not the active profile home, so every profile
appends to the one file the dream reads.  One ``os.write`` per recall (O_APPEND), so concurrent
processes interleave whole recalls.  Any failure is swallowed: recall must never break on logging.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable

from hermes_constants import get_default_hermes_root

logger = logging.getLogger(__name__)

LOG_RELATIVE = ("data", "memory-system", "recalled.jsonl")


def log_path() -> str:
    return os.path.join(str(get_default_hermes_root()), *LOG_RELATIVE)


def _ids(results: Iterable[Any]) -> list[str]:
    """Result ids, de-duplicated, order kept (observations and raw facts alike; the reader matches world facts)."""
    seen: dict[str, None] = {}
    for item in results:
        value = getattr(item, "id", None)
        if isinstance(value, str) and value:
            seen.setdefault(value, None)
    return list(seen)


def log_recalled(results: Iterable[Any], query: str, *, path: str | None = None) -> int:
    """Append one line per recalled id; returns the number of lines written (0 on any failure)."""
    try:
        ids = _ids(results or ())
        if not ids:
            return 0
        sha = hashlib.sha256(query.encode("utf-8", "replace")).hexdigest()[:16]
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = "".join(f'{{"ts":"{ts}","memory_id":"{i}","query_sha":"{sha}"}}\n' for i in ids).encode()
        target = path or log_path()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        try:
            fd = os.open(target, flags, 0o600)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            fd = os.open(target, flags, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        return len(ids)
    except Exception as exc:  # fail-open: logging must never break a recall
        logger.debug("recall id log skipped: %s", exc)
        return 0
