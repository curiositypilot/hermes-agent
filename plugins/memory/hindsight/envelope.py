"""Source envelope stamped into retain item metadata (fail open, no network).

Retain profile v1.15.0+ reads these keys from each item's metadata instead of selecting a
per-kind prompt; keys and trust values match memory-system
``evals/retain_quality/configs/universal_v3.json``. Two writers fill it here:

- auto-retained chat turns: ``source_kind=chat``, ``trust=self``;
- ``hindsight_retain`` tool items, only when ``tool_retain_strategy`` is set:
  ``source_kind=agent-note``, ``trust=agent`` (a deliberate note the agent wrote).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENVELOPE_TITLE_MAX = 120


def first_line(text: Any, limit: int = ENVELOPE_TITLE_MAX) -> str:
    """First non-empty line of *text*, capped at *limit* chars ('' when none)."""
    for line in str(text or "").splitlines():
        if line := line.strip():
            return line[:limit].rstrip()
    return ""


def lookup_session_title(hermes_home: Any, session_id: str) -> str:
    """Session title from ``<hermes_home>/state.db``; '' on any miss or error (fail open).

    Plain read-only sqlite so a retain never opens a writer, runs schema init or blocks on the
    gateway's write lock."""
    if not hermes_home or not session_id:
        return ""
    try:
        db_path = Path(hermes_home) / "state.db"
        if not db_path.is_file():
            return ""
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
            row = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
        finally:
            conn.close()
        return first_line(row[0]) if row and row[0] else ""
    except (sqlite3.Error, OSError, ValueError):  # unreadable/non-sqlite db, relative home (as_uri)
        logger.debug("Hindsight envelope: session title lookup failed", exc_info=True)
        return ""


def _envelope(*, kind: str, trust: str, author: str, platform: str, title: str,
              occurred_at: str) -> dict[str, str]:
    envelope = {"source_kind": kind, "source_origin": platform,
                "source_site_or_author": author, "trust": trust}
    if title:
        envelope["source_title"] = envelope["session_topic"] = title
    if occurred_at:
        envelope["occurred_at"] = occurred_at
    return envelope


def chat_envelope(*, platform: str, title: str, occurred_at: str) -> dict[str, str]:
    """Envelope for an auto-retained chat document; no title omits the title keys."""
    platform = platform or "cli"
    return _envelope(kind="chat", trust="self", author=f"hermes {platform} session",
                     platform=platform, title=title, occurred_at=occurred_at)


def agent_note_envelope(*, platform: str, title: str, occurred_at: str) -> dict[str, str]:
    """Envelope for a ``hindsight_retain`` tool item: a note the agent chose to store."""
    platform = platform or "cli"
    return _envelope(kind="agent-note", trust="agent",
                     author=f"hermes agent via hindsight_retain ({platform})",
                     platform=platform, title=title, occurred_at=occurred_at)
