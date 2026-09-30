"""Hindsight plugin constants and pure config normalizers (no I/O, no origin imports)."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Any, List

# Log under the plugin package's own logger name (loader-path independent).
logger = logging.getLogger(__name__.rpartition(".")[0])

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
# Keep in sync with tools/lazy_deps.py ("memory.hindsight") and plugin.yaml.
_MIN_CLIENT_VERSION = "0.6.1"
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
# ``metadata.source`` on retained memories is OPT-IN (AGENTS.md forbids
# on-by-default attribution tags): ``retain_source`` / HINDSIGHT_RETAIN_SOURCE.
_DEFAULT_RETAIN_SOURCE = ""
# Hindsight brand mark (eye ringed by graph nodes) for the recall/retain indicators.
_HINDSIGHT_GLYPH = "👁️"
# Hindsight 0.5.0 added ``update_mode='append'``; older APIs would silently
# overwrite prior turns under a stable document_id, so they keep the per-process id.
# Mirrors hindsight-integrations/openclaw — Hindsight 0.5.0 added `update_mode='append'` semantics on retain
# (vectorize-io/hindsight#932).
_MIN_VERSION_FOR_UPDATE_MODE_APPEND = "0.5.0"
_VALID_BUDGETS = {"low", "mid", "high"}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-3.6-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M2.7",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}
# The embedded daemon speaks OpenAI wire format for these providers.
_OPENAI_WIRE_PROVIDERS = {"openai_compatible", "openrouter"}
_OBSERVATION_SCOPE_KEYWORDS = {"per_tag", "combined", "all_combinations"}
# ``agent_context`` values the agent hands initialize() (agent/memory_provider.py). Auto-retain runs
# for these by default; "subagent"/"flush" never reach this provider today (delegated children run
# with skip_memory=True), so the default only has to name the contexts that do.
_KNOWN_AGENT_CONTEXTS = {"primary", "cron", "subagent", "flush"}
_DEFAULT_RETAIN_CONTEXTS = ("primary", "cron")
# Assistant text the runtime substitutes for a failed stream; such a turn carries no knowledge.
_STREAM_ERROR_PREFIX = "[stream error"
_FALSE_STRINGS = {"0", "false", "no", "off"}
_TRUE_STRINGS = {"1", "true", "yes", "on"}


def _parse_bool_setting(value: Any, default: bool) -> bool:
    """Parse a boolean config/env value ("0"/"false"/"no"/"off" and friends); unknown -> *default*."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _FALSE_STRINGS:
        return False
    if text in _TRUE_STRINGS:
        return True
    logger.warning("Invalid boolean Hindsight setting %r; using default %s", value, default)
    return default


def _normalize_retain_contexts(value: Any) -> tuple[str, ...]:
    """``retain_contexts`` (list or comma-separated) -> tuple of agent contexts that auto-retain.
    Unset/blank keeps the default; an explicit empty list disables auto-retain in every context."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return _DEFAULT_RETAIN_CONTEXTS
    contexts = tuple(c.lower() for c in _normalize_retain_tags(value))
    if unknown := [c for c in contexts if c not in _KNOWN_AGENT_CONTEXTS]:
        logger.warning("Unknown Hindsight retain_contexts %s (known: %s); those never match",
                       unknown, sorted(_KNOWN_AGENT_CONTEXTS))
    return contexts


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


def _parse_score_floor(value: Any) -> float | None:
    """Parse an optional 0-1 score floor; unset/blank/invalid/out-of-range -> None (no floor)."""
    if value is None or (isinstance(value, str) and not value.strip()) or isinstance(value, bool):
        return None
    try:
        floor = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid Hindsight score floor %r; applying no floor", value)
        return None
    if not 0.0 <= floor <= 1.0:
        logger.warning("Hindsight score floor %r outside 0-1; applying no floor", value)
        return None
    return floor


def _daemon_llm_provider(provider: str) -> str:
    return "openai" if provider in _OPENAI_WIRE_PROVIDERS else provider


def _normalize_retain_tags(value: Any) -> List[str]:
    """Normalize tag config/tool values to a deduplicated list of strings."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            with contextlib.suppress(Exception):
                parsed = json.loads(text)
        raw_items = parsed if isinstance(parsed, list) else text.split(",")
    normalized: list[str] = []
    for item in raw_items:
        tag = str(item).strip()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


# Kanban card text used as the worker's recall query (Retrieval·W2): the dispatcher's
# first message is the topic-free "work kanban task <id>". Default body share is 200 chars:
# measured on 5 live cards, the normalized reranker score of the best hit fell to <=0.04 with
# 600 body chars (spec, card bodies open with boilerplate) vs up to 0.51 at 200, so a
# recall_min_reranker floor would abstain on nearly every card at 600.
_KANBAN_QUERY_BODY_CHARS = 200


def _kanban_recall_query(title: Any, body: Any, body_chars: int = _KANBAN_QUERY_BODY_CHARS) -> str:
    """Card title + the first *body_chars* of its body; ``""`` when the card has neither."""
    title_text = str(title or "").strip()
    body_text = str(body or "").strip()[:max(0, body_chars)].strip()
    return "\n\n".join(part for part in (title_text, body_text) if part)


def _kanban_recall_tags(templates: Any, tenant: str) -> List[str]:
    """Expand ``recall_kanban_tags`` templates for one worker. ``{tenant}`` is the card's
    tenant (``HERMES_TENANT``); a template naming ``{tenant}`` is dropped when the card has
    none, so an untenanted card never filters on a literal ``project:``."""
    tenant = str(tenant or "").strip()
    tags: list[str] = []
    for template in _normalize_retain_tags(templates):
        if "{tenant}" in template:
            if not tenant:
                continue
            template = template.replace("{tenant}", tenant)
        if template not in tags:
            tags.append(template)
    return tags


def _normalize_observation_scopes(value: Any) -> Any:
    """Normalize observation_scopes to a keyword string, ``list[list[str]]`` (one inner
    list per consolidation pass), or ``None`` (Hindsight's ``combined`` default).
    Accepts a keyword, a JSON-encoded list, a flat tag list (one scope) or a list of
    tag-lists; anything unrecognized -> ``None`` so we never send an invalid payload."""
    if isinstance(value, str):
        text = value.strip()
        if text in _OBSERVATION_SCOPE_KEYWORDS:
            return text
        if text.startswith("["):
            try:
                return _normalize_observation_scopes(json.loads(text))
            except Exception:
                return None
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if all(isinstance(entry, str) for entry in value):  # flat tag list -> one scope
        value = [value]
    scopes = [
        [str(tag).strip() for tag in entry if str(tag).strip()] if isinstance(entry, (list, tuple))
        else [entry.strip()] if isinstance(entry, str) and entry.strip() else []
        for entry in value
    ]
    return [s for s in scopes if s] or None


def _sanitize_bank_segment(value: str) -> str:
    """URL/filesystem-safe bank_id placeholder: runs outside ``[A-Za-z0-9_-]`` (per
    ``str.isalnum``) become one dash; leading/trailing ``-``/``_`` are stripped."""
    # \w == str.isalnum() + "_" for str patterns, so this matches the per-char rule.
    return re.sub(r"[^\w-]+", "-", str(value)).strip("-_") if value else ""


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Render a bank_id template ({profile}, {workspace}, {platform}, {user}, {session}),
    sanitizing each placeholder; the ``-``/``_`` runs empty placeholders leave are
    collapsed (``hermes-{user}`` -> ``hermes``). Empty/invalid template -> *fallback*."""
    if not template:
        return fallback
    try:
        rendered = template.format(**{k: _sanitize_bank_segment(v) for k, v in placeholders.items()})
    except (KeyError, IndexError) as exc:
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r",
                       template, exc, fallback)
        return fallback
    return re.sub(r"([-_])\1+", r"\1", rendered).strip("-_") or fallback
