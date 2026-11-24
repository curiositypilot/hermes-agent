"""``project:`` tag gate for the hindsight_retain tool (t_38883233).

The tool used to store any ``project:<slug>`` tag, so facts landed under slugs no project owns
(``project:honda-wheels`` for ``car``) and the bank's per-project sweeps never saw them. Recall
only warns about such a tag, after the fact exists. Retain now resolves the slug first and refuses
one that is no bank label, failing open wherever the label set may be wrong rather than the tag.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from .settings import _apply_project_aliases

logger = logging.getLogger(__name__.rpartition(".")[0])

_REGISTRY_CACHE: dict[str, tuple[float, dict[str, str]]] = {}


def registry_names(path: str) -> dict[str, str]:
    """``project_registry`` JSON (``{"projects": [{"slug", "aliases"}]}``) -> {slug or alias: slug}.

    Read on use and re-read when the file's mtime changes, so the registry stays the one owner of
    project aliases (no copy in config.json to drift). Unset or unreadable -> {} (logged)."""
    if not path:
        return {}
    file = Path(path).expanduser()
    try:
        mtime = file.stat().st_mtime
        if (hit := _REGISTRY_CACHE.get(str(file))) and hit[0] == mtime:
            return hit[1]
        names: dict[str, str] = {}
        for project in json.loads(file.read_text(encoding="utf-8-sig")).get("projects") or []:
            slug = str(project.get("slug") or "").strip()
            for name in (slug, *(project.get("aliases") or [])) if slug else ():
                names.setdefault(str(name).strip(), slug)
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        logger.warning("Hindsight project_registry %s unreadable; aliases unresolved: %s: %s",
                       file, type(exc).__name__, exc)
        return {}
    _REGISTRY_CACHE[str(file)] = (mtime, names)
    return names


def gate_retain_tags(tags: list[str], *, aliases: dict, registry_path: str,
                     labels: Callable[[bool], frozenset | None]) -> tuple[list[str], list[str], str]:
    """(tags to store, notes for the model, refusal). A non-empty refusal means store nothing.

    ``project:`` tags are rewritten through ``recall_project_aliases``, then the registry's aliases.
    A slug that is no bank label is refused, except:
    - ``labels(False)`` is None (config fetch failed, no ``project`` label): stored unchecked, as
      before; ``_project_slugs`` already logs the failure once per TTL window.
    - the cached label set predates a label added within its TTL: a miss re-checks on
      ``labels(True)`` (a fresh fetch) before refusing.
    - a registry slug the bank profile lacks (profile not re-applied yet): stored, with a note.
    The refusal names no "closest" label: string similarity suggests unrelated projects
    (honda-wheels -> land-geres), and a retry with one would pass this gate under the wrong slug."""
    registry = registry_names(registry_path)
    rewritten = _apply_project_aliases(_apply_project_aliases(tags, aliases), registry)
    missing = [t[len("project:"):] for t in rewritten if t.startswith("project:")]
    for refresh in (False, True):
        if not missing:
            break
        known = labels(refresh)
        missing = [] if known is None else [s for s in missing if s not in known]
    lagging = [s for s in missing if s in registry.values()]
    refused = [s for s in missing if s not in lagging]
    if refused:
        return rewritten, [], "Not stored: " + "; ".join(
            f"project:{s} is not a project label in this bank and no alias maps it to one"
            for s in refused) + ". Retry with the project's registered slug, or without the project tag."
    notes = [f"(tags used: {json.dumps(rewritten)})"] if rewritten != tags else []
    for slug in lagging:
        logger.warning("hindsight_retain: project:%s is a registry slug but no bank label", slug)
        notes.append(f"(note: project:{slug} is a registry slug the bank's project labels lack; "
                     "re-apply the bank profile)")
    return rewritten, notes, ""
