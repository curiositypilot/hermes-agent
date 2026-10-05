"""Task-scoped operating principles: Hindsight directives fetched by tag.

One owner for the retrieval, shared by every consumer:

- ``kanban_db.build_worker_context`` appends a ``## Principles`` block to each
  worker prompt (:func:`resolve_tags` picks the tags, :func:`fetch_directives`
  fetches, the caller renders);
- ``~/.hermes/scripts/principles.py`` is a thin shim over :func:`main` (skills
  such as ``code-review`` call it from a shell).

Directives live in Hindsight (tags, priority, active flag); nothing here caches
or copies them. Stdlib only, so the shim runs under any interpreter.

Endpoint config (not secrets, read per call): ``HINDSIGHT_URL`` (default
``http://127.0.0.1:8888``) and ``HINDSIGHT_BANK`` (default ``main``).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Any, Iterable, Optional

DEFAULT_URL = "http://127.0.0.1:8888"
DEFAULT_BANK = "main"
DEFAULT_CAP_CHARS = 3000
TIMEOUT_SECONDS = 5
_FETCH_LIMIT = 200  # directives requested from the server per call

# Card title prefix -> directive tag. Used only when the card body carries no
# ``principles:`` line. A card whose title matches neither gets no block.
TITLE_PREFIX_TAGS: dict[str, str] = {
    "fork": "task:coding",
    "scripts": "task:coding",
    "plugin": "task:coding",
    "research": "task:research",
    "review": "task:code-review",
}

_BODY_LINE_RE = re.compile(r"^[ \t]*principles:[ \t]*(.+?)[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TITLE_PREFIX_RE = re.compile(r"^\s*([A-Za-z]+):")
# A directive tag is ``namespace:name`` or a bare word; placeholders such as
# ``<tag>`` or ``[<tag>]`` quoted in a card body are not tags.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class DirectivesUnavailable(RuntimeError):
    """Hindsight could not be reached or answered with something unusable.

    ``str(exc)`` is the underlying cause; ``exc.url`` is the request URL.
    """

    def __init__(self, cause: object, url: str = "") -> None:
        super().__init__(str(cause))
        self.url = url


class DirectiveList(list):  # type: ignore[type-arg]
    """``list[dict]`` of directives plus how many the char cap dropped."""

    omitted: int = 0


def resolve_tags(title: Optional[str], body: Optional[str]) -> list[str]:
    """Directive tags for a card: first ``principles: <tag> [<tag>]`` body line, else title prefix.

    Returns ``[]`` when neither yields a tag (the card then gets no principles block).
    """
    match = _BODY_LINE_RE.search(body or "")
    if match:
        tags = [t for t in re.split(r"[\s,]+", match.group(1)) if _TAG_RE.match(t)]
        if tags:
            return list(dict.fromkeys(tags))
    prefix = _TITLE_PREFIX_RE.match(title or "")
    tag = TITLE_PREFIX_TAGS.get(prefix.group(1).lower()) if prefix else None
    return [tag] if tag else []


def hindsight_target() -> tuple[str, str]:
    """``(base_url, bank)`` from the environment, defaults for the local server."""
    base = os.environ.get("HINDSIGHT_URL", DEFAULT_URL).rstrip("/")
    return base, os.environ.get("HINDSIGHT_BANK", DEFAULT_BANK)


def directives_url(tags: Iterable[str]) -> str:
    base, bank = hindsight_target()
    query = urllib.parse.urlencode(
        [("tags", t) for t in tags] + [("tags_match", "any"), ("limit", str(_FETCH_LIMIT))]
    )
    return f"{base}/v1/default/banks/{urllib.parse.quote(bank, safe='')}/directives?{query}"


def _get_json(url: str, timeout: float) -> Any:
    """GET ``url`` and decode JSON. The one network seam (tests patch it)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def render_directive(directive: dict) -> str:
    return f"- {directive['name']}: {directive['content']}"


def _apply_cap(items: list[dict], cap_chars: int) -> DirectiveList:
    """Keep directives in priority order while their rendered lines fit ``cap_chars``.

    Stops at the first one that does not fit rather than skipping to a smaller,
    lower-priority one. A first directive larger than the whole cap is truncated
    (visibly) instead of dropping everything.
    """
    kept = DirectiveList()
    used = 0
    for directive in items:
        cost = len(render_directive(directive)) + 1  # + newline
        if used + cost > cap_chars:
            if not kept:
                room = cap_chars - len(f"- {directive['name']}: ") - 2
                kept.append({**directive, "content": directive["content"][: max(room, 0)] + "…"})
            break
        kept.append(directive)
        used += cost
    kept.omitted = len(items) - len(kept)
    return kept


def fetch_directives(tags: Iterable[str], cap_chars: Optional[int] = DEFAULT_CAP_CHARS) -> list[dict]:
    """Active directives carrying any of ``tags`` (plus untagged globals), priority first.

    Ordered by priority (desc) then name. With ``cap_chars`` set, the result is cut so the
    rendered ``- name: content`` lines total at most that many chars (``None`` = no cap); the
    returned list's ``.omitted`` says how many were dropped. Empty ``tags`` -> ``[]``, no request.
    Raises :class:`DirectivesUnavailable` on any network, HTTP or payload failure (5 s timeout).
    """
    tags = list(dict.fromkeys(tags))
    if not tags:
        return DirectiveList()
    url = directives_url(tags)
    try:
        payload = _get_json(url, TIMEOUT_SECONDS)
        items = [d for d in payload.get("items", []) if d.get("is_active", True)]
        items.sort(key=lambda d: (-int(d.get("priority") or 0), d.get("name") or ""))
        items = [{**d, "name": str(d["name"]), "content": str(d["content"])} for d in items]
    except Exception as exc:  # network, HTTP, JSON or shape: report the cause, no fallback
        raise DirectivesUnavailable(exc, url) from exc
    if cap_chars is None:
        return DirectiveList(items)
    return _apply_cap(items, cap_chars)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: print the principles for one or more task tags.

    ``principles.py task:coding [task:code-review ...]`` prints a header plus every
    active directive as ``- name: content``, uncapped. Exit 1 on a server error
    (cause on stderr), exit 2 with usage when no tag is given.
    """
    tags = list(sys.argv[1:] if argv is None else argv)
    if not tags:
        print("Usage: principles.py task:coding [task:code-review ...]\n" + (main.__doc__ or "").strip(),
              file=sys.stderr)
        return 2
    try:
        items = fetch_directives(tags, cap_chars=None)
    except DirectivesUnavailable as exc:
        print(f"principles.py: {exc.url}: {exc}", file=sys.stderr)
        return 1
    print(f"# principles for {' '.join(tags)} ({len(items)} rules, bank {hindsight_target()[1]})")
    for directive in items:
        print(render_directive(directive))
    return 0


if __name__ == "__main__":
    sys.exit(main())
