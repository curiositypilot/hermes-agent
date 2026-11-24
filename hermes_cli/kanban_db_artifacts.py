"""Completion artifact staging: every file a handoff declares in
``metadata["artifacts"]`` is copied into the task's durable attachments dir
and recorded as an attachment row, whatever the workspace kind and wherever
the file lives. Declared sources are often ephemeral (the global
``cache/scratch`` prune, a removed worktree, the scratch workspace cleanup),
so a path that cannot be copied refuses the transition instead of riding the
handoff as a dangling string.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers and the size
cap are reached late-bound via ``_kb`` so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Iterable, Optional


def _stage_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict, now: int, *,
    uploaded_by: str = "kanban_complete",
) -> list[Path]:
    """Copy declared artifacts to the attachments dir and record each copy as
    an attachment row; returns the copies so the caller can discard them if
    its transaction rolls back."""
    _persist_completion_artifacts(conn, task_id, metadata)
    staged = [Path(stored_path) for stored_path in metadata.pop("_staged_artifacts", [])]
    for path in staged:
        _kb._insert_completion_attachment(
            conn, task_id, filename=path.name, stored_path=str(path),
            size=path.stat().st_size, created_at=now, uploaded_by=uploaded_by,
        )
    return staged


def _attachment_board(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Board of a managed scratch workspace; ``None`` (active board) otherwise."""
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return None
    is_managed, board = _kb._managed_scratch_path_info(workspace)
    return board if is_managed else None


def _artifact_problem(src: Path, artifact: str) -> Optional[str]:
    """Why *src* cannot be preserved, or ``None`` when it is a copyable file."""
    cap = _kb.KANBAN_ATTACHMENT_MAX_BYTES
    if not src.exists():
        return f"declared artifact {artifact}: not found"
    if src.is_dir():
        return f"declared artifact {artifact}: is a directory (list files)"
    if not src.is_file():
        return f"declared artifact {artifact}: is not a regular file"
    if src.stat().st_size > cap:
        return f"declared artifact {artifact}: exceeds {cap} bytes"
    return None


def _persist_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict,
) -> None:
    """Copy every declared artifact into the task's attachments dir.

    An entry already inside that dir (a re-declared, already-staged path) is
    kept as is. Any other entry must be a regular file within the attachment
    cap; it is copied and its metadata entry rewritten to the copy. A missing,
    directory, special or oversize path raises
    :class:`ArtifactPreservationError` after discarding the copies staged so
    far. The copies to record ride ``metadata["_staged_artifacts"]``.
    """
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return
    attachment_dir = _kb.task_attachments_dir(task_id, board=_attachment_board(conn, task_id))
    attachment_root = attachment_dir.resolve()
    persisted: list[str] = []
    copies: list[Path] = []
    used_destinations: set[Path] = set()

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        src = Path(artifact).expanduser()
        resolved_src: Optional[Path] = None
        try:
            resolved_src = src.resolve()
            if resolved_src.is_relative_to(attachment_root):
                persisted.append(artifact)
                continue
            problem = _artifact_problem(src, artifact)
        except OSError as exc:
            problem = f"declared artifact {artifact}: {exc}"
        if problem or resolved_src is None:
            _discard_staged_copies(copies, attachment_dir)
            raise _kb.ArtifactPreservationError(problem)
        dest = _copy_artifact(resolved_src, artifact, attachment_dir, used_destinations, copies)
        persisted.append(str(dest.resolve()))

    if copies:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = [str(path.resolve()) for path in copies]


def _copy_artifact(
    src: Path, artifact: str, attachment_dir: Path, used: set[Path], copies: list[Path],
) -> Path:
    """Copy *src* into *attachment_dir*; on failure discard every copy staged
    so far and raise :class:`ArtifactPreservationError`."""
    dest: Optional[Path] = None
    try:
        attachment_dir.mkdir(parents=True, exist_ok=True)
        dest = _unique_attachment_path(attachment_dir, src.name, used)
        _copy_capped(src, dest, artifact)
    except (OSError, _kb.ArtifactPreservationError) as exc:
        if dest is not None:
            with contextlib.suppress(OSError):
                dest.unlink(missing_ok=True)
        _discard_staged_copies(copies, attachment_dir)
        if isinstance(exc, _kb.ArtifactPreservationError):
            raise
        raise _kb.ArtifactPreservationError(
            f"could not preserve declared artifact {artifact}: {exc}"
        ) from exc
    used.add(dest)
    copies.append(dest)
    return dest


def _discard_staged_copies(copies: Iterable[Path], attachment_dir: Path) -> None:
    """Remove staged attachment copies whose DB rows never committed; a leaked
    copy would make the retry stage ``name_1.ext`` next to an orphan."""
    for copied in copies:
        with contextlib.suppress(OSError):
            Path(copied).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        attachment_dir.rmdir()


def _copy_capped(src: Path, dest: Path, artifact: str) -> None:
    """Chunked copy that aborts if the file grows past the attachment cap mid-copy."""
    cap = _kb.KANBAN_ATTACHMENT_MAX_BYTES
    with src.open("rb") as source_file, dest.open("xb") as destination_file:
        copied = 0
        while chunk := source_file.read(1024 * 1024):
            copied += len(chunk)
            if copied > cap:
                raise _kb.ArtifactPreservationError(
                    f"declared artifact {artifact}: grew beyond {cap} bytes while copying"
                )
            destination_file.write(chunk)


def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return a non-conflicting path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    stem, suffix = Path(safe_name).stem or "artifact", Path(safe_name).suffix
    candidate = directory / safe_name
    idx = 1
    while candidate in used or candidate.exists():
        candidate = directory / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


from hermes_cli import kanban_db as _kb  # late: kanban_db imports this module mid-file
