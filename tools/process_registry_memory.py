"""Memory bounds for the ``systemd-run --user --scope`` children of ``tools.process_registry``.

The facade owns the limits (``_MIN_WORKER_MEMORY_MAX_BYTES``, the 4 GiB default cap) and the
default ``_worker_memory_max_bytes()``; this module reads the host facts both paths clamp to
and clamps an explicit per-caller request (``kanban.worker_memory_max_mb``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def gateway_cgroup_memory_max(floor: int) -> Optional[int]:
    """This process's cgroup-v2 ``memory.max`` when it is a finite value of at least *floor*."""
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw_limit = (
                    Path("/sys/fs/cgroup") / relative / "memory.max"
                ).read_text(encoding="utf-8-sig").strip()
                if raw_limit.isdigit() and int(raw_limit) >= floor:
                    return int(raw_limit)
                return None
    except (OSError, ValueError):
        return None
    return None


def physical_memory_bytes() -> Optional[int]:
    """Physical RAM in bytes, or None where ``sysconf`` cannot report it."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, TypeError):
        return None


def explicit_worker_memory_max_bytes(requested: int) -> int:
    """Clamp an explicit MemoryMax request to [64 MiB, min(gateway cgroup, physical RAM // 2)].

    Unlike the default bound, the 4 GiB cap does not apply: the caller asked for more on purpose.
    The floor wins over an upper bound below it (a tiny cgroup or host), as the default path does.
    With no readable bound the request cannot be checked, so the default's fallback applies.
    """
    from tools.process_registry import (
        _DEFAULT_WORKER_MEMORY_MAX_BYTES,
        _MIN_WORKER_MEMORY_MAX_BYTES,
    )

    bounds: List[int] = []
    cgroup_limit = gateway_cgroup_memory_max(_MIN_WORKER_MEMORY_MAX_BYTES)
    if cgroup_limit is not None:
        bounds.append(cgroup_limit)
    physical = physical_memory_bytes()
    if physical is not None:
        bounds.append(physical // 2)
    upper = min(bounds) if bounds else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return max(_MIN_WORKER_MEMORY_MAX_BYTES, min(requested, upper))
