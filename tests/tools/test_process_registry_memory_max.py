"""Scope MemoryMax: the default bound vs an explicit request (``kanban.worker_memory_max_mb``).

The default stays capped at 4 GiB; an explicit request may exceed it but never half of physical
RAM or the gateway cgroup, and never drops below the 64 MiB floor (intake t_058789f6).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tools.process_registry as pr

GiB = 1024 * 1024 * 1024
MiB = 1024 * 1024


@pytest.fixture
def fake_host(monkeypatch: pytest.MonkeyPatch):
    """Pin physical RAM and the gateway cgroup's memory.max; the real host never leaks in."""
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
    real_read_text = Path.read_text

    def install(*, physical: int, cgroup_max: str = "max") -> None:
        def read_text(self, *args, **kwargs):
            if str(self) == "/proc/self/cgroup":
                return "0::/user.slice/hermes-gateway.service\n"
            if str(self).startswith("/sys/fs/cgroup/") and self.name == "memory.max":
                return f"{cgroup_max}\n"
            return real_read_text(self, *args, **kwargs)

        page = 4096
        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(
            os, "sysconf", lambda name: physical // page if name == "SC_PHYS_PAGES" else page,
        )

    return install


def _memory_max(argv: list[str]) -> int:
    (value,) = [arg for arg in argv if arg.startswith("MemoryMax=")]
    return int(value.split("=", 1)[1])


def _scope_argv(memory_max_bytes):
    return pr._build_systemd_scope_argv(
        ["/bin/sh", "-c", "true"], unit_suffix="test", memory_max_bytes=memory_max_bytes,
    )


def test_explicit_memory_max_exceeds_default_cap(fake_host) -> None:
    fake_host(physical=64 * GiB)

    assert _memory_max(_scope_argv(12 * GiB)) == 12884901888


def test_explicit_memory_max_clamped_to_half_ram_and_cgroup(fake_host) -> None:
    fake_host(physical=64 * GiB)
    assert _memory_max(_scope_argv(200 * GiB)) == 32 * GiB

    fake_host(physical=64 * GiB, cgroup_max=str(8 * GiB))
    assert _memory_max(_scope_argv(200 * GiB)) == 8 * GiB


def test_explicit_memory_max_floor_wins_over_a_smaller_upper_bound(fake_host) -> None:
    """Half of a 64 MiB host is below the floor: the floor wins, as the default path does."""
    fake_host(physical=64 * MiB)
    assert _memory_max(_scope_argv(12 * GiB)) == pr._MIN_WORKER_MEMORY_MAX_BYTES

    fake_host(physical=64 * GiB)
    assert _memory_max(_scope_argv(1 * MiB)) == pr._MIN_WORKER_MEMORY_MAX_BYTES


def test_default_memory_max_unchanged_without_request(fake_host) -> None:
    fake_host(physical=64 * GiB)

    assert _memory_max(_scope_argv(None)) == pr._worker_memory_max_bytes() == 4 * GiB

    fake_host(physical=64 * GiB, cgroup_max=str(2 * GiB))
    assert _memory_max(_scope_argv(None)) == 2 * GiB
