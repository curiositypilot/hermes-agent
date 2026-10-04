"""Unit tests for memory-aware worker cap in scripts/run_tests_parallel.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worker_jobs_unlimited_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_runner()
    # Explicitly test unlimited (mem_limit=None)
    jobs = mod._default_worker_jobs(env_workers=None, cpu_count=36, mem_limit=None)
    assert jobs == 72

    jobs_low_cpu = mod._default_worker_jobs(env_workers=None, cpu_count=2, mem_limit=None)
    assert jobs_low_cpu == 4


def test_worker_jobs_4g_scope_caps_at_4(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_runner()
    # 4 GiB scope = 4 * 1024 * 1024 * 1024 bytes -> 4 workers
    limit_4g = 4 * 1024 * 1024 * 1024
    jobs = mod._default_worker_jobs(env_workers=None, cpu_count=36, mem_limit=limit_4g)
    assert jobs == 4


def test_worker_jobs_explicit_env_honoured() -> None:
    mod = _load_runner()
    limit_4g = 4 * 1024 * 1024 * 1024
    assert mod._default_worker_jobs(env_workers="12", cpu_count=36, mem_limit=limit_4g) == 12
    assert mod._default_worker_jobs(env_workers="1", cpu_count=36, mem_limit=None) == 1
    assert mod._default_worker_jobs(env_workers="8", cpu_count=2, mem_limit=limit_4g) == 8


def test_cgroup_v2_detection_walks_parents(tmp_path: Path) -> None:
    mod = _load_runner()
    cgroup_root = tmp_path / "cgroup"
    cgroup_proc = tmp_path / "proc_cgroup"

    # Simulate cgroup tree:
    # cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/worker.scope
    scope_dir = cgroup_root / "user.slice" / "user-1000.slice" / "user@1000.service" / "app.slice" / "worker.scope"
    scope_dir.mkdir(parents=True)

    cgroup_proc.write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/worker.scope\n", encoding="utf-8")

    # Parent has max, scope has 4294967296
    (scope_dir.parent / "memory.max").write_text("max\n", encoding="utf-8")
    (scope_dir / "memory.max").write_text("4294967296\n", encoding="utf-8")

    detected = mod._detect_cgroup_v2_memory_limit(cgroup_file=cgroup_proc, cgroup_root=cgroup_root)
    assert detected == 4294967296

    # When parent tightens limit, minimum is chosen
    (scope_dir.parent / "memory.max").write_text("2147483648\n", encoding="utf-8")
    detected = mod._detect_cgroup_v2_memory_limit(cgroup_file=cgroup_proc, cgroup_root=cgroup_root)
    assert detected == 2147483648
