#!/usr/bin/env python3
"""Kanban test gate for this repo (``.hermes-test``): run the tests related to the branch diff.

Related = changed ``tests/**/test_*.py`` files (wherever they live), plus, for each changed source
module ``<dir>/<stem>.py``, the ``test_<stem>*.py`` files under the nearest mirror directory: search
``tests/<dir>/`` recursively, and if nothing matches walk up one level at a time (``tests/<parent>/``,
..., ``tests/``) and stop at the first level with a match. Stem matches under an ``integration``,
``e2e`` or ``docker`` directory are dropped: those suites have their own CI jobs and need sandboxes or
images. The full suite (~50 min) is too slow to gate every card; CI still runs it. A change that can
alter the rendered prompt or tool schemas also selects ``tests/ci/test_prompt_surface.py`` (its stem
never matches a changed module). Base ref:
``KANBAN_TEST_BASE`` or ``main``. No related tests -> exit 0 with a note (docs-only diffs). Runs
through ``scripts/run_tests.sh`` so CI isolation rules apply.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent

# Mirrors scripts/run_tests_parallel.py ``_SKIP_PARTS``. The runner applies its skip list only to
# discovered files, and this gate passes explicit paths, so the gate filters them itself.
SKIP_PARTS = frozenset({"integration", "e2e", "docker"})

# Explicit cross-package contract test mapping for critical runtime/root modules whose
# dependents live outside the mirror directory structure.
CROSS_DIR_TESTS: dict[str, tuple[str, ...]] = {
    "hermes_cli/venv_sync.py": (
        "tests/pm/test_source_update_launch.py",
        "tests/cron/test_restart_safe_worker.py",
    ),
    "hermes_bootstrap.py": (
        "tests/pm/test_bootstrap_import_closure.py",
        "tests/pm/test_source_update_launch.py",
    ),
    "cron/worker_bootstrap.py": (
        "tests/cron/test_restart_safe_worker.py",
    ),
}


# The snapshot test's stem never matches a changed module, so select it explicitly.
PROMPT_SURFACE_TEST = "tests/ci/test_prompt_surface.py"
# Trailing slash: directory prefix. Code dirs select non-test .py only; the snapshot tree
# selects any file. Bare names are exact paths.
PROMPT_SURFACE_PREFIXES = (
    "agent/",
    "tools/",
    "plugins/",
    "hermes_cli/",
    "tests/fixtures/prompt_surface/",
    "scripts/ci/prompt_surface.py",
    "toolsets.py",
    "run_agent.py",
    "model_tools.py",
)


def _affects_prompt_surface(path: str) -> bool:
    for prefix in PROMPT_SURFACE_PREFIXES:
        if prefix.endswith("/"):
            if not path.startswith(prefix):
                continue
            if prefix.startswith("tests/"):
                return True
            return path.endswith(".py") and not PurePosixPath(path).name.startswith("test_")
        if path == prefix:
            return True
    return False


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", stdin=subprocess.DEVNULL, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def changed_files(base: str) -> set[str]:
    merge_base = _git("merge-base", "HEAD", base)[0]
    files = set(_git("diff", "--name-only", merge_base))
    files |= set(_git("ls-files", "--others", "--exclude-standard"))
    return {f for f in files if (ROOT / f).exists()}


def _mirror_tests(source: str) -> set[str]:
    """``test_<stem>*.py`` under the nearest ``tests/<dir>`` level that has one, minus SKIP_PARTS."""
    path = PurePosixPath(source)
    tests_root = ROOT / "tests"
    for depth in range(len(path.parent.parts), -1, -1):
        level = tests_root.joinpath(*path.parent.parts[:depth])
        if not level.is_dir():
            continue
        found = {p.relative_to(ROOT).as_posix() for p in level.rglob(f"test_{path.stem}*.py")}
        found = {f for f in found if not SKIP_PARTS.intersection(PurePosixPath(f).parts)}
        if found:
            return found
    return set()


def related_tests(files: set[str]) -> list[str]:
    tests = {f for f in files if f.startswith("tests/") and Path(f).name.startswith("test_") and f.endswith(".py")}
    for f in files:
        if f.endswith(".py") and not f.startswith("tests/") and Path(f).stem not in {"__init__", "conftest"}:
            tests |= _mirror_tests(f)
            mapped = CROSS_DIR_TESTS.get(f) or ()
            for target in mapped:
                if not SKIP_PARTS.intersection(PurePosixPath(target).parts) and (ROOT / target).is_file():
                    tests.add(target)
    if any(_affects_prompt_surface(f) for f in files) and (ROOT / PROMPT_SURFACE_TEST).is_file():
        tests.add(PROMPT_SURFACE_TEST)
    return sorted(tests)


def main() -> int:
    base = os.environ.get("KANBAN_TEST_BASE", "main")
    tests = related_tests(changed_files(base))
    if not tests:
        print(f"kanban test gate: no tests relate to the diff against {base}; nothing to run")
        return 0
    print(f"kanban test gate: {len(tests)} related test file(s) against {base}", flush=True)
    return subprocess.run([str(ROOT / "scripts/run_tests.sh"), "-j", "8", *tests], cwd=ROOT,
                          stdin=subprocess.DEVNULL, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
