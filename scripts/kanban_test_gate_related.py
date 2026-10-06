#!/usr/bin/env python3
"""Kanban test gate for this repo (``.hermes-test``): run the tests related to the branch diff.

Related = changed ``tests/**/test_*.py`` files, plus every ``tests/**/test_<stem>*.py`` for each
changed source module ``<stem>.py``. The full suite (~50 min) is too slow to gate every card;
CI still runs it. Base ref: ``KANBAN_TEST_BASE`` or ``main``. No related tests -> exit 0 with a
note (docs-only diffs). Runs through ``scripts/run_tests.sh`` so CI isolation rules apply.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def changed_files(base: str) -> set[str]:
    merge_base = _git("merge-base", "HEAD", base)[0]
    files = set(_git("diff", "--name-only", merge_base))
    files |= set(_git("ls-files", "--others", "--exclude-standard"))
    return {f for f in files if (ROOT / f).exists()}


def related_tests(files: set[str]) -> list[str]:
    tests = {f for f in files if f.startswith("tests/") and Path(f).name.startswith("test_") and f.endswith(".py")}
    stems = {Path(f).stem for f in files if f.endswith(".py") and not f.startswith("tests/")}
    for stem in stems - {"__init__", "conftest"}:
        tests |= {str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob(f"test_{stem}*.py")}
    return sorted(tests)


def main() -> int:
    base = os.environ.get("KANBAN_TEST_BASE", "main")
    tests = related_tests(changed_files(base))
    if not tests:
        print(f"kanban test gate: no tests relate to the diff against {base}; nothing to run")
        return 0
    print(f"kanban test gate: {len(tests)} related test file(s) against {base}", flush=True)
    return subprocess.run([str(ROOT / "scripts/run_tests.sh"), "-j", "8", *tests], cwd=ROOT,
                          stdin=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    sys.exit(main())
