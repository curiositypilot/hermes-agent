"""The restart-safe external worker (``-m cron.scheduler``) loads the scheduler exactly once.

Regression for #132732: ``cron/__init__.py`` imported ``cron.scheduler`` eagerly, so ``-m
cron.scheduler`` ran the module body once as ``cron.scheduler`` and again as ``__main__``. The split
modules (``_sched``) and lazy importers (``jobs.is_job_running``) then read running sets and a lock
the worker's own fire never touched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cron

REPO_ROOT = Path(cron.__file__).resolve().parent.parent


def _run(args, tmp_path, *, extra_env=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "_HERMES_CRON_WORKER_BOOT"}
    env.update({"HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT)})
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        timeout=120)


def test_worker_entry_runs_without_runpy_double_import_warning(tmp_path):
    proc = _run(
        ["-W", "error::RuntimeWarning", "-m", "cron.scheduler",
         "--external-worker-file", str(tmp_path / "missing.json"),
         "--ack-file", str(tmp_path / "a.ready")],
        tmp_path)
    assert "found in sys.modules" not in proc.stderr
    # Reached the payload loader (missing payload -> exit 1), not a runpy traceback.
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr


def test_importing_the_package_does_not_import_the_scheduler(tmp_path):
    proc = _run(
        ["-c", "import cron, sys; assert 'cron.scheduler' not in sys.modules, 'eager'"], tmp_path)
    assert proc.returncode == 0, proc.stderr


def test_tick_is_resolved_lazily_and_unknown_names_still_raise():
    from cron import tick
    import cron.scheduler

    assert tick is cron.scheduler.tick
    with pytest.raises(AttributeError):
        cron.nope  # noqa: B018


_MAIN_ALIAS_PROBE = """
import json, runpy, sys
sys.argv = ["cron.scheduler", "--external-worker-file", sys.argv[1], "--ack-file", sys.argv[2]]
seen = {}
try:
    runpy.run_module("cron.scheduler", run_name="__main__", alter_sys=True)
except SystemExit as exc:
    seen["exit"] = exc.code
import cron.scheduler_delivery
# alter_sys restored the caller's ``__main__`` on the way out; the alias keeps the run module.
main = sys.modules["cron.scheduler"]
print(json.dumps({
    "exit": seen.get("exit"),
    "aliased": main.__name__ == "__main__",
    "split_binds_main": cron.scheduler_delivery._sched is main,
}))
"""


def test_main_module_is_the_one_the_split_modules_bind(tmp_path):
    proc = _run(
        ["-c", _MAIN_ALIAS_PROBE, str(tmp_path / "missing.json"), str(tmp_path / "a.ready")],
        tmp_path)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"exit": 1, "aliased": True, "split_binds_main": True}
