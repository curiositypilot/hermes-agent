"""Local test-command completion gate (``completion_contract = "test:<cmd>"``).

A card whose contract is ``test:<cmd>`` cannot reach ``done`` or ``review``
until ``<cmd>`` exits 0 in the card's workspace. The command runs outside any
SQLite transaction; the receipt is persisted by the lifecycle owner under the
same ownership snapshot as the terminal write
(:mod:`hermes_cli.kanban_pr_acceptance_store`).

:func:`resolve_test_command` is the create-time default for repo workspaces
(``dir:``/``worktree:`` with no explicit contract). It is stdlib-only so the
filing preflight script can import it without pulling in the kanban DB.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

TEST_PREFIX = "test:"
TAIL_LINES = 40
TIMEOUT_SECONDS = 3600
HERMES_TEST_FILE = ".hermes-test"
_NPM_PLACEHOLDER = 'echo "Error: no test specified"'
_MAKE_TEST = re.compile(r"^test\s*:(?!=)", re.MULTILINE)


def is_test_contract(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.startswith(TEST_PREFIX)


def test_command(contract: str) -> str:
    return contract[len(TEST_PREFIX):].strip()


def resolve_test_command(repo: Optional[str | os.PathLike]) -> Optional[str]:
    """The repo's test command, or None. Order: ``.hermes-test`` (first
    non-blank, non-comment line), Makefile ``test`` target, pyproject pytest
    config, package.json ``scripts.test``."""
    if not repo:
        return None
    root = Path(repo).expanduser()
    if not root.is_dir():
        return None
    marker = root / HERMES_TEST_FILE
    if marker.is_file():
        for line in _read(marker).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    for name in ("Makefile", "makefile", "GNUmakefile"):
        if (root / name).is_file() and _MAKE_TEST.search(_read(root / name)):
            return "make test"
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and "[tool.pytest" in _read(pyproject):
        # Absolute: the command runs in a linked worktree, which has no venv of its own.
        venv_python = next((str(root.resolve() / d / "bin" / "python") for d in (".venv", "venv")
                            if (root / d / "bin" / "python").exists()), None)
        return f"{venv_python or 'python3'} -m pytest -q"
    package = root / "package.json"
    if package.is_file():
        try:
            script = (json.loads(_read(package)).get("scripts") or {}).get("test")
        except (ValueError, AttributeError):
            script = None
        if isinstance(script, str) and script.strip() and _NPM_PLACEHOLDER not in script:
            return "npm test"
    return None


def default_contract(workspace_kind: Optional[str], workspace_path: Optional[str]) -> Optional[str]:
    """``test:<cmd>`` for a repo workspace whose test command resolves, else None."""
    if workspace_kind not in {"dir", "worktree"}:
        return None
    command = resolve_test_command(workspace_path)
    return TEST_PREFIX + command if command else None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def run_test_contract(contract: str, workspace: Optional[str]) -> dict:
    """Run the contract's command in ``workspace``; return the receipt.

    The receipt is what lands in run metadata ``tests`` and the ``test_gate``
    event: command, cwd, exit_code (None when it never ran or timed out),
    duration and the last ``TAIL_LINES`` lines of combined output (redacted).
    """
    command = test_command(contract)
    receipt: dict = {"kind": "test_gate", "ok": False, "command": command, "cwd": workspace,
                     "exit_code": None, "duration_s": 0.0, "tail": ""}
    if not command:
        receipt["tail"] = "completion contract test: has an empty command"
        return receipt
    wp = Path(workspace).expanduser() if workspace else None
    if wp is None or not wp.is_dir():
        receipt["tail"] = f"workspace {workspace or '(none)'} does not exist; the test gate needs the worker's checkout"
        return receipt
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=str(wp), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except OSError as exc:
        receipt["tail"] = f"could not start test command: {exc}"
        return receipt
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_SECONDS)
        receipt["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, _ = proc.communicate()
        out = (out or b"") + f"\n[test gate] timed out after {TIMEOUT_SECONDS}s\n".encode()
    receipt["duration_s"] = round(time.monotonic() - started, 2)
    receipt["tail"] = _tail(out or b"")
    receipt["ok"] = receipt["exit_code"] == 0
    return receipt


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()


def _tail(output: bytes) -> str:
    text = "\n".join(output.decode("utf-8", errors="replace").splitlines()[-TAIL_LINES:])
    try:
        from agent.redact import redact_sensitive_text
    except ImportError:  # pragma: no cover - agent package always ships with hermes_cli
        return text
    return redact_sensitive_text(text, force=True)


def refusal_detail(receipt: dict) -> str:
    code = receipt.get("exit_code")
    status = "did not run" if code is None else f"exited {code}"
    return (f"Test gate refused: `{receipt.get('command')}` {status} in {receipt.get('cwd')}. "
            f"The card stays in-flight; fix the failures and retry. Last output:\n{receipt.get('tail', '')}")
