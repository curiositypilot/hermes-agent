"""Kanban worker sandbox: run a dispatcher-spawned worker inside bubblewrap.

Opt-in per host and tenant (``kanban.sandbox: bwrap`` + ``kanban.sandbox_tenants``); off by
default. Policy, ported from the 2026-10 sandbox prototype (decision card t_60716020):

- ``/`` read-only; read-write for ``~/.hermes``, ``~/.cache``, ``~/.local/share/uv``, the task
  workspace and, for a git worktree, the repo's common git dir.
- Private ``/tmp`` and a per-run scratch dir: the shared ``~/.hermes/cache/scratch`` (which holds
  other sessions' ``hermes_rpc_*.sock`` kernel sockets) is covered by a tmpfs and only this run's
  ``kanban-<task>-r<run>`` subdir is bound back; ``TMPDIR`` and :func:`hermes_constants.get_scratch_dir`
  both resolve to it (``HERMES_SANDBOX_SCRATCH_DIR``).
- tmpfs over ``~/.ssh ~/.config/gh ~/.gnupg ~/.hermes/mcp-tokens ~/.hermes/.wallet-backups
  $XDG_RUNTIME_DIR`` (the last hides the systemd user bus, which can run commands on the host).
- ``/dev/null`` over wallets, the Telegram user session, ``.anthropic_oauth.json``, project
  ``.env`` files holding bank/transcription keys, ``.env`` backups, other profiles' ``.env`` /
  ``auth.json``, and every gateway control / loop-tick socket present at spawn.
- ``~/.hermes/.env`` (and the assignee profile's ``.env``) replaced by a Tier-1-only copy fed
  through a memfd, so no filtered secret file ever lands on disk.
- ``--clearenv`` plus an allow-list (POSIX basics, ``HERMES_*`` / ``TERMINAL_*`` minus secrets);
  no secret value is ever placed in argv.

The spawned PID is a tiny signal forwarder (``python -m hermes_cli.kanban_sandbox exec``): bwrap
does not relay SIGTERM into the sandbox, so without it a reclaim/timeout SIGTERM would either
orphan the worker or (with ``--die-with-parent``) SIGKILL it before its graceful flush. The
forwarder relays SIGTERM/SIGINT/SIGHUP to the worker's process group and exits with bwrap's code.

Only stdlib imports at module level: the forwarder and the in-sandbox probe run this module.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

SANDBOX_OFF = "off"
SANDBOX_BWRAP = "bwrap"
_OFF_VALUES = frozenset({"", "off", "false", "no", "0", "none", "disabled"})

# Marker env vars set inside the sandbox. SCRATCH_ENV is read by hermes_constants.get_scratch_dir.
SANDBOX_ENV = "HERMES_SANDBOX"
SCRATCH_ENV = "HERMES_SANDBOX_SCRATCH_DIR"

# Tier-1 keys kept in the filtered .env (worker-credential-inventory.md section 2): LLM provider
# keys, the repo PAT, terminal/tool settings. Everything else in .env is withheld.
TIER1_ENV_KEYS = frozenset({
    "OPENROUTER_API_KEY", "ANTIGRAVITY_API_KEY", "FEATHERLESS_API_KEY", "GITHUB_PAT",
    "TERMINAL_TIMEOUT", "TERMINAL_LIFETIME_SECONDS", "MESSAGING_CWD", "CAMOFOX_URL",
    "FIRECRAWL_API_URL", "BROWSER_INACTIVITY_TIMEOUT",
})

# Process env allow-list (bwrap --clearenv, then --setenv for these).
_ENV_ALLOW_EXACT = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TZ", "LANG", "LANGUAGE",
    "VIRTUAL_ENV", "PYTHONUTF8", "PYTHONIOENCODING",
    # The dispatcher pins the running install's root here for `python -m hermes_cli.main` workers.
    "PYTHONPATH",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "all_proxy", "GIT_PAGER", "PAGER", "NO_COLOR",
})
_ENV_ALLOW_PREFIXES = ("HERMES_", "TERMINAL_", "LC_")
_ENV_DENY_EXACT = frozenset({"HERMES_GATEWAY_TOKEN", "HERMES_SCRATCH_DIR", "TMP", "TEMP", "TMPDIR"})
_ENV_DENY_PREFIXES = ("HERMES_DASHBOARD_",)
_SECRET_NAME_RE = re.compile(r"(_TOKEN|_SECRET|_PASSWORD|_PASSWD|_API_KEY|_PRIVATE_KEY|_CREDENTIALS?)$")

# Masks. Home-relative and hermes-root-relative; missing paths are skipped at build time.
MASKED_HOME_DIRS = (".ssh", ".config/gh", ".gnupg")
MASKED_HERMES_DIRS = ("mcp-tokens", ".wallet-backups")
MASKED_HERMES_FILES = (
    ".solana-wallet.json", ".evm-wallet.json", ".telegram-user-session.session",
    ".bountybook_token.json", ".anthropic_oauth.json",
    "workspace/projects/finance-management/.env", "workspace/projects/autodidactics/.env",
)
MASKED_HERMES_GLOBS = (".env.bak*", "profiles/*/.env.bak*")
SOCKET_GLOBS = ("gateway.sock", "profiles/*/gateway.sock", "state/*.sock", "profiles/*/state/*.sock")
RW_HOME_DIRS = (".cache", ".local/share/uv")


class SandboxUnavailable(RuntimeError):
    """The host cannot build the configured sandbox (no bwrap). Infrastructure, never the card's
    fault; the dispatcher defers the spawn instead of running the worker unsandboxed."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _tenant_list(raw: Any) -> list[str]:
    if raw is None or raw is False:
        return []
    if isinstance(raw, str):
        items: Iterable[Any] = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        return []
    return [str(t).strip() for t in items if str(t).strip()]


def resolve_sandbox_mode(kanban_cfg: Optional[Mapping[str, Any]], tenant: Optional[str]) -> str:
    """``bwrap`` when ``kanban.sandbox`` is ``bwrap`` and the tenant is listed in
    ``kanban.sandbox_tenants`` (an empty list means every tenant); otherwise ``off``.

    YAML 1.1 reads a bare ``off`` as ``False``, so falsy values are off too. An unknown mode is
    logged and treated as off.
    """
    cfg = kanban_cfg or {}
    raw = cfg.get("sandbox")
    mode = "" if raw is None or raw is False else str(raw).strip().lower()
    if mode in _OFF_VALUES:
        return SANDBOX_OFF
    if mode != SANDBOX_BWRAP:
        logger.warning("kanban.sandbox=%r is not one of off|bwrap; workers run unsandboxed", raw)
        return SANDBOX_OFF
    tenants = _tenant_list(cfg.get("sandbox_tenants"))
    if tenants and (tenant or "").strip() not in tenants:
        return SANDBOX_OFF
    return SANDBOX_BWRAP


# ---------------------------------------------------------------------------
# Env filtering
# ---------------------------------------------------------------------------

def env_key_allowed(key: str) -> bool:
    """Whether a worker env var crosses into the sandbox (values never secrets: argv-visible)."""
    if key in _ENV_DENY_EXACT or key.startswith(_ENV_DENY_PREFIXES) or _SECRET_NAME_RE.search(key):
        return False
    return key in _ENV_ALLOW_EXACT or key.startswith(_ENV_ALLOW_PREFIXES)


def sandbox_env(worker_env: Mapping[str, str], scratch_dir: str | Path) -> dict[str, str]:
    """The env the sandboxed worker sees: the allow-listed subset plus the per-run scratch."""
    env = {k: v for k, v in worker_env.items() if env_key_allowed(k) and v is not None}
    scratch = str(scratch_dir)
    for key in ("TMPDIR", "TMP", "TEMP", SCRATCH_ENV):
        env[key] = scratch
    env[SANDBOX_ENV] = SANDBOX_BWRAP
    return env


_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def filter_env_file(text: str, allowed: Iterable[str] = TIER1_ENV_KEYS) -> str:
    """Keep only ``KEY=...`` lines whose key is allow-listed (comments and blanks dropped)."""
    keep = set(allowed)
    out = []
    for line in text.splitlines():
        m = _ENV_LINE_RE.match(line)
        if m and m.group(1) in keep:
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def _memfd_with(data: bytes, name: str) -> int:
    fd = os.memfd_create(name, 0)
    os.write(fd, data)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


# ---------------------------------------------------------------------------
# argv builder
# ---------------------------------------------------------------------------

def forwarder_argv(python: Optional[str] = None) -> list[str]:
    """Run THIS file as a script (stdlib only), never ``-m hermes_cli...``: the worker's cwd may
    be a hermes-agent worktree whose package would shadow the installed one. ``-P`` keeps the
    script dir off ``sys.path``."""
    return [python or sys.executable, "-P", str(Path(__file__).resolve())]

@dataclass
class SandboxPlan:
    """A built sandbox launch: ``argv`` (forwarder + bwrap + worker command), the minimal env
    for the outer wrapper, the fds bwrap reads its data binds from, and the per-run scratch."""

    argv: list[str]
    outer_env: dict[str, str]
    pass_fds: tuple[int, ...]
    scratch_dir: Path
    bwrap_argv: list[str] = field(default_factory=list)

    def close(self) -> None:
        """Close the memfds in this process once the child holds them."""
        for fd in self.pass_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.pass_fds = ()

    def log_line(self) -> bytes:
        return ("[kanban sandbox] bwrap argv: " + shlex.join(self.argv) + "\n").encode("utf-8", "replace")


def _git_common_dir(workspace: Path) -> Optional[Path]:
    """Common git dir of a linked worktree (its objects/refs live outside the workspace)."""
    if not (workspace / ".git").is_file():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return Path(path) if out.returncode == 0 and path and os.path.isdir(path) else None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _existing(paths: Iterable[Path], *, kind: str) -> list[Path]:
    out = []
    for p in paths:
        if kind == "dir" and p.is_dir() and not p.is_symlink():
            out.append(p)
        elif kind == "file" and (p.is_file() or p.is_socket()) and not p.is_symlink():
            out.append(p)
    return out


def masked_paths(hermes_root: Path, home: Path, *, runtime_dir: Optional[str],
                 assignee_home: Optional[Path] = None) -> tuple[list[Path], list[Path]]:
    """``(dirs_for_tmpfs, files_for_dev_null)`` present on this host right now."""
    dirs = [home / d for d in MASKED_HOME_DIRS] + [hermes_root / d for d in MASKED_HERMES_DIRS]
    if runtime_dir:
        dirs.append(Path(runtime_dir))
    files = [hermes_root / f for f in MASKED_HERMES_FILES]
    for pattern in (*MASKED_HERMES_GLOBS, *SOCKET_GLOBS):
        files.extend(Path(p) for p in sorted(glob.glob(str(hermes_root / pattern))))
    own = assignee_home.resolve() if assignee_home else None
    for prof in sorted((hermes_root / "profiles").glob("*/")):
        if own is not None and prof.resolve() == own:
            continue
        files.extend((prof / ".env", prof / "auth.json"))
    return _existing(dirs, kind="dir"), _existing(dict.fromkeys(files), kind="file")


def build_sandbox(
    command: Sequence[str],
    *,
    worker_env: Mapping[str, str],
    workspace: str | Path,
    run_key: str,
    hermes_root: Optional[Path] = None,
    home: Optional[Path] = None,
    assignee_home: Optional[str | Path] = None,
    bwrap: Optional[str] = None,
    python: Optional[str] = None,
) -> SandboxPlan:
    """Build the forwarder + bwrap argv around ``command``. Raises :class:`SandboxUnavailable`
    when bwrap is not installed (fail closed: never fall back to an unsandboxed worker)."""
    bwrap = bwrap or shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable("kanban.sandbox=bwrap but `bwrap` is not on PATH (apt install bubblewrap)")
    if hermes_root is None:
        from hermes_constants import get_default_hermes_root
        hermes_root = get_default_hermes_root()
    hermes_root = Path(hermes_root)
    home = Path(home or worker_env.get("HOME") or Path.home())
    ws = Path(workspace)
    own_home = Path(assignee_home) if assignee_home else None

    shared_scratch = hermes_root / "cache" / "scratch"
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", run_key) or "run"
    scratch = shared_scratch / f"kanban-{safe_key}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chmod(scratch, 0o700)

    a: list[str] = [
        bwrap, "--die-with-parent",
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
        "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--tmpfs", "/tmp",  # no-tmp: ok — a private, empty /tmp inside the sandbox
    ]
    # Read-write grants first: a later mask must win over any grant that contains it.
    rw = [hermes_root, *(home / d for d in RW_HOME_DIRS)]
    if ws.is_dir():
        rw.append(ws)
        common = _git_common_dir(ws)
        if common is not None:
            rw.append(common)
    seen: set[str] = set()
    for p in rw:
        if p.is_dir() and str(p) not in seen:
            seen.add(str(p))
            a += ["--bind", str(p), str(p)]

    runtime_dir = worker_env.get("XDG_RUNTIME_DIR") or os.environ.get("XDG_RUNTIME_DIR")
    mask_dirs, mask_files = masked_paths(hermes_root, home, runtime_dir=runtime_dir, assignee_home=own_home)
    for d in mask_dirs:
        a += ["--tmpfs", str(d)]
    for f in mask_files:
        a += ["--ro-bind", "/dev/null", str(f)]

    # Private scratch: hide every other session's entries (kernel RPC sockets), keep this run's.
    if shared_scratch.is_dir():
        a += ["--tmpfs", str(shared_scratch)]
    a += ["--bind", str(scratch), str(scratch)]

    # Tier-1-only .env copies, fed via memfd (bwrap copies the fd's content at setup).
    fds: list[int] = []
    env_files = [hermes_root / ".env"]
    if own_home is not None and own_home.resolve() != hermes_root.resolve():
        env_files.append(own_home / ".env")
    try:
        for env_path in env_files:
            if not env_path.is_file():
                continue
            try:
                text = env_path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                text = ""
            fd = _memfd_with(filter_env_file(text).encode("utf-8"), "hermes-env")
            fds.append(fd)
            a += ["--perms", "0600", "--ro-bind-data", str(fd), str(env_path)]
    except Exception:
        for fd in fds:
            os.close(fd)
        raise

    a.append("--clearenv")
    inner_env = sandbox_env(worker_env, scratch)
    for key in sorted(inner_env):
        a += ["--setenv", key, inner_env[key]]
    if ws.is_dir():
        a += ["--chdir", str(ws)]
    bwrap_argv = [*a, *command]

    argv = [*forwarder_argv(python), "exec", *bwrap_argv]
    outer = {k: v for k, v in inner_env.items()}
    for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        if worker_env.get(key):
            outer[key] = worker_env[key]
    return SandboxPlan(argv=argv, outer_env=outer, pass_fds=tuple(fds), scratch_dir=scratch,
                       bwrap_argv=bwrap_argv)


# ---------------------------------------------------------------------------
# Dispatcher seam
# ---------------------------------------------------------------------------

# Per-task Kanban vars a spawn pins for itself; anything else HERMES_KANBAN_* in the worker env
# was inherited from a parent worker (nested dispatch) and must not cross into this sandbox.
_HOST_KANBAN_VARS = frozenset({
    "HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_ROOT",
    "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_LOGS_ROOT",
})


def _pinned_kanban_vars(task: Any) -> set[str]:
    pinned = set(_HOST_KANBAN_VARS)
    optional = (
        ("HERMES_KANBAN_BRANCH", task.branch_name),
        ("HERMES_KANBAN_RUN_ID", task.current_run_id is not None),
        ("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock),
        ("HERMES_KANBAN_PINNED", getattr(task, "route_pinned", False)),
        ("HERMES_KANBAN_GOAL_MODE", task.goal_mode),
        ("HERMES_KANBAN_GOAL_MAX_TURNS", task.goal_mode),
    )
    pinned.update(name for name, applies in optional if applies)
    return pinned


def plan_for_task(task: Any, command: Sequence[str], env: Mapping[str, str], workspace: str,
                  profile_home: Optional[str]) -> Optional[SandboxPlan]:
    """The :class:`SandboxPlan` for this worker spawn when ``kanban.sandbox`` applies to the task's
    tenant, else None. Config is read per spawn, so a flip applies at the next spawn with no
    restart. An unreadable config raises (fail closed: never guess the sandbox is off), and so
    does :class:`SandboxUnavailable` (the dispatcher defers the spawn, never runs it bare).
    """
    if os.name == "nt":
        return None
    from hermes_cli.config import load_config

    if resolve_sandbox_mode(load_config().get("kanban") or {}, task.tenant) != SANDBOX_BWRAP:
        return None
    pinned = _pinned_kanban_vars(task)
    worker_env = {k: v for k, v in env.items() if not k.startswith("HERMES_KANBAN_") or k in pinned}
    run = task.current_run_id if task.current_run_id is not None else "x"
    return build_sandbox(command, worker_env=worker_env, workspace=workspace,
                         run_key=f"{task.id}-r{run}", assignee_home=profile_home)


@contextlib.contextmanager
def worker_sandbox(task: Any, command: Sequence[str], env: dict[str, str], workspace: str,
                   profile_home: Optional[str]):
    """Yield ``(argv, env, pass_fds, log_prefix)`` for the worker spawn: the sandboxed launch
    when ``kanban.sandbox`` applies, else the inputs unchanged with no fds and an empty prefix.
    The memfds the sandbox reads its data binds from are closed on exit, once Popen has handed
    them to the child.
    """
    plan = plan_for_task(task, command, env, workspace, profile_home)
    if plan is None:
        yield list(command), env, (), b""
        return
    try:
        yield list(plan.argv), dict(plan.outer_env), plan.pass_fds, plan.log_line()
    finally:
        plan.close()


# ---------------------------------------------------------------------------
# Signal forwarder (the spawned PID)
# ---------------------------------------------------------------------------

def _process_group_members(pgid: int, exclude: set[int]) -> list[int]:
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in exclude:
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                stat = fh.read().decode("utf-8", "replace")
            # Fields after the ")" closing comm: state ppid pgrp ...
            fields = stat.rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                members.append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    return members


def exec_forwarding(argv: Sequence[str]) -> int:
    """Run ``argv`` (bwrap ...) and relay shutdown signals to the sandboxed worker.

    bwrap's monitor dies on SIGTERM without relaying it, and ``--die-with-parent`` then SIGKILLs
    the worker. Instead every SIGTERM/SIGINT/SIGHUP is sent to the members of this process group
    except this process and bwrap's monitor: that is the worker (bwrap's in-namespace init has no
    handler, so the kernel drops it there). The worker's own handler flushes and exits; bwrap then
    returns the worker's status, which becomes this process's exit code.
    """
    proc = subprocess.Popen(list(argv), close_fds=False)
    pgid = os.getpgrp()

    def _relay(signum, _frame):
        for pid in _process_group_members(pgid, {os.getpid(), proc.pid}):
            try:
                os.kill(pid, signum)
            except OSError:
                pass

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signal.signal(getattr(signal, name), _relay)
    # The forwarder lives exactly as long as the sandboxed worker: reclaim and timeout end it
    # through the relayed SIGTERM, and the dispatcher owns the runtime cap.
    # health: allow HX006 -- unbounded by design, the wait IS the worker's lifetime
    rc = proc.wait()
    return rc if rc >= 0 else 128 - rc


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _probe(spec: dict) -> list[dict]:
    """Runs INSIDE the sandbox: check each hidden path and each capability a worker needs."""
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    def run(argv: list[str], cwd: Optional[str] = None, timeout: int = 120) -> tuple[int, str]:
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
            return p.returncode, (p.stdout + p.stderr).strip()[-300:]
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, str(exc)

    for path in spec["hidden_files"]:
        try:
            with open(path, "rb") as fh:
                data = fh.read(64)
            check(f"hidden file {path}", data == b"", f"{len(data)} bytes readable" if data else "empty")
        except PermissionError:
            check(f"hidden file {path}", True, "permission denied")
        except OSError as exc:
            check(f"hidden file {path}", True, type(exc).__name__)
    for path in spec["hidden_dirs"]:
        try:
            entries = os.listdir(path)
        except OSError as exc:
            entries, path = [], f"{path} ({type(exc).__name__})"
        check(f"hidden dir {path}", not entries, f"{len(entries)} entries" if entries else "empty")
    for path in spec["sockets"]:
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(1)
        try:
            s.connect(path)
            check(f"socket {path}", False, "CONNECTED")
        except OSError as exc:
            check(f"socket {path}", True, type(exc).__name__)
        finally:
            s.close()
    try:
        visible = set(os.listdir(spec["shared_scratch"]))
    except OSError:
        visible = set()
    leaked = sorted(visible & set(spec["shared_scratch_entries"]))
    check("shared scratch hides other sessions", not leaked,
          f"{len(leaked)} foreign entries visible" if leaked else f"{len(spec['shared_scratch_entries'])} hidden")

    leaked_env = sorted(k for k in spec["forbidden_env"] if k in os.environ)
    check("Tier-3 env vars absent", not leaked_env, ",".join(leaked_env) or f"{len(spec['forbidden_env'])} absent")
    for env_path in spec["env_files"]:
        try:
            text = Path(env_path).read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            check(f"filtered {env_path}", False, str(exc))
            continue
        keys = {m.group(1) for m in map(_ENV_LINE_RE.match, text.splitlines()) if m}
        extra = sorted(keys - set(spec["tier1"]))
        check(f"filtered {env_path} is Tier-1 only", not extra, ",".join(extra) or f"{len(keys)} Tier-1 keys")

    scratch = os.environ.get("TMPDIR", "")
    try:
        Path(scratch, ".probe").write_text("ok", encoding="utf-8")
        check("TMPDIR is the per-run scratch and writable", scratch == spec["scratch"], scratch)
    except OSError as exc:
        check("TMPDIR is the per-run scratch and writable", False, str(exc))
    private_tmp = Path("/tmp")  # no-tmp: ok — probes the sandbox's private /tmp, not scratch space
    try:
        (private_tmp / ".probe").write_text("ok", encoding="utf-8")
        entries = sorted(os.listdir(private_tmp))
        check("private tmp is empty and writable", entries == [".probe"], ",".join(entries)[:80])
    except OSError as exc:
        check("private tmp is empty and writable", False, str(exc))
    bashrc = Path(spec["home"]) / ".bashrc"
    try:
        with open(bashrc, "a", encoding="utf-8"):
            pass
        check("write ~/.bashrc blocked", False, "writable")
    except OSError as exc:
        check("write ~/.bashrc blocked", True, type(exc).__name__)

    ws = spec["workspace"]
    git = ["git", "-c", "user.email=probe@sandbox", "-c", "user.name=probe"]
    rc, out = run([*git, "init", "-q", "repo"], cwd=ws)
    if rc == 0:
        rc, out = run([*git, "-C", "repo", "commit", "-q", "--allow-empty", "-m", "init"], cwd=ws)
    if rc == 0:
        rc, out = run([*git, "-C", "repo", "worktree", "add", "-q", "../wt", "-b", "probe"], cwd=ws)
    check("git init + commit + worktree add", rc == 0, out)
    uv = shutil.which("uv")
    rc, out = run([uv, "venv", "-q", os.path.join(ws, "venv")], cwd=ws) if uv else (127, "uv not on PATH")
    check("uv venv", rc == 0, out)
    rc, out = run([*spec["hermes_argv"], "kanban", "stats", "--json"], cwd=ws)
    check("hermes kanban CLI (stats)", rc == 0, out.splitlines()[-1][-120:] if out else "")
    try:
        conn = sqlite3.connect(spec["kanban_db"], timeout=10)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        conn.close()
        check("kanban DB write lock", True, spec["kanban_db"])
    except sqlite3.Error as exc:
        check("kanban DB write lock", False, str(exc))
    return results


def run_selftest(*, hermes_root: Optional[Path] = None, as_json: bool = False,
                 stream=None) -> int:
    """``hermes kanban sandbox-selftest``: build the worker sandbox exactly as the dispatcher
    would, run the probe inside it, print the checks, return 0 only when every check passes."""
    import tempfile

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _resolve_hermes_argv
    from hermes_constants import get_default_hermes_root

    out = stream or sys.stdout
    root = Path(hermes_root or get_default_hermes_root())
    home = Path.home()
    base = root / "cache" / "kanban-sandbox-selftest"
    base.mkdir(parents=True, exist_ok=True)
    ws = Path(tempfile.mkdtemp(prefix="ws-", dir=base))
    run_key = f"selftest-{os.getpid()}"
    worker_env = dict(os.environ)
    worker_env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path())
    worker_env["HERMES_KANBAN_WORKSPACE"] = str(ws)
    runtime_dir = worker_env.get("XDG_RUNTIME_DIR")
    mask_dirs, mask_files = masked_paths(root, home, runtime_dir=runtime_dir, assignee_home=root)
    shared = root / "cache" / "scratch"
    try:
        shared_entries = sorted(e for e in os.listdir(shared) if not e.startswith(f"kanban-{run_key}"))
    except OSError:
        shared_entries = []
    plan = None
    try:
        hermes_argv = _resolve_hermes_argv()
        spec_holder: dict[str, Any] = {}
        plan = build_sandbox(["true"], worker_env=worker_env, workspace=ws, run_key=run_key,
                             hermes_root=root, home=home, assignee_home=root)
        spec_holder.update({
            "hidden_files": [str(f) for f in mask_files if not f.is_socket()],
            "sockets": [str(f) for f in mask_files if f.is_socket()],
            "hidden_dirs": [str(d) for d in mask_dirs],
            "shared_scratch": str(shared), "shared_scratch_entries": shared_entries,
            "forbidden_env": sorted(k for k in worker_env if not env_key_allowed(k)
                                    and k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR",
                                                  "PWD", "OLDPWD", "SHLVL", "_")),
            "env_files": [str(root / ".env")] if (root / ".env").is_file() else [],
            "tier1": sorted(TIER1_ENV_KEYS), "scratch": str(plan.scratch_dir), "home": str(home),
            "workspace": str(ws), "hermes_argv": hermes_argv, "kanban_db": worker_env["HERMES_KANBAN_DB"],
        })
        probe_cmd = [*forwarder_argv(), "probe", json.dumps(spec_holder)]
        argv = [*plan.argv[:-1], *probe_cmd]  # swap the placeholder command for the probe
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=600, pass_fds=plan.pass_fds, env=plan.outer_env,
                              stdin=subprocess.DEVNULL)
    except SandboxUnavailable as exc:
        print(f"sandbox-selftest: FAIL — {exc}", file=out)
        return 1
    finally:
        if plan is not None:
            plan.close()
    try:
        results = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        print(f"sandbox-selftest: FAIL — probe produced no result (rc={proc.returncode})\n"
              f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", file=out)
        return 1
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        if plan is not None:
            shutil.rmtree(plan.scratch_dir, ignore_errors=True)
    failed = [r for r in results if not r["ok"]]
    if as_json:
        print(json.dumps({"ok": not failed, "checks": results}, indent=2), file=out)
    else:
        for r in results:
            print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['check']}  ({r['detail']})", file=out)
        print(f"sandbox-selftest: {len(results) - len(failed)}/{len(results)} checks passed", file=out)
    return 0 if not failed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["exec"] and len(args) > 1:
        return exec_forwarding(args[1:])
    if args[:1] == ["probe"] and len(args) == 2:
        print(json.dumps(_probe(json.loads(args[1]))))
        return 0
    print("usage: python -m hermes_cli.kanban_sandbox exec BWRAP_ARGV... | probe SPEC_JSON", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
