"""kanban.sandbox (bwrap) — policy resolution, env filtering, argv builder, spawn wiring.

The argv builder runs against a fake home tree in tmp_path; the real-bwrap tests (skipped where
bwrap or unprivileged user namespaces are missing) prove the built argv actually hides what it
claims to hide, which a string assertion on the argv cannot.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_sandbox as kbs

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bwrap sandbox is Linux-only")


def _bwrap_works() -> bool:
    if not shutil.which("bwrap"):
        return False
    try:
        return subprocess.run(["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bwrap = pytest.mark.skipif(not _bwrap_works(), reason="bwrap with unprivileged userns unavailable")


# --- policy --------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg,tenant,expected", [
    ({}, "style-and-clothes", "off"),
    ({"sandbox": False}, "x", "off"),  # YAML 1.1 reads bare `off` as False
    ({"sandbox": "off", "sandbox_tenants": ["x"]}, "x", "off"),
    ({"sandbox": "bwrap"}, None, "bwrap"),  # empty tenant list = every tenant
    ({"sandbox": "bwrap", "sandbox_tenants": ["style-and-clothes"]}, "style-and-clothes", "bwrap"),
    ({"sandbox": "bwrap", "sandbox_tenants": ["style-and-clothes"]}, "movis", "off"),
    ({"sandbox": "bwrap", "sandbox_tenants": ["style-and-clothes"]}, None, "off"),
    ({"sandbox": "BWRAP", "sandbox_tenants": "a, style-and-clothes"}, "style-and-clothes", "bwrap"),
    ({"sandbox": "gvisor"}, "x", "off"),  # unknown mode never half-applies
])
def test_resolve_sandbox_mode(cfg, tenant, expected):
    assert kbs.resolve_sandbox_mode(cfg, tenant) == expected


def test_default_config_keeps_sandbox_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    kanban = DEFAULT_CONFIG["kanban"]
    assert kbs.resolve_sandbox_mode(kanban, "style-and-clothes") == kbs.SANDBOX_OFF


# --- env filtering ------------------------------------------------------------------------

def test_env_allow_list_drops_secrets_and_keeps_worker_vars():
    worker_env = {
        "PATH": "/usr/bin", "HOME": "/h", "LANG": "C.UTF-8", "LC_ALL": "C",
        "HERMES_HOME": "/h/.hermes", "HERMES_KANBAN_TASK": "t_1", "TERMINAL_CWD": "/ws", "PYTHONPATH": "/src",
        "HERMES_GATEWAY_TOKEN": "s", "HERMES_DASHBOARD_BASIC_AUTH_SECRET": "s",
        "BINANCE_SECRET": "s", "OPENROUTER_API_KEY": "s", "GITHUB_PAT": "s",
        "TELEGRAM_BOT_TOKEN": "s", "HERMES_SOME_API_KEY": "s", "DBUS_SESSION_BUS_ADDRESS": "unix:x",
        "TMPDIR": "/shared/scratch", "HERMES_SCRATCH_DIR": "/shared/scratch",
    }
    env = kbs.sandbox_env(worker_env, "/run/scratch")
    for kept in ("PATH", "HOME", "LANG", "LC_ALL", "HERMES_HOME", "HERMES_KANBAN_TASK", "TERMINAL_CWD", "PYTHONPATH"):
        assert env[kept] == worker_env[kept]
    for dropped in ("HERMES_GATEWAY_TOKEN", "HERMES_DASHBOARD_BASIC_AUTH_SECRET", "BINANCE_SECRET",
                    "OPENROUTER_API_KEY", "GITHUB_PAT", "TELEGRAM_BOT_TOKEN", "HERMES_SOME_API_KEY",
                    "DBUS_SESSION_BUS_ADDRESS", "HERMES_SCRATCH_DIR"):
        assert dropped not in env
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env[kbs.SCRATCH_ENV] == "/run/scratch"


def test_filter_env_file_keeps_only_tier1_lines():
    text = ("# comment\nOPENROUTER_API_KEY=k1\nBINANCE_SECRET=nope\nexport GITHUB_PAT=k2\n"
            "HERMES_GATEWAY_TOKEN=nope\n\nTERMINAL_TIMEOUT=60\n")
    out = kbs.filter_env_file(text)
    keys = [line.split("=")[0].replace("export ", "") for line in out.splitlines()]
    assert set(keys) == {"OPENROUTER_API_KEY", "GITHUB_PAT", "TERMINAL_TIMEOUT"}
    assert set(keys) <= kbs.TIER1_ENV_KEYS


# --- argv builder -------------------------------------------------------------------------

def _fake_host(tmp_path: Path):
    home = tmp_path / "home"
    root = home / ".hermes"
    for d in (".ssh", ".config/gh", ".cache", ".local/share/uv", ".hermes/mcp-tokens",
              ".hermes/cache/scratch", ".hermes/state", ".hermes/profiles/other/state",
              ".hermes/profiles/mine", "run"):
        (home / d).mkdir(parents=True, exist_ok=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE")
    (home / ".bashrc").write_text("# rc\n")
    (root / ".solana-wallet.json").write_text("WALLET")
    (root / ".env").write_text("OPENROUTER_API_KEY=ok\nBINANCE_SECRET=leak\n")
    (root / ".env.bak-1").write_text("BINANCE_SECRET=leak\n")
    (root / "profiles" / "other" / ".env").write_text("TELEGRAM_BOT_TOKEN=leak\n")
    (root / "profiles" / "other" / "auth.json").write_text("{\"leak\": 1}")
    (root / "profiles" / "mine" / ".env").write_text("FEATHERLESS_API_KEY=ok\nANTHROPIC_TOKEN=leak\n")
    (root / "profiles" / "mine" / "auth.json").write_text("{\"own\": 1}")
    (root / "cache" / "scratch" / "foreign-entry").write_text("x")
    socks = []
    cwd = os.getcwd()
    for rel in ("gateway.sock", "profiles/other/gateway.sock", "state/gateway.loop-tick.1.sock",
                "profiles/other/state/gateway.loop-tick.2.sock"):
        # tmp_path can exceed sun_path (108 bytes): bind relative to the socket's own dir.
        target = root / rel
        os.chdir(target.parent)
        try:
            s = socket.socket(socket.AF_UNIX)
            s.bind(target.name)
            s.listen(1)
        finally:
            os.chdir(cwd)
        socks.append(s)
    ws = tmp_path / "ws"
    ws.mkdir()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "XDG_RUNTIME_DIR": str(home / "run"),
           "HERMES_HOME": str(root / "profiles" / "mine"), "BINANCE_SECRET": "leak"}
    return SimpleNamespace(home=home, root=root, ws=ws, env=env, socks=socks)


def _pairs(argv, flag):
    return [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == flag]


def test_build_sandbox_argv_shape(tmp_path):
    host = _fake_host(tmp_path)
    plan = kbs.build_sandbox(["hermes", "chat"], worker_env=host.env, workspace=host.ws, run_key="t_1-r5",
                             hermes_root=host.root, home=host.home, assignee_home=host.root / "profiles" / "mine",
                             bwrap="/usr/bin/bwrap", python="/py")
    try:
        argv = plan.argv
        assert argv[:3] == ["/py", "-P", str(Path(kbs.__file__).resolve())]
        assert argv[3] == "exec" and argv[4] == "/usr/bin/bwrap"
        assert argv[-2:] == ["hermes", "chat"]
        assert "--die-with-parent" in argv and "--clearenv" in argv
        assert ("/", "/") in _pairs(argv, "--ro-bind")
        binds = dict(_pairs(argv, "--bind"))
        assert binds[str(host.root)] == str(host.root) and binds[str(host.ws)] == str(host.ws)
        assert str(plan.scratch_dir) in binds and plan.scratch_dir.name == "kanban-t_1-r5"

        tmpfs = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
        for d in (host.home / ".ssh", host.home / ".config/gh", host.root / "mcp-tokens", host.home / "run",
                  host.root / "cache" / "scratch"):
            assert str(d) in tmpfs
        devnull = [dst for src, dst in _pairs(argv, "--ro-bind") if src == "/dev/null"]
        for f in (".solana-wallet.json", ".env.bak-1", "profiles/other/.env", "profiles/other/auth.json",
                  "gateway.sock", "profiles/other/gateway.sock", "state/gateway.loop-tick.1.sock",
                  "profiles/other/state/gateway.loop-tick.2.sock"):
            assert str(host.root / f) in devnull
        # The assignee's own auth.json (LLM OAuth) stays readable; its .env is filtered instead.
        assert str(host.root / "profiles/mine/auth.json") not in devnull
        data_dsts = [argv[i + 2] for i, a in enumerate(argv) if a == "--ro-bind-data"]
        assert data_dsts == [str(host.root / ".env"), str(host.root / "profiles/mine/.env")]
        assert len(plan.pass_fds) == 2

        # Masks come after every rw grant, the private scratch bind after the shared-scratch tmpfs.
        last_grant = max(i for i, a in enumerate(argv) if a == "--bind" and argv[i + 1] != str(plan.scratch_dir))
        first_mask = min(i for i, a in enumerate(argv) if a in ("--tmpfs",) and i > 20)
        assert first_mask > last_grant
        assert argv.index(str(host.root / "cache" / "scratch")) < argv.index(str(plan.scratch_dir))
        # No secret value ever lands in argv.
        assert "leak" not in " ".join(argv)
        setenv = dict(_pairs(argv, "--setenv"))
        assert "BINANCE_SECRET" not in setenv and setenv["TMPDIR"] == str(plan.scratch_dir)
        assert b"bwrap argv" in plan.log_line()
    finally:
        plan.close()


def test_build_sandbox_fails_closed_without_bwrap(tmp_path, monkeypatch):
    host = _fake_host(tmp_path)
    monkeypatch.setattr(kbs.shutil, "which", lambda _name: None)
    with pytest.raises(kbs.SandboxUnavailable):
        kbs.build_sandbox(["true"], worker_env=host.env, workspace=host.ws, run_key="r",
                          hermes_root=host.root, home=host.home)


@needs_bwrap
def test_built_sandbox_hides_what_it_claims(tmp_path):
    host = _fake_host(tmp_path)
    script = (
        "import os,socket,sys\n"
        "def rd(p):\n"
        "    try: return open(p).read()\n"
        "    except OSError as e: return type(e).__name__\n"
        "def conn(p):\n"
        "    s=socket.socket(socket.AF_UNIX)\n"
        "    try: s.connect(p); return 'CONNECTED'\n"
        "    except OSError as e: return type(e).__name__\n"
        "def wr(p):\n"
        "    try: open(p,'a').close(); return 'WRITABLE'\n"
        "    except OSError as e: return type(e).__name__\n"
        f"H={str(host.home)!r}; R={str(host.root)!r}\n"
        "print(repr({'ssh': os.listdir(H+'/.ssh'), 'wallet': rd(R+'/.solana-wallet.json'),"
        " 'env': rd(R+'/.env'), 'own_env': rd(R+'/profiles/mine/.env'), 'own_auth': rd(R+'/profiles/mine/auth.json'),"
        " 'other_auth': rd(R+'/profiles/other/auth.json'), 'bak': rd(R+'/.env.bak-1'),"
        " 'gw': conn(R+'/gateway.sock'), 'tick': conn(R+'/state/gateway.loop-tick.1.sock'),"
        " 'scratch': os.listdir(R+'/cache/scratch'), 'bashrc': wr(H+'/.bashrc'), 'ws': wr(os.getcwd()+'/f'),"
        " 'tmpdir': os.environ.get('TMPDIR'), 'secret': os.environ.get('BINANCE_SECRET')}))\n"
    )
    plan = kbs.build_sandbox([sys.executable, "-c", script], worker_env=host.env, workspace=host.ws,
                             run_key="t_2-r1", hermes_root=host.root, home=host.home,
                             assignee_home=host.root / "profiles" / "mine")
    try:
        proc = subprocess.run(plan.argv, capture_output=True, text=True, timeout=60, env=plan.outer_env,
                              pass_fds=plan.pass_fds)
    finally:
        plan.close()
    assert proc.returncode == 0, proc.stderr
    seen = eval(proc.stdout.strip().splitlines()[-1])  # noqa: S307 -- our own repr output
    assert seen["ssh"] == []
    assert seen["wallet"] in ("", "PermissionError")
    assert seen["env"] == "OPENROUTER_API_KEY=ok\n"
    assert seen["own_env"] == "FEATHERLESS_API_KEY=ok\n"
    assert seen["own_auth"] == "{\"own\": 1}"
    assert seen["other_auth"] in ("", "PermissionError") and seen["bak"] in ("", "PermissionError")
    assert seen["gw"] != "CONNECTED" and seen["tick"] != "CONNECTED"
    assert seen["scratch"] == ["kanban-t_2-r1"]
    assert seen["bashrc"] != "WRITABLE" and seen["ws"] == "WRITABLE"
    assert seen["tmpdir"] == str(plan.scratch_dir) and seen["secret"] is None
    # Host files are untouched by the masks.
    assert (host.root / ".env").read_text().startswith("OPENROUTER_API_KEY=ok\nBINANCE_SECRET=leak")


@needs_bwrap
def test_forwarder_relays_sigterm_to_the_worker(tmp_path):
    host = _fake_host(tmp_path)
    marker = host.ws / "got-term"
    script = (
        "import signal,sys,time\n"
        f"signal.signal(signal.SIGTERM, lambda *a: (open({str(marker)!r},'w').write('1'), sys.exit(7)))\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    plan = kbs.build_sandbox([sys.executable, "-c", script], worker_env=host.env, workspace=host.ws,
                             run_key="t_3-r1", hermes_root=host.root, home=host.home)
    try:
        proc = subprocess.Popen(plan.argv, stdout=subprocess.PIPE, text=True, env=plan.outer_env,
                                pass_fds=plan.pass_fds, start_new_session=True)
    finally:
        plan.close()
    assert proc.stdout.readline().strip() == "ready"
    proc.terminate()
    assert proc.wait(timeout=20) == 7
    assert marker.read_text() == "1"


# --- spawn wiring -------------------------------------------------------------------------

def _patch_spawn(monkeypatch, tmp_path, cfg):
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles, config
    from tools.environments import local
    from tools import process_registry
    from agent import secret_scope

    monkeypatch.setattr(profiles, "normalize_profile_name", lambda name: name)
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda _name: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(local, "build_subprocess_env", lambda **_kw: {"PATH": "/usr/bin", "HERMES_KANBAN_TASK": "parent",
                                                                     "HERMES_KANBAN_BRANCH": "parent-branch"})
    monkeypatch.setattr(local, "_is_routed_home", lambda _home: False)
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: False)
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda env: env)
    monkeypatch.setattr(kbd, "_worker_argv", lambda *_a: ["hermes", "chat"])
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda _t, command: ["systemd-run", "--scope", *command])
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kbd._kb, "worker_logs_dir", lambda **_kw: tmp_path / "logs")
    monkeypatch.setattr(kbd._kb, "kanban_db_path", lambda **_kw: tmp_path / "kanban.db")
    monkeypatch.setattr(kbd._kb, "workspaces_root", lambda **_kw: str(tmp_path / "workspaces"))
    monkeypatch.setattr(kbd._kb, "get_current_board", lambda: "default")
    monkeypatch.setattr(kbd._kb, "_normalize_board_slug", lambda _b: "default")
    monkeypatch.setattr(config, "load_config", lambda: {"kanban": cfg})
    captured = {}
    monkeypatch.setattr(kbd.subprocess, "Popen",
                        lambda argv, **kw: (captured.update(kw, argv=argv) or SimpleNamespace(pid=4242)))
    return kbd, captured


def _task(tenant):
    return SimpleNamespace(id="t_9", assignee="worker", tenant=tenant, data_class=None, branch_name=None,
                           current_run_id=3, claim_lock="host:1", goal_mode=False, max_runtime_seconds=None)


def test_spawn_unsandboxed_when_flag_off(monkeypatch, tmp_path):
    kbd, captured = _patch_spawn(monkeypatch, tmp_path, {"sandbox": "off", "sandbox_tenants": ["pilot"]})
    assert kbd._default_spawn(_task("pilot"), str(tmp_path)) == 4242
    assert captured["argv"] == ["systemd-run", "--scope", "hermes", "chat"]
    assert captured["pass_fds"] == ()


def test_spawn_wraps_listed_tenant_between_scope_and_hermes(monkeypatch, tmp_path):
    host = _fake_host(tmp_path)
    monkeypatch.setattr(kbs, "build_sandbox", _builder_in(host))
    kbd, captured = _patch_spawn(monkeypatch, tmp_path, {"sandbox": "bwrap", "sandbox_tenants": ["pilot"]})

    kbd._default_spawn(_task("other"), str(host.ws))
    assert "exec" not in captured["argv"]

    kbd._default_spawn(_task("pilot"), str(host.ws))
    argv = captured["argv"]
    assert argv[:2] == ["systemd-run", "--scope"] and argv[-2:] == ["hermes", "chat"]
    assert argv.index("exec") < argv.index("--clearenv") < argv.index("hermes")
    setenv = dict(_pairs(argv, "--setenv"))
    # The worker's own pins win; a parent worker's leftover branch pin is stripped.
    assert setenv["HERMES_KANBAN_TASK"] == "t_9" and "HERMES_KANBAN_BRANCH" not in setenv
    assert "BINANCE_SECRET" not in captured["env"]
    log = (tmp_path / "logs" / "t_9.log").read_bytes()
    assert b"[kanban sandbox] bwrap argv:" in log


def _builder_in(host):
    real = kbs.build_sandbox

    def build(command, **kw):
        kw.update(hermes_root=host.root, home=host.home, bwrap="/usr/bin/bwrap")
        return real(command, **kw)
    return build


def test_unbuildable_sandbox_is_an_infrastructure_deferral(monkeypatch, tmp_path):
    kbd, _captured = _patch_spawn(monkeypatch, tmp_path, {"sandbox": "bwrap"})
    monkeypatch.setattr(kbs.shutil, "which", lambda _n: None)
    with pytest.raises(kbs.SandboxUnavailable):
        kbd._default_spawn(_task("any"), str(tmp_path))
