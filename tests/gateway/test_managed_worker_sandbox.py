"""kanban.sandbox wraps the gateway's managed-worker spawn, where a card's agent now runs.

Slice C of t_dd989123 (intake t_4c0f0c53): since the upstream owner-run model the dispatcher's
thin client only dials ``gateway.sock`` and the agent runs in ``python -m agent.managed_worker``,
which the gateway spawns. A card in a sandboxed tenant must spawn inside the forwarder + bwrap
plan, with only its owner's control socket left reachable; every other spawn stays byte-identical.

Unit tests drive the real ``_spawn_registered_worker`` with a capturing Popen; the real-bwrap tests
(skipped where bwrap or unprivileged user namespaces are missing) run built plans for real. Their
fake hermes root lives outside /tmp: the sandbox mounts a private ``--tmpfs /tmp``.
"""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from gateway.session_contract import SessionRef
from hermes_cli import kanban_sandbox as kbs
from hermes_state_runtime import RuntimeStoreError

ROOT = Path(__file__).resolve().parents[2]
WORKER_ARGV = [sys.executable, '-m', 'agent.managed_worker']


def _bwrap_works():
    if not shutil.which('bwrap'):
        return False
    try:
        return subprocess.run(['bwrap', '--unshare-user', '--unshare-pid', '--ro-bind', '/', '/', 'true'],
                              capture_output=True, timeout=10, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bwrap = pytest.mark.skipif(not _bwrap_works(), reason='bwrap with unprivileged userns unavailable')


def _listen(path):
    """A listening AF_UNIX socket at ``path`` (bound relative to its dir: tmp paths can exceed sun_path)."""
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(path.name)
        sock.listen(4)
    finally:
        os.chdir(cwd)
    return sock


def _fake_host(base, *, env_file=True):
    home = base / 'home'
    root = home / '.hermes'
    for d in ('.ssh', 'run', '.cache', '.hermes/state', '.hermes/profiles/other', '.hermes/cache/scratch', '.hermes/logs'):
        (home / d).mkdir(parents=True, exist_ok=True)
    (home / '.ssh' / 'id_ed25519').write_text('PRIVATE')
    (root / '.anthropic_oauth.json').write_text('{"refresh": "OAUTH-SECRET"}')
    if env_file:
        (root / '.env').write_text('OPENROUTER_API_KEY=ok\nBINANCE_SECRET=leak\n')
    socks = {rel: _listen(root / rel) for rel in
             ('gateway.sock', 'profiles/other/gateway.sock', 'state/gateway.loop-tick.1.sock')}
    ws = base / 'ws'
    ws.mkdir()
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(home), 'XDG_RUNTIME_DIR': str(home / 'run'),
           'HERMES_HOME': str(root), 'BINANCE_SECRET': 'leak', 'OPENROUTER_API_KEY': 'leak',
           'HERMES_KANBAN_TASK': 'parent-task'}
    return SimpleNamespace(home=home, root=root, ws=ws, env=env, socks=socks)


def _under_tmp(path):
    resolved = path.resolve()
    return resolved == Path('/tmp') or Path('/tmp') in resolved.parents


@pytest.fixture
def host_dir(tmp_path):
    """A base dir for a real-bwrap fake host that does not resolve under /tmp."""
    if not _under_tmp(tmp_path):
        yield tmp_path
        return
    if not os.path.isdir('/var/tmp'):
        pytest.skip('tmp_path is under /tmp (masked by the sandbox) and /var/tmp is missing')
    base = Path(tempfile.mkdtemp(prefix='hermes-mw-sbx-', dir='/var/tmp'))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _pairs(argv, flag):
    return [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == flag]


def _context(host, tenant='pilot', skills=()):
    return {'task_id': 't_7', 'run_id': 11, 'tenant': tenant, 'skills': list(skills), 'workspace': str(host.ws),
            'db': str(host.root / 'kanban.db'), 'board': 'default', 'claim_lock': 'h:1'}


# --- spawn wiring (real _spawn_registered_worker, capturing Popen) ----------------------------

@pytest.fixture
def spawn(tmp_path, monkeypatch):
    from gateway import session_kanban
    from gateway import session_managed_worker as managed
    from gateway import session_runtime_workers
    from hermes_cli import config

    host = _fake_host(tmp_path)
    state = SimpleNamespace(host=host, calls=[], losses=[], watches=[], cfg={})
    original = subprocess.Popen

    def popen(args, **kw):
        state.calls.append((list(args), dict(kw)))
        kw.pop('pass_fds', None)
        return original([sys.executable, '-c', 'import sys; sys.stdin.read()'], **kw)

    def config_read():
        if isinstance(state.cfg, Exception):
            raise state.cfg
        return {'kanban': state.cfg}

    real_build = kbs.build_sandbox

    def build(command, **kw):
        kw.update(hermes_root=host.root, home=host.home)
        kw.setdefault('bwrap', state.bwrap)
        return real_build(command, **kw)

    state.bwrap = '/usr/bin/bwrap'
    monkeypatch.setattr(managed.subprocess, 'Popen', popen)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    monkeypatch.setattr(managed, '_owner_control_socket', lambda: host.root / 'gateway.sock')
    monkeypatch.setattr(config, 'load_config', config_read)
    monkeypatch.setattr(kbs, 'build_sandbox', build)
    monkeypatch.setattr(session_kanban, 'watch_submitter_from_owner',
                        lambda context, closed, kill: state.watches.append(context))
    monkeypatch.setattr(session_kanban, 'record_worker_loss',
                        lambda policy, reason, **kw: state.losses.append(reason))
    monkeypatch.setattr(session_runtime_workers, 'persist_work_count', lambda authority: None)
    for key, value in host.env.items():
        monkeypatch.setenv(key, value)

    def run(kanban_context):
        authority = SimpleNamespace(profile_id=str(host.root), pending_results={}, waiters={})
        policy = SimpleNamespace(kanban_json=json.dumps(kanban_context) if kanban_context is not None else None)
        row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 2, 'payload': {'text': 'go'}}

        async def go():
            try:
                return await managed._spawn_registered_worker(authority, SessionRef('p', 's'), row, policy)
            finally:
                for task in list(getattr(authority, '_mutation_tasks', ())):
                    await task
        spawned = asyncio.run(go())
        if spawned is not None:
            spawned[0].close()
        return spawned
    state.run = run
    state.log = lambda: (host.root / 'logs' / 'managed-worker.log').read_text() if (
        host.root / 'logs' / 'managed-worker.log').exists() else ''
    return state


@pytest.mark.platforms('linux')
def test_chat_and_cron_spawn_is_unchanged_and_never_reads_the_sandbox_config(spawn):
    spawn.cfg = RuntimeError('a chat turn must not read kanban.sandbox')
    spawn.run(None)
    (argv, kw), = spawn.calls
    assert argv == WORKER_ARGV and 'pass_fds' not in kw and kw['env'] is None
    assert '[kanban sandbox]' not in spawn.log()


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('cfg,tenant', [
    ({'sandbox': 'off', 'sandbox_tenants': ['pilot']}, 'pilot'),
    ({'sandbox': False}, 'pilot'),
    ({'sandbox': 'bwrap', 'sandbox_tenants': ['pilot']}, 'movis'),
])
def test_unsandboxed_card_spawns_the_plain_worker(spawn, cfg, tenant):
    spawn.cfg = cfg
    spawn.run(_context(spawn.host, tenant=tenant))
    (argv, kw), = spawn.calls
    assert argv == WORKER_ARGV and 'pass_fds' not in kw and kw['env'] is None
    assert spawn.watches == [] and '[kanban sandbox]' not in spawn.log()


@pytest.mark.platforms('linux')
def test_exempt_skill_spawns_the_plain_worker_and_logs_the_exemption(spawn):
    spawn.cfg = {'sandbox': 'bwrap', 'sandbox_tenants': ['pilot'], 'sandbox_exempt_skills': ['hindsight-ops']}
    spawn.bwrap = None  # an exempt card never needs bwrap
    spawn.run(_context(spawn.host, skills=['x', 'hindsight-ops']))
    (argv, kw), = spawn.calls
    assert argv == WORKER_ARGV and 'pass_fds' not in kw
    assert '[kanban sandbox] exempt: skill hindsight-ops' in spawn.log()
    assert 'bwrap argv' not in spawn.log() and spawn.watches == []


@pytest.mark.platforms('linux')
def test_sandboxed_tenant_spawns_the_worker_inside_bwrap(spawn):
    spawn.cfg = {'sandbox': 'bwrap', 'sandbox_tenants': ['pilot']}
    host = spawn.host
    spawn.run(_context(host))
    (argv, kw), = spawn.calls
    assert argv[:3] == [sys.executable, '-P', str(Path(kbs.__file__).resolve())]
    assert argv[3:5] == ['exec', '/usr/bin/bwrap'] and argv[-3:] == WORKER_ARGV
    assert '--unshare-pid' in argv and '--die-with-parent' in argv
    assert ('--chdir', str(host.ws)) in [(argv[i], argv[i + 1]) for i in range(len(argv) - 1)]
    setenv = dict(_pairs(argv, '--setenv'))
    # --chdir moves the cwd off the tree: `-m agent.managed_worker` needs the tree on PYTHONPATH.
    assert str(ROOT) in setenv['PYTHONPATH'].split(os.pathsep)
    # The card binds its Kanban context from the bootstrap; a stale spawn-env pin never crosses.
    assert not [k for k in setenv if k.startswith('HERMES_KANBAN_')]
    secret_named = [k for k in (*setenv, *kw['env']) if kbs._SECRET_NAME_RE.search(k)]
    assert secret_named == [] and 'leak' not in ' '.join(argv)
    assert 'BINANCE_SECRET' not in kw['env'] and kw['env']['HERMES_SANDBOX'] == 'bwrap'
    assert kw['pass_fds'] and kw['start_new_session'] is True  # the filtered .env memfd rides along
    assert kw['stdin'] == subprocess.PIPE and kw['stdout'] == subprocess.PIPE
    assert '[kanban sandbox] bwrap argv:' in spawn.log()
    assert [c['task_id'] for c in spawn.watches] == ['t_7']  # the owner watches the submitter


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('breakage', ['no-bwrap', 'config', 'workspace'])
def test_unbuildable_sandbox_fails_the_admission_closed(spawn, breakage, monkeypatch):
    spawn.cfg = {'sandbox': 'bwrap', 'sandbox_tenants': ['pilot']}
    context = _context(spawn.host)
    if breakage == 'no-bwrap':
        spawn.bwrap = None
        monkeypatch.setattr(kbs.shutil, 'which', lambda name: None)
    elif breakage == 'config':
        from hermes_cli.config_backend import ConfigBackendUnavailable
        spawn.cfg = ConfigBackendUnavailable('config.yaml unreadable')
    else:
        context['workspace'] = None
    with pytest.raises(RuntimeStoreError) as raised:
        spawn.run(context)
    assert raised.value.reason == 'sandbox_unavailable'
    assert spawn.calls == []  # never a bare spawn
    assert spawn.losses == ['sandbox_unavailable']  # the card's run learns why


# --- mask selection ---------------------------------------------------------------------------

def test_only_the_owner_control_socket_is_unmasked(tmp_path):
    host = _fake_host(tmp_path)
    plan = kbs.build_sandbox(WORKER_ARGV, worker_env=host.env, workspace=host.ws, run_key='t_7-r11',
                             hermes_root=host.root, home=host.home, bwrap='/usr/bin/bwrap',
                             allow_control_socket=host.root / 'gateway.sock')
    try:
        devnull = [dst for src, dst in _pairs(plan.argv, '--ro-bind') if src == '/dev/null']
        assert str(host.root / 'gateway.sock') not in devnull
        # Must-not-unmask: another profile's control socket and every loop-tick socket.
        assert str(host.root / 'profiles/other/gateway.sock') in devnull
        assert str(host.root / 'state/gateway.loop-tick.1.sock') in devnull
        assert str(host.root / '.anthropic_oauth.json') in devnull
        binds = [src for src, _dst in _pairs(plan.argv, '--bind')]
        assert not [b for b in binds if b.startswith(str(host.home / 'run'))]  # no fallback dir bind
    finally:
        plan.close()


def test_pointer_fallback_binds_the_socket_dir_after_the_masks(tmp_path):
    host = _fake_host(tmp_path)
    gw_dir = host.home / 'run' / 'hermes-gw-0123456789abcdef'
    gw_dir.mkdir()
    sock = _listen(gw_dir / 'control.sock')
    try:
        plan = kbs.build_sandbox(WORKER_ARGV, worker_env=host.env, workspace=host.ws, run_key='r',
                                 hermes_root=host.root, home=host.home, bwrap='/usr/bin/bwrap',
                                 allow_control_socket=gw_dir / 'control.sock')
        try:
            argv = plan.argv
            assert (str(gw_dir), str(gw_dir)) in _pairs(argv, '--bind')
            # The bind must come after the $XDG_RUNTIME_DIR tmpfs it punches through.
            runtime_mask = next(i for i, a in enumerate(argv) if a == '--tmpfs' and argv[i + 1] == str(host.home / 'run'))
            dir_bind = next(i for i, a in enumerate(argv) if a == '--bind' and argv[i + 1] == str(gw_dir))
            assert dir_bind > runtime_mask
            devnull = [dst for src, dst in _pairs(argv, '--ro-bind') if src == '/dev/null']
            assert str(host.root / 'gateway.sock') in devnull  # the stale direct socket stays masked
        finally:
            plan.close()
    finally:
        sock.close()


# --- owner-side submitter watch and namespace-aware pids ---------------------------------------

def _runs_db(path, pid, started_at):
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute('CREATE TABLE task_runs (id INTEGER PRIMARY KEY, worker_pid INTEGER, worker_started_at TEXT)')
        conn.execute('INSERT INTO task_runs VALUES (11, ?, ?)', (pid, started_at))
        conn.commit()


def test_owner_watch_kills_the_worker_when_the_submitter_dies(tmp_path):
    from gateway.session_kanban import watch_submitter_from_owner
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait(timeout=10)
    _runs_db(tmp_path / 'kanban.db', dead.pid, None)
    killed, closed = threading.Event(), threading.Event()
    thread = watch_submitter_from_owner({'db': str(tmp_path / 'kanban.db'), 'run_id': 11}, closed, killed.set)
    assert killed.wait(10), 'a dead submitter must end the sandboxed worker'
    thread.join(5)


def test_owner_watch_leaves_a_live_submitter_alone_and_stops_on_close(tmp_path):
    from gateway.session_kanban import watch_submitter_from_owner
    _runs_db(tmp_path / 'kanban.db', os.getpid(), None)
    killed, closed = threading.Event(), threading.Event()
    thread = watch_submitter_from_owner({'db': str(tmp_path / 'kanban.db'), 'run_id': 11}, closed, killed.set)
    assert not killed.wait(2.5)
    closed.set()
    thread.join(5)
    assert not thread.is_alive() and not killed.is_set()


def test_namespaced_worker_reports_its_host_pid_and_skips_the_blind_watch(tmp_path, monkeypatch):
    from agent.runtime_session_store import bind_owner_view_pid
    from gateway import session_kanban
    db = tmp_path / 'kanban.db'
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute('CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, run_id INTEGER, kind TEXT, payload TEXT)')
        conn.execute("INSERT INTO task_events(task_id, run_id, kind, payload) VALUES ('t_7', 11, 'worker_bound', ?)",
                     (json.dumps({'pid': 4242424, 'claim_lock': 'h:1'}),))
        conn.commit()
    context = {'task_id': 't_7', 'run_id': 11, 'claim_lock': 'h:1'}
    monkeypatch.setattr(session_kanban.threading, 'Thread',
                        lambda *a, **k: pytest.fail('a namespaced worker cannot see the host submitter'))
    bind_owner_view_pid(4242424)
    try:
        with contextlib.closing(sqlite3.connect(db)) as conn:
            assert session_kanban._bound_worker_matches(conn, context)
        session_kanban._watch_submitter(db, 11)
    finally:
        bind_owner_view_pid(None)
    with contextlib.closing(sqlite3.connect(db)) as conn:
        assert not session_kanban._bound_worker_matches(conn, context)


# --- real bwrap -------------------------------------------------------------------------------

_PROBE = textwrap.dedent('''
    import json, os, socket, stat, sys
    def conn(p):
        # Relative to the socket's dir: a deep test tmp path can exceed sun_path (108 bytes).
        s, cwd = socket.socket(socket.AF_UNIX), os.getcwd()
        try:
            os.chdir(os.path.dirname(p)); s.connect(os.path.basename(p)); return 'CONNECTED'
        except OSError as e:
            return type(e).__name__
        finally:
            os.chdir(cwd); s.close()
    def rd(p):
        try:
            return open(p, 'rb').read().decode()
        except OSError as e:
            return type(e).__name__
    R = sys.argv[1]
    line = sys.stdin.readline()
    out = {'echo': json.loads(line), 'own_is_sock': stat.S_ISSOCK(os.stat(R + '/gateway.sock').st_mode),
           'own': conn(R + '/gateway.sock'), 'other': conn(R + '/profiles/other/gateway.sock'),
           'tick': conn(R + '/state/gateway.loop-tick.1.sock'),
           'oauth': rd(R + '/.anthropic_oauth.json'), 'pid': os.getpid(),
           'secret': os.environ.get('BINANCE_SECRET')}
    sys.stdout.write(json.dumps(out) + '\\n'); sys.stdout.flush()
''')


@needs_bwrap
@pytest.mark.platforms('linux')
def test_built_plan_reaches_only_its_owner_socket_and_relays_frames(host_dir):
    host = _fake_host(host_dir)
    plan = kbs.build_sandbox([sys.executable, '-c', _PROBE, str(host.root)], worker_env=host.env,
                             workspace=host.ws, run_key='t_7-r11', hermes_root=host.root, home=host.home,
                             assignee_home=host.root, allow_control_socket=host.root / 'gateway.sock')
    try:
        proc = subprocess.run(plan.argv, input=json.dumps({'type': 'hello', 'pid': 2}) + '\n', capture_output=True,
                              text=True, timeout=60, env=plan.outer_env, pass_fds=plan.pass_fds, check=False)
    finally:
        plan.close()
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen['echo'] == {'type': 'hello', 'pid': 2}  # stdin/stdout frames pass through bwrap
    assert seen['own_is_sock'] is True and seen['own'] == 'CONNECTED'
    assert seen['other'] == 'ConnectionRefusedError' and seen['tick'] == 'ConnectionRefusedError'  # /dev/null binds
    assert seen['oauth'] in ('', 'PermissionError') and seen['secret'] is None
    assert seen['pid'] == 2  # its own pid namespace: the owner maps it (slice A)


def _descendants(pid):
    try:
        return [p.pid for p in psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        return []


def _gone(pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(.1)  # health: allow HX013 -- polling real process exit, no event to wait on
    return False


_TOOL_CHILD = textwrap.dedent('''
    import subprocess, sys, time
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    print('ready', flush=True)
    time.sleep(120)
''')


@needs_bwrap
@pytest.mark.platforms('linux')
@pytest.mark.live_system_guard_bypass  # cleanup may kill our own descendants
def test_owner_close_reaps_a_sandboxed_worker_and_its_tool_children(host_dir):
    from gateway.session_managed_worker import WORKER_POPEN, ManagedWorker
    host = _fake_host(host_dir, env_file=False)
    plan = kbs.build_sandbox([sys.executable, '-c', _TOOL_CHILD], worker_env=host.env, workspace=host.ws,
                             run_key='t_8-r1', hermes_root=host.root, home=host.home)
    process = subprocess.Popen(plan.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               env=plan.outer_env, **WORKER_POPEN)
    plan.close()
    pids = []
    try:
        assert process.stdout.readline() == b'ready\n'
        pids = _descendants(process.pid)
        assert len(pids) >= 4  # bwrap monitor, namespace init, worker, its tool child
        ManagedWorker(process).close()
        assert process.poll() is not None
        assert all(_gone(pid, 10) for pid in pids), 'a sandboxed process outlived the owner close()'
    finally:
        for pid in pids:
            if not _gone(pid, 0):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, 9)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


_ORPHANED = textwrap.dedent('''
    import subprocess, sys, time
    import agent.managed_worker as mw
    mw.ORPHAN_GRACE_SECONDS = 1
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    controls = mw.WorkerControls(channel=None, route='r')
    controls.agent = type('Stubborn', (), {'interrupt': lambda self, *a, **k: None})()
    print('ready', flush=True)
    while True:
        time.sleep(1)
''')

_OWNER = textwrap.dedent('''
    import json, subprocess, sys
    from gateway.session_managed_worker import WORKER_POPEN
    spec = json.loads(sys.argv[1])
    worker = subprocess.Popen(spec['argv'], env=spec['env'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, **WORKER_POPEN)
    print(worker.pid, worker.stdout.readline().decode().strip(), flush=True)
    worker.wait()
''')


@needs_bwrap
@pytest.mark.platforms('linux')
@pytest.mark.live_system_guard_bypass  # cleanup may signal our own descendants after init adopted them
def test_sandboxed_worker_and_its_tool_children_die_with_the_owner(host_dir):
    host = _fake_host(host_dir, env_file=False)
    env = dict(host.env, PYTHONPATH=str(ROOT))
    plan = kbs.build_sandbox([sys.executable, '-c', _ORPHANED], worker_env=env, workspace=host.ws,
                             run_key='t_9-r1', hermes_root=host.root, home=host.home)
    assert plan.pass_fds == ()  # no .env on the fake host: nothing to hand through the owner
    owner = subprocess.Popen([sys.executable, '-c', _OWNER, json.dumps({'argv': plan.argv, 'env': plan.outer_env})],
                             cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    pids = []
    try:
        forwarder, ready = owner.stdout.readline().split()
        assert ready == b'ready'
        pids = [int(forwarder), *_descendants(int(forwarder))]
        assert len(pids) >= 5  # forwarder, bwrap monitor, namespace init, worker, its tool child
        owner.kill()
        owner.wait(timeout=5)
        assert all(_gone(pid, 20) for pid in pids), 'a sandboxed process outlived its killed owner'
    finally:
        owner.stdout.close()
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        for pid in pids:
            if not _gone(pid, 0):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, 9)
