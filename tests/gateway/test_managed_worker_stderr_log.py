"""Managed-worker stderr (its tracebacks and redirected prints) lands in the owning profile's
private, redacted, size-capped ``logs/managed-worker.log`` instead of being discarded."""
import asyncio
import io
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef

SECRET = 'sk-proj-' + 'Q' * 40
CRASH = f"""import sys
sys.stderr.write('Traceback (most recent call last):\\n')
sys.stderr.write('RuntimeError: provider refused api_key={SECRET} Authorization: Bearer {SECRET}\\n')
sys.stderr.flush()
raise SystemExit(1)
"""


@pytest.mark.platforms("linux")
def test_worker_stderr_reaches_a_private_redacted_profile_log(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    original = subprocess.Popen

    def crashing_child(args, **kwargs):
        return original([sys.executable, '-c', CRASH], **kwargs)
    monkeypatch.setattr(managed.subprocess, 'Popen', crashing_child)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    authority = SimpleNamespace(profile_id=str(tmp_path), pending_results={}, waiters={},
                                adopt_agent=lambda session_id, generation, worker: None)
    row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 3, 'payload': {'text': 'go'}}
    with pytest.raises(EOFError):
        asyncio.run(asyncio.wait_for(managed.execute_managed(authority, SessionRef('p', 's'), row, None), 20))
    log = tmp_path / 'logs' / 'managed-worker.log'
    text = log.read_text()
    assert 'Traceback (most recent call last)' in text and 'RuntimeError: provider refused' in text
    assert SECRET not in text and 'QQQQQQQQQQ' not in text
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_worker_log_is_bounded_and_scrubs_the_workers_exact_secrets(tmp_path):
    from gateway.session_managed_worker_log import LOG_NAME, MAX_LOG_BYTES, drain_worker_stderr
    path = tmp_path / 'logs' / LOG_NAME
    assignment = 'opaque-assignment-secret-value'
    noise = (f'line {assignment} ' + 'x' * 200 + '\n') * 12_000 + 'y' * 300_000 + '\nlast\n'
    drain_worker_stderr(io.BytesIO(noise.encode()), path, 7, lambda: [assignment])
    files = [path.with_name(LOG_NAME + '.1'), path]  # oldest first
    assert all(f.stat().st_size <= MAX_LOG_BYTES for f in files)
    assert not list(path.parent.glob(LOG_NAME + '.2'))
    text = ''.join(f.read_text() for f in files)
    assert assignment not in text and '[truncated]' in text and text.endswith('[pid 7] last\n')



# --- a worker lost before hello names its cause on the board (t_02251ce3) -----------------
# The child dies in env bootstrap, before any frame: its stderr line is the only record of why,
# so the owner's worker_lost note and the crashed run's error must carry it.

from tests.hermes_cli import test_kanban_managed_worker_loss as loss_board

board = loss_board.board  # the isolated HERMES_HOME board fixture, shared
_owner_policy = loss_board._owner_policy

NO_ENV = 'hermes: no dependency environment is committed for this install; run `hermes pm repair`'
BWRAP_NOTE = b'kanban sandbox: bwrap --ro-bind /usr /usr --unshare-pid -- python -m agent.managed_worker\n'


def _lose_before_hello(board, monkeypatch, tid, script):
    """Run one Kanban card turn whose child runs ``script`` and dies before hello; the
    ``worker_lost`` payloads and the run id."""
    import json
    import os
    from gateway import session_managed_worker as managed
    original = subprocess.Popen

    def dying_child(args, **kwargs):
        return original([sys.executable, '-c', script], **kwargs)

    async def unsandboxed(authority, policy, command, env, cwd):
        return None, BWRAP_NOTE  # the owner's launch note, logged beside the worker's own lines

    async def unscoped(policy, argv, env):
        return argv, env
    monkeypatch.setattr(managed.subprocess, 'Popen', dying_child)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    monkeypatch.setattr(managed, '_kanban_sandbox', unsandboxed)
    monkeypatch.setattr(managed, '_kanban_turn_scope', unscoped)
    run_id = board.running(tid, claim_expires_in=900, heartbeat_age=30)
    authority = SimpleNamespace(profile_id=os.environ['HERMES_HOME'], pending_results={}, waiters={},
                                adopt_agent=lambda session_id, generation, worker: None)
    row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 3, 'payload': {'text': 'go'}}

    async def turn():
        with pytest.raises(EOFError):
            await managed.execute_managed(authority, SessionRef('p', 's'), row, _owner_policy(board, tid, run_id))
        await asyncio.gather(*list(getattr(authority, '_mutation_tasks', ())))
    asyncio.run(asyncio.wait_for(turn(), 20))
    lost = [json.loads(r[0]) for r in board.conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='worker_lost'", (tid,))]
    return lost, run_id


@pytest.mark.platforms("linux")
def test_pre_hello_death_carries_its_stderr_cause(board, monkeypatch):
    script = f"import sys\nsys.stderr.write({NO_ENV!r} + '\\n')\nraise SystemExit(1)\n"
    lost, run_id = _lose_before_hello(board, monkeypatch, 't_noenv', script)
    assert len(lost) == 1 and lost[0]['reason'] == 'EOFError'
    assert 'no dependency environment is committed' in lost[0]['detail']
    assert 'bwrap' not in lost[0]['detail']  # the owner's launch note is not the worker's output
    board.tick(failure_limit=5)
    run = board.run(run_id)
    assert run['outcome'] == 'crashed'
    assert 'managed worker lost: EOFError — stderr: hermes: no dependency environment' in run['error']


@pytest.mark.platforms("linux")
def test_a_pre_hello_cause_is_redacted(board, monkeypatch):
    lost, _ = _lose_before_hello(board, monkeypatch, 't_secret', CRASH)
    assert 'RuntimeError: provider refused' in lost[0]['detail']
    assert SECRET not in lost[0]['detail'] and 'QQQQQQQQQQ' not in lost[0]['detail']


@pytest.mark.platforms("linux")
def test_a_silent_pre_hello_death_keeps_the_old_note(board, monkeypatch):
    lost, run_id = _lose_before_hello(board, monkeypatch, 't_silent', 'raise SystemExit(1)\n')
    assert len(lost) == 1 and 'detail' not in lost[0]
    board.tick(failure_limit=5)
    assert '[managed worker lost: EOFError]' in board.run(run_id)['error']


def test_the_detail_keeps_the_last_lines_and_the_end_of_a_long_one(tmp_path):
    from collections import deque
    from gateway.session_managed_worker import STDERR_DETAIL_CHARS, ManagedWorker
    from gateway.session_managed_worker_log import LOG_NAME, drain_worker_stderr
    worker = ManagedWorker(SimpleNamespace(poll=lambda: 1))
    drain_worker_stderr(io.BytesIO(b'one\n\n  \ntwo\nthree\nfour\n'), tmp_path / LOG_NAME, 7, list, worker.stderr_tail)
    assert worker.stderr_detail(0) == 'two | three | four'  # blank lines never take a slot
    worker.stderr_tail = deque(maxlen=3)
    drain_worker_stderr(io.BytesIO(b'a\nb\n' + b'x' * 400 + b' the cause\n'), tmp_path / LOG_NAME, 7, list,
                        worker.stderr_tail)
    detail = worker.stderr_detail(0)
    assert len(detail) == STDERR_DETAIL_CHARS and detail.startswith('…') and detail.endswith('x the cause')
    assert 'a | b' not in detail
