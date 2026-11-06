"""rust-analyzer's flycheck: rustc-only errors (borrow check, lifetimes) reach the verdict.

rust-analyzer answers the native diagnostics pull in under a second, but rustc-only errors come from
flycheck (``cargo check`` on didSave), pushed seconds later under ``$/progress`` tokens
``rust-analyzer/flycheck/<n>``.  Taking the native pull as the whole verdict reported a borrow error
as "clean".  The contract: a verdict on a saved version waits for the check that save started
(``CHECK_START_GRACE`` for it to begin, ``CHECK_WAIT`` cap); check diagnostics count only when they
describe the synced text (saved, still equal to the disk), and a rustc duplicate of a native
diagnostic is dropped.  The turn end waits for such a held verdict past ``TURN_END_WAIT``.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.lsp import client as lsp_client
from agent.lsp import deferred
from agent.lsp.client import CHECK_START_GRACE, LSPClient
from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec, check_prefix_for

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")
PREFIX = "rust-analyzer/flycheck/"
CLEAN = "fn f() {\n    ok;\n}\n"
BORROW = "fn f() {\n    borrow_bad;\n}\n"  # cargo check: E0502 at 1:4; rust-analyzer itself: clean
NATIVE_AND_BORROW = "native_bad\n    borrow_bad;\n"  # E0308 from both producers at 0:0, E0502 from rustc


def _codes(diags):
    return sorted((d["source"], d["code"]) for d in diags)


def _fly_client(workspace: Path, monkeypatch, *, prefix: str = PREFIX, **env: str) -> LSPClient:
    for key in ("MOCK_LSP_CHECK_BEGIN", "MOCK_LSP_CHECK_END", "MOCK_LSP_CHECK_DELAY", "MOCK_LSP_CHECK_TOKEN",
                "MOCK_LSP_CHECK_START_DELAY"):
        monkeypatch.delenv(key, raising=False)
    full_env = {"MOCK_LSP_SCRIPT": "flycheck", "PYTHONPATH": os.environ.get("PYTHONPATH", ""), **env}
    return LSPClient(server_id="mock-ra", workspace_root=str(workspace), command=[sys.executable, MOCK_SERVER],
                     env=full_env, cwd=str(workspace), check_progress_prefix=prefix)


async def _saved_verdict(client: LSPClient, path: Path, *, save: bool = True) -> tuple[bool, list, float]:
    version = await client.open_file(str(path), language_id="rust")
    started = time.monotonic()
    if save:
        await client.save_file(str(path))
    fresh = await client.wait_for_diagnostics(str(path), version, timeout=10.0)
    return fresh, client.diagnostics_for(str(path), fresh_only=True), time.monotonic() - started


@pytest.mark.asyncio
async def test_rustc_only_error_after_clean_native_pull_is_in_the_verdict(tmp_path, monkeypatch):
    """(a) The native pull is clean; cargo check pushes E0502 1s later: that is the verdict."""
    f = tmp_path / "lib.rs"
    f.write_text(BORROW)
    client = _fly_client(tmp_path, monkeypatch, MOCK_LSP_CHECK_DELAY="1.0")
    await client.start()
    try:
        fresh, diags, _ = await _saved_verdict(client, f)
        assert fresh is True
        assert _codes(diags) == [("rustc", "E0502")]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_check_that_never_begins_returns_native_after_the_grace(tmp_path, monkeypatch):
    """(b) No check begins: the native verdict comes back about CHECK_START_GRACE after the save."""
    f = tmp_path / "lib.rs"
    f.write_text(NATIVE_AND_BORROW)
    client = _fly_client(tmp_path, monkeypatch, MOCK_LSP_CHECK_BEGIN="0")
    await client.start()
    try:
        fresh, diags, elapsed = await _saved_verdict(client, f)
        assert fresh is True
        assert _codes(diags) == [("rust-analyzer", "E0308")]
        assert CHECK_START_GRACE * 0.9 <= elapsed < CHECK_START_GRACE + 6.0, elapsed
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_check_that_never_ends_keeps_the_native_verdict_at_the_cap(tmp_path, monkeypatch):
    """(c) A check that never ends: the cap returns True with the native data, never "no verdict"."""
    monkeypatch.setattr(lsp_client, "CHECK_WAIT", 2.0)
    f = tmp_path / "lib.rs"
    f.write_text(NATIVE_AND_BORROW)
    client = _fly_client(tmp_path, monkeypatch, MOCK_LSP_CHECK_END="0", MOCK_LSP_CHECK_DELAY="60")
    await client.start()
    try:
        fresh, diags, elapsed = await _saved_verdict(client, f)
        assert fresh is True
        assert _codes(diags) == [("rust-analyzer", "E0308")]
        assert elapsed < 30.0, elapsed
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_unsaved_sync_does_not_wait_for_a_check(tmp_path, monkeypatch):
    """(d) A snapshot/baseline sync (no didSave) never waits: there is no check of that text."""
    monkeypatch.setattr(lsp_client, "CHECK_START_GRACE", 30.0)
    f = tmp_path / "lib.rs"
    f.write_text(NATIVE_AND_BORROW)
    client = _fly_client(tmp_path, monkeypatch)
    await client.start()
    try:
        fresh, diags, elapsed = await _saved_verdict(client, f, save=False)
        assert fresh is True
        assert _codes(diags) == [("rust-analyzer", "E0308")]
        assert elapsed < 15.0, elapsed
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_server_without_a_check_prefix_is_unchanged(tmp_path, monkeypatch):
    """(e) Without a prefix (pyright, every server but rust-analyzer) the native verdict returns at once:
    the check's push, 6s later, is not waited for."""
    monkeypatch.setattr(lsp_client, "CHECK_START_GRACE", 30.0)
    f = tmp_path / "lib.rs"
    f.write_text(BORROW)
    client = _fly_client(tmp_path, monkeypatch, prefix="", MOCK_LSP_CHECK_DELAY="6")
    await client.start()
    try:
        fresh, diags, elapsed = await _saved_verdict(client, f)
        assert fresh is True
        assert diags == []
        assert elapsed < 5.0, elapsed
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_rustc_duplicate_of_a_native_diagnostic_is_dropped(tmp_path, monkeypatch):
    """(f) rustc's E0308 at the native E0308's start is dropped; its E0502 elsewhere is kept."""
    f = tmp_path / "lib.rs"
    f.write_text(NATIVE_AND_BORROW)
    client = _fly_client(tmp_path, monkeypatch, MOCK_LSP_CHECK_DELAY="0.5")
    await client.start()
    try:
        fresh, diags, _ = await _saved_verdict(client, f)
        assert fresh is True
        assert _codes(diags) == [("rust-analyzer", "E0308"), ("rustc", "E0502")]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_progress_without_the_prefix_is_not_a_check(tmp_path, monkeypatch):
    """(h) ``rustAnalyzer/Indexing`` is not the check: the verdict does not wait for its end (6s)."""
    f = tmp_path / "lib.rs"
    f.write_text(BORROW)
    client = _fly_client(tmp_path, monkeypatch, MOCK_LSP_CHECK_TOKEN="rustAnalyzer/Indexing", MOCK_LSP_CHECK_DELAY="6")
    await client.start()
    try:
        fresh, diags, elapsed = await _saved_verdict(client, f)
        assert fresh is True
        assert diags == []
        assert elapsed < 5.5, elapsed
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_check_queued_behind_a_running_one_is_waited_for(tmp_path, monkeypatch):
    """A check already running at the save outlasts the grace, and the save's own check begins only after
    it ends: a running check never times the grace out, and the grace restarts at its end."""
    f = tmp_path / "lib.rs"
    f.write_text(BORROW)
    client = _fly_client(tmp_path, monkeypatch, MOCK_LSP_OPEN_CHECK=str(CHECK_START_GRACE + 1.5),
                         MOCK_LSP_CHECK_START_DELAY=str(CHECK_START_GRACE / 2), MOCK_LSP_CHECK_DELAY="0.5")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="rust")
        # The unsaved native verdict is answered after the didOpen check's begin (one ordered stream).
        assert await client.wait_for_diagnostics(str(f), version, timeout=10.0) is True
        assert await client.wait_progress_idle(timeout=0.1) is False  # the didOpen check is running
        await client.save_file(str(f))
        assert await client.wait_for_diagnostics(str(f), version, timeout=10.0) is True
        assert _codes(client.diagnostics_for(str(f), fresh_only=True)) == [("rustc", "E0502")]
    finally:
        await client.shutdown()


def test_check_prefix_registry_and_check_on_save_off():
    """rust-analyzer carries the flycheck prefix; a spawn with ``checkOnSave`` off expects no check."""
    ra = next(s for s in SERVERS if s.server_id == "rust-analyzer")
    assert check_prefix_for(ra, {}) == PREFIX
    assert check_prefix_for(ra, {"checkOnSave": False}) == ""
    assert check_prefix_for(ra, {"checkOnSave": {"enable": False}}) == ""
    assert all(not s.check_progress_prefix for s in SERVERS if s.server_id != "rust-analyzer")


# ---- the service: edit check, non-edit callers, turn end ----


@pytest.fixture
def rust_service(monkeypatch, tmp_path):
    """An LSPService whose rust-analyzer is the flycheck mock (check prefix set, as registered)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "Cargo.toml").write_text("[package]\nname = \"probe\"\n")
    monkeypatch.chdir(str(repo))
    for key in ("MOCK_LSP_CHECK_BEGIN", "MOCK_LSP_CHECK_END", "MOCK_LSP_CHECK_TOKEN", "MOCK_LSP_CHECK_START_DELAY"):
        monkeypatch.delenv(key, raising=False)
    index = next(i for i, s in enumerate(SERVERS) if s.server_id == "rust-analyzer")
    original = SERVERS[index]

    def spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(command=[sys.executable, MOCK_SERVER], workspace_root=root, cwd=root,
                         env={"MOCK_LSP_SCRIPT": "flycheck"})

    SERVERS[index] = ServerDef(server_id="rust-analyzer", extensions=(".rs",), resolve_root=lambda fp, ws: ws,
                               build_spawn=spawn, description="mock rust-analyzer",
                               check_progress_prefix=original.check_progress_prefix)
    services = []

    def make(wait_timeout: float = 10.0, check_delay: float = 1.0) -> tuple[LSPService, Path]:
        monkeypatch.setenv("MOCK_LSP_CHECK_DELAY", str(check_delay))
        svc = LSPService(enabled=True, wait_mode="document", wait_timeout=wait_timeout, install_strategy="manual",
                         trusted_workspaces=[str(repo)])
        services.append(svc)
        return svc, repo
    try:
        yield make
    finally:
        for svc in services:
            svc.shutdown()
        SERVERS[index] = original


def _edit(svc: LSPService, path: Path, pre: str | None, post: str, owner: str | None = None):
    path.write_text(post)
    future = svc.start_edit_check(str(path), pre_text=pre, post_text=post, owner=owner)
    assert future is not None
    return future.result(timeout=60)


def test_edit_check_reports_the_borrow_error_and_not_a_recalled_preexisting_one(rust_service):
    """(a) through the edit check: the introduced E0502 is reported; (g) the next edit, which keeps it,
    does not report it again (the recalled verdict includes the rustc diagnostics)."""
    svc, repo = rust_service()
    f = repo / "lib.rs"
    f.write_text(CLEAN)
    first = _edit(svc, f, CLEAN, BORROW)
    assert _codes(first.diagnostics) == [("rustc", "E0502")]
    second = _edit(svc, f, BORROW, BORROW + "// note\n")
    assert second.diagnostics == []


def test_edit_check_baseline_falls_back_to_the_pre_save_check(rust_service):
    """(g) Without a recalled verdict the unsaved baseline sync has no rustc data: the check pushed
    before this edit's save (the pre-edit disk) is the rustc baseline."""
    svc, repo = rust_service()
    f = repo / "lib.rs"
    f.write_text(BORROW)
    assert _codes(svc.get_diagnostics_sync(str(f), delta=False)) == [("rustc", "E0502")]
    result = _edit(svc, f, BORROW, BORROW + "// note\n")
    assert result.diagnostics == []


def test_pre_save_check_of_other_bytes_is_not_the_baseline(rust_service):
    """The last check pushed before the save read different bytes than the edit's baseline (a write
    landed mid-check): its errors may be this edit's own, so they are no baseline."""
    svc, repo = rust_service()
    f = repo / "lib.rs"
    planted = CLEAN + "    borrow_bad;\n"
    f.write_text(planted)  # the edit's bytes, already on disk while the previous check ran
    assert _codes(svc.get_diagnostics_sync(str(f), delta=False)) == [("rustc", "E0502")]
    result = _edit(svc, f, CLEAN, planted)
    assert _codes(result.diagnostics) == [("rustc", "E0502")]


def test_check_of_another_writers_disk_is_not_this_edits_verdict(rust_service):
    """Cargo check reads the disk: when another writer changed the file, its errors are not this edit's."""
    svc, repo = rust_service()
    f = repo / "lib.rs"
    f.write_text(CLEAN)
    edited = CLEAN + "// mine\n"
    future = svc.start_edit_check(str(f), pre_text=CLEAN, post_text=edited)
    assert future is not None
    f.write_text(BORROW)  # another session's write lands before the check reads the disk
    assert future.result(timeout=60).diagnostics == []


def test_non_edit_caller_waits_out_a_slow_check_without_breaking_the_server(rust_service):
    """``get_diagnostics_sync`` inherits the hold: a check slower than the outer join's old budget
    (wait_timeout + 2s) still lands its rustc error and leaves rust-analyzer usable."""
    svc, repo = rust_service(wait_timeout=1.0, check_delay=4.0)
    f = repo / "lib.rs"
    f.write_text(BORROW)
    assert _codes(svc.get_diagnostics_sync(str(f), delta=False)) == [("rustc", "E0502")]
    assert svc.get_status()["broken"] == []


def test_turn_end_waits_for_a_held_verdict_past_turn_end_wait(rust_service, monkeypatch):
    """The last edit of a turn: the turn end waits for the cargo-check verdict past ``TURN_END_WAIT``
    (a one-shot run has no next turn to deliver it to)."""
    from model_tools import handle_function_call

    svc, repo = rust_service(check_delay=2.0)
    monkeypatch.setattr("agent.lsp.get_service", lambda: svc)
    f = repo / "lib.rs"
    f.write_text(CLEAN)
    session = "S-fly"
    handle_function_call("read_file", {"path": str(f)}, task_id="t-fly", session_id=session)
    out = json.loads(handle_function_call("patch", {"path": str(f), "old_string": "ok;", "new_string": "borrow_bad;"},
                                          task_id="t-fly", session_id=session))
    assert out["success"] is True and "lsp_diagnostics" not in out, out
    notes = deferred.settle_for_agent(SimpleNamespace(session_id=session, _current_task_id=None), timeout=0.2)
    assert [path for path, _ in notes] == [str(f)]
    assert "E0502" in notes[0][1]


def test_settle_ends_when_the_held_check_settles():
    """A hold extends the turn-end wait only while the check runs: it ends once the check settles."""
    owner = "S-hold"
    token = deferred.track_pending(owner)
    deferred.hold_until(owner, time.monotonic() + 30.0)

    def finish():
        deferred.post(owner, "/x.rs", "late verdict", None)
        deferred.settled(owner, token)
    threading.Timer(1.0, finish).start()
    started = time.monotonic()
    notes = deferred.settle_for_agent(SimpleNamespace(session_id=owner, _current_task_id=None), timeout=0.1)
    assert notes == [("/x.rs", "late verdict")]
    assert time.monotonic() - started < 20.0
