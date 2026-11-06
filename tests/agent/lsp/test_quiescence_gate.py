"""rust-analyzer in a cold workspace: no verdict until the server reports quiescent.

rust-analyzer answers diagnostics with ``[]`` (and may push ``[]``) while it is still loading the
workspace (build scripts, proc macros, indexing), then reports ``experimental/serverStatus``
quiescent=true and the real diagnostics.  Taking that first answer as the verdict reported a cold
workspace's first edit as clean.  The contract: nothing published or answered mid-load satisfies
``wait_for_diagnostics``; once the server is quiescent the real diagnostics do.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.servers import SERVERS

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _loading_client(workspace: Path, gate: Path, *, status_early: bool, await_quiescence: bool) -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": "loading", "MOCK_LSP_GATE": str(gate),
           "MOCK_LSP_STATUS_EARLY": "1" if status_early else "0", "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    return LSPClient(server_id="mock-loading", workspace_root=str(workspace), command=[sys.executable, MOCK_SERVER],
                     env=env, cwd=str(workspace), await_quiescence=await_quiescence)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_early,await_quiescence", [
    (True, False),   # the server says it is loading before it answers (generic serverStatus path)
    (False, True),   # it answers before any status report: the client knows it reports quiescence
], ids=["status-first", "answer-first"])
async def test_empty_answer_mid_load_is_not_a_verdict(tmp_path: Path, status_early: bool, await_quiescence: bool):
    f = tmp_path / "math.rs"
    f.write_text("pub fn probe() -> u32 { \"planted\" }\n")
    gate = tmp_path / "loaded"
    client = _loading_client(tmp_path, gate, status_early=status_early, await_quiescence=await_quiescence)
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="rust")
        await client.save_file(str(f))
        # Still loading: the empty push / empty pull are no verdict, so the wait runs out its budget.
        assert await client.wait_for_diagnostics(str(f), version, timeout=1.0) is False
        assert client.diagnostics_for(str(f), fresh_only=True) == []

        gate.touch()  # workspace loaded: quiescent, then the real diagnostics
        assert await client.wait_for_diagnostics(str(f), version, timeout=30.0) is True
        diags = client.diagnostics_for(str(f), fresh_only=True)
        assert [d["code"] for d in diags] == ["MOCK001"]
    finally:
        await client.shutdown()


def test_rust_analyzer_awaits_quiescence():
    """The registry flags rust-analyzer, so its client holds verdicts from spawn on."""
    assert next(s for s in SERVERS if s.server_id == "rust-analyzer").await_quiescence
