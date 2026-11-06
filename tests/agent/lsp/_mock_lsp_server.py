#!/usr/bin/env python3
"""A minimal in-process LSP server used by tests.

Speaks just enough LSP to drive :class:`agent.lsp.client.LSPClient`
through a full lifecycle: ``initialize``, ``initialized``,
``textDocument/didOpen``, ``textDocument/didChange``, then a
``textDocument/publishDiagnostics`` notification followed by
``shutdown`` + ``exit``.

Behaviour (all behaviours selectable via env var ``MOCK_LSP_SCRIPT``):

- ``"clean"`` — initialize, accept didOpen/didChange, push empty
  diagnostics on every open/change, exit cleanly on shutdown.
- ``"errors"`` — same as ``clean`` but the published diagnostics
  carry one severity-1 entry pointing at line 0:0.
- ``"crash"`` — exit immediately after responding to ``initialize``
  (simulates a crashing server).
- ``"oom_abort"`` — prints a V8-style out-of-memory trace to stderr and
  aborts (SIGABRT) before answering ``initialize`` — models a Node
  language-server whose heap ceiling is too small for the workspace.
- ``"slow"`` — same as ``clean`` but sleeps 1s before responding to
  ``initialize`` (lets us test timeout behaviour).
- ``"slow_tree"`` — like ``slow``, with a child that ignores SIGTERM and
  a launcher that exits on SIGTERM (tests hard process-tree cleanup).
- ``"stale"`` — pushes one error on ``didOpen``, then goes SILENT on
  ``didChange`` (no push) and rejects the pull endpoint with
  method-not-found.  Models a slow tsserver that hasn't re-checked
  the edited content yet — the ghost-diagnostics scenario.
- ``"slow_push"`` — like ``stale`` on didOpen (one error) but on
  ``didChange`` sleeps ``MOCK_LSP_PUSH_DELAY`` seconds (default 1.0)
  and then pushes EMPTY diagnostics.  Models a server that fixes
  the ghost if you actually wait for it.  Pull endpoint rejects.
- ``"versionless"`` — errors on ``didOpen``, clean on ``didChange``, and
  no ``version`` field in any publishDiagnostics (the client credits
  each push with its current document version at receipt).  Push-only:
  the pull endpoint rejects.
- ``"by_content"`` — push-only; pushes one error iff the synced text contains ``bad``, else empty
  diagnostics.  Holds every push until the file named by ``MOCK_LSP_GATE`` exists (a server slower
  than the write, released by the test), else after ``MOCK_LSP_PUSH_DELAY`` (default 0.4s).
- ``"incremental"`` — apply ranged edits as UTF-16 and expose the server's
  document mirror through hover, for synchronization contract tests.
- ``"nav"`` — answers definition / references / hover / workspace symbol / document symbol with
  canned locations in ``MOCK_LSP_NAV_TARGET`` (see ``_NAV_ANSWERS``); logs each request to
  ``MOCK_LSP_REQUESTS_LOG``; ``MOCK_LSP_INDEX_SECONDS`` keeps an indexing progress open that long.
- ``"clean_eof"`` — closes stdout after ``didOpen`` but keeps the
  process and stdin alive.
- ``"malformed_frame"`` — writes an invalid frame after ``didOpen``,
  then keeps the process and stdin alive.

The script writes JSON-RPC framed messages to stdout and reads from
stdin.  No third-party dependencies — uses only stdlib so it runs
under whatever Python the test process picks up.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time


def _push_by_content(params, is_change, error_diag):
    """``by_content``: once MOCK_LSP_GATE exists (else after MOCK_LSP_PUSH_DELAY), one error iff the
    synced text contains "bad" (so a baseline of the pre-edit text differs from the edit's verdict)."""
    td = params.get("textDocument") or {}
    text = (params.get("contentChanges") or [{}])[-1].get("text", "") if is_change else td.get("text")
    gate = os.environ.get("MOCK_LSP_GATE")
    deadline = time.monotonic() + 60
    while gate and not os.path.exists(gate) and time.monotonic() < deadline:
        # health: allow HX013 -- a separate process can only poll the test's gate file
        time.sleep(0.01)
    if not gate:
        # health: allow HX013 -- the delay IS the behaviour under test (a server slower than the write)
        time.sleep(float(os.environ.get("MOCK_LSP_PUSH_DELAY", "0.4")))
    write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                   "params": {"uri": td.get("uri", ""), "version": td.get("version", 0),
                              "diagnostics": error_diag if "bad" in (text or "") else []}})


def _push_slow(params, is_change, error_diag):
    """``slow_push``: one error on didOpen; on didChange sleep MOCK_LSP_PUSH_DELAY, then push empty."""
    td = params.get("textDocument") or {}
    if is_change:
        # health: allow HX013 -- the delay IS the behaviour under test (a slow re-check)
        time.sleep(float(os.environ.get("MOCK_LSP_PUSH_DELAY", "1.0")))
    write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                   "params": {"uri": td.get("uri", ""), "version": td.get("version", 0),
                              "diagnostics": [] if is_change else error_diag}})


_PUSH_HANDLERS = {"by_content": _push_by_content, "slow_push": _push_slow}


def _nav_log(method, params):
    """``nav``: append each navigation request to ``MOCK_LSP_REQUESTS_LOG`` (position assertions)."""
    if log_path := os.environ.get("MOCK_LSP_REQUESTS_LOG"):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"method": method, "params": params}) + "\n")


def _nav_loc(path, line, character=0):
    uri = "file://" + path
    return {"uri": uri, "range": {"start": {"line": line, "character": character},
                                  "end": {"line": line, "character": character + 1}}}


def _nav_targets():
    """``MOCK_LSP_NAV_TARGET`` (the answer file) and optional ``MOCK_LSP_NAV_FOREIGN`` (a sibling worktree)."""
    return os.environ["MOCK_LSP_NAV_TARGET"], os.environ.get("MOCK_LSP_NAV_FOREIGN")


def _nav_definition(params):
    _nav_log("textDocument/definition", params)
    target, _ = _nav_targets()
    loc = _nav_loc(target, 1, 6)
    return [{"targetUri": loc["uri"], "targetRange": loc["range"], "targetSelectionRange": loc["range"]}]


_INDEXED = threading.Event()
_INDEXED.set()


def _nav_start_indexing(seconds):
    """``MOCK_LSP_INDEX_SECONDS``: a work-done progress open for that long; references answered
    before it ends cover the open file only (one hit), like a server still listing its files."""
    _INDEXED.clear()
    write_message({"jsonrpc": "2.0", "method": "$/progress",
                   "params": {"token": "index", "value": {"kind": "begin", "title": "Indexing"}}})

    def done():
        _INDEXED.set()
        write_message({"jsonrpc": "2.0", "method": "$/progress", "params": {"token": "index", "value": {"kind": "end"}}})
    threading.Timer(seconds, done).start()


def _nav_references(params):
    _nav_log("textDocument/references", params)
    target, foreign = _nav_targets()
    count = int(os.environ.get("MOCK_LSP_NAV_REFS", "3")) if _INDEXED.is_set() else 1
    refs = [_nav_loc(target, i) for i in reversed(range(count))]
    return refs + ([_nav_loc(foreign, 0)] if foreign else [])


def _nav_hover(params):
    _nav_log("textDocument/hover", params)
    return {"contents": {"kind": "markdown", "value": "```python\nclass Brain\n```\nThe robot's brain."}}


def _nav_workspace_symbol(params):
    _nav_log("workspace/symbol", params)
    target, foreign = _nav_targets()
    found = [{"name": params["query"], "kind": 5, "location": _nav_loc(target, 1, 6), "containerName": "brain_node"}]
    return found + ([{"name": params["query"], "kind": 5, "location": _nav_loc(foreign, 1, 6)}] if foreign else [])


def _nav_document_symbol(params):
    _nav_log("textDocument/documentSymbol", params)
    rng = {"start": {"line": 1, "character": 6}, "end": {"line": 1, "character": 11}}
    child = {"start": {"line": 2, "character": 8}, "end": {"line": 2, "character": 12}}
    return [{"name": "Brain", "kind": 5, "range": rng, "selectionRange": rng,
             "children": [{"name": "tick", "kind": 6, "range": child, "selectionRange": child}]}]


_NAV_ANSWERS = {
    "textDocument/definition": _nav_definition, "textDocument/references": _nav_references,
    "textDocument/hover": _nav_hover, "workspace/symbol": _nav_workspace_symbol,
    "textDocument/documentSymbol": _nav_document_symbol,
}
_IGNORED_NOTIFICATIONS = frozenset({
    "initialized", "workspace/didChangeConfiguration", "workspace/didChangeWatchedFiles", "textDocument/didSave",
})


def _nav_dispatch(msg):
    """``nav``: answer a navigation request (True = handled); start indexing on ``initialized``."""
    method = msg.get("method")
    if method == "initialized" and os.environ.get("MOCK_LSP_INDEX_SECONDS"):
        _nav_start_indexing(float(os.environ["MOCK_LSP_INDEX_SECONDS"]))
    if method not in _NAV_ANSWERS:
        return False
    write_message({"jsonrpc": "2.0", "id": msg["id"], "result": _NAV_ANSWERS[method](msg["params"])})
    return True


def read_message():
    """Read one Content-Length framed JSON-RPC message from stdin."""
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.rstrip(b"\r\n")
        if not line:
            break
        k, _, v = line.decode("ascii").partition(":")
        headers[k.strip().lower()] = v.strip()
    n = int(headers["content-length"])
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))


_WRITE_LOCK = threading.Lock()  # the ``nav`` indexing timer writes from its own thread


def write_message(obj):
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _WRITE_LOCK:
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()


def main():
    script = os.environ.get("MOCK_LSP_SCRIPT", "clean")
    documents = {}
    if script == "slow_tree":
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, pathlib, signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(os.environ['MOCK_LSP_CHILD_PID']).write_text(str(os.getpid())); "
                "time.sleep(60)",
            ],
            env=os.environ,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if script == "oom_abort":
        sys.stderr.write(
            "<oproject>:28982 ms: Mark-Compact 2041.4 (2055.6) -> 2038.4 (2058.4) MB\n"
            "  1317.07 ms (average mu = 0.307, current mu = 0.134)\n"
            "<--- Last few GCs --->\n"
            "FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory\n"
        )
        sys.stderr.flush()
        os.abort()

    if script == "large_stderr":
        # Emit a single stderr line larger than asyncio's 64 KiB default limit
        # to verify the client drain task doesn't crash or deadlock.
        sys.stderr.write("X" * 131_072 + "\n")  # 128 KiB
        sys.stderr.flush()
        script = "clean"

    if script == "oversized_stderr":
        # Exceed the client's StreamReader limit (size passed from _STREAM_LIMIT). readline() converts
        # this into ValueError after discarding the buffered line; the client
        # must keep draining stderr and serve requests normally.
        sys.stderr.write("X" * int(os.environ["MOCK_LSP_STDERR_BYTES"]) + "\n")
        sys.stderr.flush()
        script = "clean"

    while True:
        msg = read_message()
        if msg is None:
            return 0

        if "id" in msg and msg.get("method") == "initialize":
            if script == "init_error":
                # A conformant JSON-RPC error to `initialize`, then exit: the client must
                # surface it as an LSPRequestError carrying the exit details.
                write_message({"jsonrpc": "2.0", "id": msg["id"],
                               "error": {"code": -32602, "message": "bad init"}})
                return 0
            if script in {"slow", "slow_tree"}:
                time.sleep(1.0)
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "capabilities": {
                            "textDocumentSync": 2 if script == "incremental" else 1,
                            "diagnosticProvider": {"interFileDependencies": False, "workspaceDiagnostics": False},
                        },
                        "serverInfo": {"name": "mock-lsp", "version": "0.1"},
                    },
                }
            )
            if script == "crash":
                return 0
            continue

        if script == "nav" and _nav_dispatch(msg):
            continue

        if msg.get("method") in _IGNORED_NOTIFICATIONS:
            continue

        if msg.get("method") == "workspace/didChangeWorkspaceFolders":
            # Multi-root tests observe attached folders through this log.
            log_path = os.environ.get("MOCK_LSP_FOLDERS_LOG")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(msg.get("params")) + "\n")
            continue

        if msg.get("method") in {"textDocument/didOpen", "textDocument/didChange"}:
            params = msg.get("params") or {}
            td = params.get("textDocument") or {}
            uri = td.get("uri", "")
            version = td.get("version", 0)
            is_change = msg.get("method") == "textDocument/didChange"
            if script == "incremental":
                text = documents.get(uri, "") if is_change else td["text"]
                for change in params.get("contentChanges", []):
                    if "range" not in change:
                        text = change["text"]
                        continue
                    # Apply the range using the protocol's UTF-16 offsets, as a real server does.
                    lines = text.splitlines(keepends=True)
                    def byte_offset(position):
                        prefix = "".join(lines[:position["line"]])
                        return len(prefix.encode("utf-16-le")) + position["character"] * 2
                    start = byte_offset(change["range"]["start"])
                    end = byte_offset(change["range"]["end"])
                    encoded = text.encode("utf-16-le")
                    text = (encoded[:start] + change["text"].encode("utf-16-le") + encoded[end:]).decode("utf-16-le")
                documents[uri] = text
            if not is_change and script in {"clean_eof", "malformed_frame"}:
                if script == "malformed_frame":
                    sys.stdout.buffer.write(b"Content-Length: invalid\r\n\r\n")
                    sys.stdout.buffer.flush()
                os.close(sys.stdout.fileno())
                while read_message() is not None:
                    pass
                return 0
            error_diag = [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 5},
                    },
                    "severity": 1,
                    "code": "MOCK001",
                    "source": "mock-lsp",
                    "message": "synthetic error from mock-lsp",
                }
            ]
            if script == "silent":
                # Never publishes diagnostics and the pull endpoint is
                # rejected (below).  Models a slow server that only
                # re-reports after a didChange it never receives — the
                # baseline snapshot has to wait out its full budget.
                continue
            if script == "stale":
                # Ghost scenario: publish an error for the ORIGINAL
                # content, then never publish again after edits.
                if not is_change:
                    write_message(
                        {
                            "jsonrpc": "2.0",
                            "method": "textDocument/publishDiagnostics",
                            "params": {"uri": uri, "version": version, "diagnostics": error_diag},
                        }
                    )
                continue
            if script in _PUSH_HANDLERS:
                _PUSH_HANDLERS[script](params, is_change, error_diag)
                continue
            diagnostics = []
            if script == "errors":
                diagnostics = error_diag
            if script == "versionless":
                # Servers that never echo a document version: the client credits the
                # push with its current version at receipt.
                diagnostics = [] if is_change else error_diag
            params = {"uri": uri, "version": version, "diagnostics": diagnostics}
            if script == "versionless":
                del params["version"]
            write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": params})
            continue

        if script == "incremental" and msg.get("method") == "textDocument/hover":
            uri = msg["params"]["textDocument"]["uri"]
            write_message({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "contents": {"kind": "plaintext", "value": documents[uri]},
            }})
            continue

        if msg.get("method") == "textDocument/diagnostic":
            if script in {"stale", "slow_push", "versionless", "silent", "by_content"}:
                # These scripts model push-only servers so the ghost
                # can't be papered over by the pull channel.
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "error": {"code": -32601, "message": "method not found"},
                    }
                )
                continue
            # Pull endpoint — return empty.
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {"kind": "full", "items": []},
                }
            )
            continue

        if msg.get("method") == "shutdown":
            write_message({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            continue

        if msg.get("method") == "exit":
            return 0

        # Unknown request: respond with method-not-found.
        if "id" in msg:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {"code": -32601, "message": f"method not found: {msg.get('method')}"},
                }
            )


if __name__ == "__main__":
    sys.exit(main())
