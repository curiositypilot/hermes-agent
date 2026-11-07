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
- ``"loading"`` — models rust-analyzer in a cold workspace: until the file named by ``MOCK_LSP_GATE``
  exists it pushes EMPTY diagnostics on every open/change and answers pulls with ``[]``; then it sends
  ``experimental/serverStatus`` quiescent=true and the real diagnostics (one error) by push and pull.
  ``MOCK_LSP_STATUS_EARLY=1`` also sends quiescent=false right after ``initialized``.
- ``"flycheck"`` — models rust-analyzer's two producers.  Native diagnostics by pull only: an E0308
  (source ``rust-analyzer``) at 0:0 iff the synced text contains ``native_bad``.  On didSave, after
  ``MOCK_LSP_CHECK_START_DELAY`` (default 0) a ``$/progress`` begin on ``MOCK_LSP_CHECK_TOKEN`` (default
  ``rust-analyzer/flycheck/0``), then after ``MOCK_LSP_CHECK_DELAY`` (default 1.0) a push of what cargo
  check finds in the file ON DISK (rustc E0502 at ``borrow_bad``'s line, column 4; a rustc E0308 at 0:0
  iff ``native_bad``), then the ``end``.  ``MOCK_LSP_CHECK_BEGIN=0``: no check at all;
  ``MOCK_LSP_CHECK_END=0``: the check never ends.  Checks run one at a time (a save queues behind a
  running one).  ``MOCK_LSP_OPEN_CHECK=<s>``: the first didOpen starts a check of that duration that
  pushes nothing (a check already running when the edit is saved).
- ``"incremental"`` — apply ranged edits as UTF-16 and expose the server's
  document mirror through hover, for synchronization contract tests.
- ``"nav"`` — answers definition / references / hover / workspace symbol / document symbol with
  canned locations in ``MOCK_LSP_NAV_TARGET`` (see ``_NAV_ANSWERS``); logs each request to
  ``MOCK_LSP_REQUESTS_LOG``; ``MOCK_LSP_INDEX_SECONDS`` keeps an indexing progress open that long;
  ``MOCK_LSP_NAV_NO_OUTLINE=1`` answers documentSymbol with ``[]``; ``MOCK_LSP_NAV_DEAD_LINES`` (0-based
  csv) answers definition / references / hover at those lines with ``[]`` / ``null``.
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
import re
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
_WRITE_LOCK = threading.Lock()
_MOCK_ERROR = [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
                "severity": 1, "code": "MOCK001", "source": "mock-lsp", "message": "synthetic error from mock-lsp"}]
_LOADING = {"loaded": False, "docs": {}}  # ``loading`` script: load state, uri -> last synced version


def _status(quiescent):
    write_message({"jsonrpc": "2.0", "method": "experimental/serverStatus",
                   "params": {"health": "ok", "quiescent": quiescent}})


def _loading_verdict(error_diag):
    return error_diag if _LOADING["loaded"] else []


def _finish_loading_when_gated(error_diag):
    """``loading``: once MOCK_LSP_GATE exists, report quiescent and publish the real diagnostics."""
    gate = os.environ["MOCK_LSP_GATE"]
    while not os.path.exists(gate):
        # health: allow HX013 -- a separate process can only poll the test's gate file
        time.sleep(0.01)
    with _WRITE_LOCK:
        _LOADING["loaded"] = True
        _status(True)
        for uri, version in _LOADING["docs"].items():
            write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                           "params": {"uri": uri, "version": version, "diagnostics": error_diag}})


def _push_loading(params, is_change, error_diag):
    td = params.get("textDocument") or {}
    with _WRITE_LOCK:
        _LOADING["docs"][td.get("uri", "")] = td.get("version", 0)
        write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                       "params": {"uri": td.get("uri", ""), "version": td.get("version", 0),
                                  "diagnostics": _loading_verdict(error_diag)}})


def _start_loading(_msg):
    if os.environ.get("MOCK_LSP_STATUS_EARLY") == "1":
        _status(False)
    threading.Thread(target=_finish_loading_when_gated, args=(_MOCK_ERROR,), daemon=True).start()
    return True


def _pull_loading(msg):
    with _WRITE_LOCK:
        write_message({"jsonrpc": "2.0", "id": msg["id"],
                       "result": {"kind": "full", "items": _loading_verdict(_MOCK_ERROR)}})
    return True


_PUSH_HANDLERS["loading"] = _push_loading
# (script, method) -> handler run before the generic dispatch; True = the message is handled.
_SCRIPT_HOOKS = {("loading", "initialized"): _start_loading, ("loading", "textDocument/diagnostic"): _pull_loading}


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


def _nav_class_at():
    """``(line0, char)`` of the first ``class <name>`` identifier in ``MOCK_LSP_NAV_TARGET``."""
    target, _ = _nav_targets()
    with open(target, encoding="utf-8") as fh:
        for line0, src in enumerate(fh.read().split("\n")):
            if m := re.match(r"\s*class (\w+)", src):
                return line0, m.start(1)
    return 1, 6


def _nav_dead(params):
    """``MOCK_LSP_NAV_DEAD_LINES`` (0-based csv): positions on those lines have nothing to answer."""
    dead = {int(x) for x in os.environ.get("MOCK_LSP_NAV_DEAD_LINES", "").split(",") if x.strip()}
    return params["position"]["line"] in dead


def _nav_definition(params):
    _nav_log("textDocument/definition", params)
    if _nav_dead(params):
        return []
    target, _ = _nav_targets()
    loc = _nav_loc(target, *_nav_class_at())
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
    if _nav_dead(params):
        return []
    target, foreign = _nav_targets()
    count = int(os.environ.get("MOCK_LSP_NAV_REFS", "3")) if _INDEXED.is_set() else 1
    refs = [_nav_loc(target, i) for i in reversed(range(count))]
    return refs + ([_nav_loc(foreign, 0)] if foreign else [])


def _nav_hover(params):
    _nav_log("textDocument/hover", params)
    if _nav_dead(params):
        return None
    return {"contents": {"kind": "markdown", "value": "```python\nclass Brain\n```\nThe robot's brain."}}


def _nav_workspace_symbol(params):
    _nav_log("workspace/symbol", params)
    target, foreign = _nav_targets()
    found = [{"name": params["query"], "kind": 5, "location": _nav_loc(target, *_nav_class_at()),
              "containerName": "brain_node"}]
    return found + ([{"name": params["query"], "kind": 5, "location": _nav_loc(foreign, 1, 6)}] if foreign else [])


def _nav_document_symbol(params):
    _nav_log("textDocument/documentSymbol", params)
    if os.environ.get("MOCK_LSP_NAV_NO_OUTLINE") == "1":
        return []
    line0, char = _nav_class_at()
    rng = {"start": {"line": line0, "character": char}, "end": {"line": line0, "character": char + 5}}
    child = {"start": {"line": line0 + 1, "character": 8}, "end": {"line": line0 + 1, "character": 12}}
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


def _nav_answer(msg):
    """``nav``: answer one navigation request."""
    write_message({"jsonrpc": "2.0", "id": msg["id"], "result": _NAV_ANSWERS[msg["method"]](msg["params"])})
    return True


def _nav_initialized(_msg):
    if os.environ.get("MOCK_LSP_INDEX_SECONDS"):
        _nav_start_indexing(float(os.environ["MOCK_LSP_INDEX_SECONDS"]))
    return True


_FLYCHECK_DOCS = {}  # ``flycheck``: uri -> (version, synced text)


def _fly_diag(line, character, code, source, message):
    return {"range": {"start": {"line": line, "character": character},
                      "end": {"line": line, "character": character + 3}},
            "severity": 1, "code": code, "source": source, "message": message}


def _fly_sync(msg):
    td = msg["params"]["textDocument"]
    changes = msg["params"].get("contentChanges")
    first_open = not _FLYCHECK_DOCS and msg["method"] == "textDocument/didOpen"
    _FLYCHECK_DOCS[td["uri"]] = (td.get("version", 0), changes[-1]["text"] if changes else td["text"])
    if first_open and os.environ.get("MOCK_LSP_OPEN_CHECK"):
        _FLY_CHECK_LOCK.acquire()
        _fly_progress(_fly_token(), "begin")
        threading.Thread(target=_fly_open_check, args=(float(os.environ["MOCK_LSP_OPEN_CHECK"]),), daemon=True).start()
    return True


def _fly_pull(msg):
    text = _FLYCHECK_DOCS.get(msg["params"]["textDocument"]["uri"], (0, ""))[1]
    items = [_fly_diag(0, 0, "E0308", "rust-analyzer", "mismatched types")] if "native_bad" in text else []
    write_message({"jsonrpc": "2.0", "id": msg["id"], "result": {"kind": "full", "items": items}})
    return True


def _fly_progress(token, kind):
    write_message({"jsonrpc": "2.0", "method": "$/progress",
                   "params": {"token": token, "value": {"kind": kind, "title": "cargo check"}}})


def _fly_cargo_check(uri):
    """What ``cargo check`` reports for the file: it reads the disk, not the synced text."""
    try:
        with open(uri[len("file://"):], encoding="utf-8") as fh:
            disk = fh.read()
    except OSError:
        disk = ""
    diags = [_fly_diag(0, 0, "E0308", "rustc", "mismatched types")] if "native_bad" in disk else []
    if "borrow_bad" in disk:
        line = next(i for i, text in enumerate(disk.splitlines()) if "borrow_bad" in text)
        diags.append(_fly_diag(line, 4, "E0502", "rustc", "cannot borrow `v` as immutable"))
    return diags


_FLY_CHECK_LOCK = threading.Lock()  # one cargo check at a time; a later save's check queues behind it


def _fly_token():
    return os.environ.get("MOCK_LSP_CHECK_TOKEN", "rust-analyzer/flycheck/0")


def _fly_run_check(uri):
    with _FLY_CHECK_LOCK:
        # health: allow HX013 -- the delays ARE the behaviour under test (cargo check starting and running)
        time.sleep(float(os.environ.get("MOCK_LSP_CHECK_START_DELAY", "0")))
        _fly_progress(_fly_token(), "begin")
        time.sleep(float(os.environ.get("MOCK_LSP_CHECK_DELAY", "1.0")))  # health: allow HX013 -- as above
        version = _FLYCHECK_DOCS.get(uri, (0, ""))[0]
        write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                       "params": {"uri": uri, "version": version, "diagnostics": _fly_cargo_check(uri)}})
        if os.environ.get("MOCK_LSP_CHECK_END", "1") != "0":
            _fly_progress(_fly_token(), "end")


def _fly_open_check(seconds):
    """``MOCK_LSP_OPEN_CHECK``: the running check started at didOpen (holds the lock taken there)."""
    try:
        time.sleep(seconds)  # health: allow HX013 -- the check's duration is the behaviour under test
        _fly_progress(_fly_token(), "end")
    finally:
        _FLY_CHECK_LOCK.release()


def _fly_save(msg):
    if os.environ.get("MOCK_LSP_CHECK_BEGIN", "1") != "0":
        threading.Thread(target=_fly_run_check, args=(msg["params"]["textDocument"]["uri"],), daemon=True).start()
    return True


_SCRIPT_HOOKS.update({("flycheck", "textDocument/didOpen"): _fly_sync, ("flycheck", "textDocument/didChange"): _fly_sync,
                      ("flycheck", "textDocument/diagnostic"): _fly_pull, ("flycheck", "textDocument/didSave"): _fly_save})
_SCRIPT_HOOKS.update({("nav", method): _nav_answer for method in _NAV_ANSWERS})
_SCRIPT_HOOKS[("nav", "initialized")] = _nav_initialized


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


def write_message(obj):
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _FRAME_LOCK:  # the ``loading`` script also writes from its gate thread
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()


_FRAME_LOCK = threading.Lock()


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

        hook = _SCRIPT_HOOKS.get((script, msg.get("method")))
        if hook is not None and hook(msg):
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
            error_diag = _MOCK_ERROR
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
