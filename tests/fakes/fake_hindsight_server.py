"""Recording loopback Hindsight API for end-to-end tests (stdlib only).

Answers what the Hindsight memory plugin's cloud client (``hindsight_client``) sends on a turn:
``GET /version``, recall (``POST .../memories/recall``), retain (``POST .../memories``) and the
bank config read; anything else gets ``{}``. Every request is recorded (method, path, decoded body)
so a test can assert on exactly what reached the memory server. The response shapes follow the
client's models (``RetainResponse``, ``RecallResponse``): a client bump that adds a required field
fails loudly here.

Usage::

    with FakeHindsightServer() as hs:
        config["api_url"] = hs.base_url
        ...run hermes...
        assert not hs.requests_containing("SECRET")
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

API_VERSION = "0.10.1"
_BANK = re.compile(r"/v1/[^/]+/banks/([^/]+)/")


class FakeHindsightServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeHindsightServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.requests)

    def retain_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.snapshot() if r["method"] == "POST" and r["path"].rstrip("/").endswith("/memories")]

    def requests_containing(self, text: str) -> list[dict[str, Any]]:
        return [r for r in self.snapshot() if text in r["raw"]]

    def _record(self, method: str, path: str, raw: str) -> None:
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        with self._lock:
            self.requests.append({"method": method, "path": path, "raw": raw, "body": body})

    @staticmethod
    def _answer(method: str, path: str, body: Any) -> dict[str, Any]:
        bare = path.split("?", 1)[0].rstrip("/")
        if bare.endswith("/version"):
            return {"version": API_VERSION, "api_version": API_VERSION}
        if method == "POST" and bare.endswith("/memories/recall"):
            return {"results": []}
        if method == "POST" and bare.endswith("/memories"):
            bank = (m.group(1) if (m := _BANK.search(bare + "/")) else "bank")
            items = body.get("items") if isinstance(body, dict) else None
            return {"success": True, "bank_id": bank, "items_count": len(items or []), "async": False}
        return {}

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
                fake._record(self.command, self.path, raw)
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                payload = json.dumps(fake._answer(self.command, self.path, body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _serve

            def log_message(self, *args: object) -> None:
                pass

        return Handler
