"""A dead refresh token is presented to the authorization server at most once per connect attempt.

Regression for intake t_9dc6370b: with the refresh token rejected (400) and the MCP endpoint
answering 401, one connect attempt of the adamchik gateway POSTed the same dead grant to
``/token`` 16-38 times in ~40 s. After a failed refresh the SDK drops ``_initialized``, so every
later request of the attempt re-ran ``_initialize``, re-read the unchanged pair from disk and
refreshed it again. A pair a peer rotated in on disk must still be adopted.
"""

import asyncio
import json
import time

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")

MCP_URL = "https://mcp.example.test/mcp"
AUTH = "https://auth.example.test"


def _write_credentials(home, refresh_token="dead-refresh", access_token="dead-access", expires_at=None):
    tokens = home / "mcp-tokens"
    tokens.mkdir(parents=True, exist_ok=True)
    (tokens / "srv.json").write_text(json.dumps({
        "access_token": access_token, "token_type": "Bearer", "expires_in": 3600,
        "expires_at": expires_at if expires_at is not None else time.time() - 3600,
        "refresh_token": refresh_token}))
    (tokens / "srv.client.json").write_text(json.dumps({
        "client_id": "cid", "redirect_uris": ["http://127.0.0.1:12345/callback"],
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        "token_endpoint_auth_method": "none"}))


def _install_fake_servers(monkeypatch, token_handler=None, mcp_handler=None):
    """Route every SDK-flavoured httpx client through one mock: PRM/ASM discovery succeeds, the token
    endpoint rejects the grant (400) and the MCP endpoint wants auth (401) unless overridden."""
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    seen = []

    def handler(request):
        url = str(request.url)
        seen.append((request.method, url, request.headers.get("authorization")))
        if "/.well-known/oauth-protected-resource" in url:
            return httpx.Response(200, json={"resource": MCP_URL, "authorization_servers": [AUTH]})
        if "/.well-known/oauth-authorization-server" in url or "/.well-known/openid-configuration" in url:
            return httpx.Response(200, json={
                "issuer": AUTH, "authorization_endpoint": f"{AUTH}/authorize", "token_endpoint": f"{AUTH}/token",
                "registration_endpoint": f"{AUTH}/register", "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"]})
        if url == f"{AUTH}/token":
            return token_handler(request) if token_handler else httpx.Response(400, json={"error": "invalid_grant"})
        if url.startswith(MCP_URL):
            if mcp_handler:
                return mcp_handler(request)
            return httpx.Response(401, headers={"WWW-Authenticate": (
                'Bearer resource_metadata="https://mcp.example.test/.well-known/oauth-protected-resource/mcp"')})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.pop("mounts", None)
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


async def _connect_once(task):
    from tools.mcp_oauth import _oauth_interactive_enabled
    _oauth_interactive_enabled.set(False)  # a gateway reconnect: never a browser flow
    cfg = {"url": MCP_URL, "auth": "oauth", "connect_timeout": 15}
    task._config, task._auth_type = cfg, "oauth"
    with pytest.raises(BaseException):
        await asyncio.wait_for(task._run_http(cfg), 30)


def _token_posts(seen):
    return [s for s in seen if s[0] == "POST" and s[1] == f"{AUTH}/token"]


def test_dead_refresh_token_is_posted_once_per_connect_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_credentials(tmp_path)
    seen = _install_fake_servers(monkeypatch)
    from tools.mcp_oauth_manager import reset_manager_for_tests
    from tools.mcp_tool import MCPServerTask
    reset_manager_for_tests()

    asyncio.run(_connect_once(MCPServerTask("srv")))

    assert any(s[1].startswith(MCP_URL) for s in seen), "the attempt never reached the MCP endpoint"
    assert len(_token_posts(seen)) == 1, f"dead refresh token POSTed {len(_token_posts(seen))}x in one attempt"


def test_later_requests_and_reconnects_never_replay_the_rejected_grant(monkeypatch, tmp_path):
    """The burst mechanism: each request after the failed refresh re-ran ``_initialize``, re-read the
    unchanged dead pair from disk and refreshed it again (one /token POST per request)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_credentials(tmp_path)
    seen = _install_fake_servers(monkeypatch)
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from tools.mcp_tool import MCPServerTask, sdk_httpx
    reset_manager_for_tests()
    httpx = sdk_httpx()

    async def scenario():
        from tools.mcp_oauth import _oauth_interactive_enabled
        _oauth_interactive_enabled.set(False)
        provider = get_manager().get_or_build_provider("srv", MCP_URL, None)
        async with httpx.AsyncClient(auth=provider) as client:
            for i in range(4):
                with pytest.raises(BaseException):
                    await client.post(MCP_URL, json={"jsonrpc": "2.0", "id": i, "method": "ping"})
        await _connect_once(MCPServerTask("srv"))  # a later revival on the same cached provider

    asyncio.run(scenario())

    assert sum(1 for s in seen if s[1].startswith(MCP_URL)) >= 5, "the requests never reached the MCP endpoint"
    assert len(_token_posts(seen)) == 1, f"dead refresh token POSTed {len(_token_posts(seen))}x"


def test_peer_rotated_pair_on_disk_is_still_adopted_after_a_failed_refresh(monkeypatch, tmp_path):
    """Must-not-match: once the dead pair failed, a DIFFERENT pair written to disk (a peer rotation or
    `hermes mcp login`) is picked up and presented, not suppressed as 'already failed'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_credentials(tmp_path)
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()

    def mcp_handler(request):
        if request.headers.get("authorization") == "Bearer fresh-access":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 0, "result": {}})
        return httpx.Response(401, headers={"WWW-Authenticate": "Bearer"})

    seen = _install_fake_servers(monkeypatch, mcp_handler=mcp_handler)
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()

    async def scenario():
        from tools.mcp_oauth import _oauth_interactive_enabled
        _oauth_interactive_enabled.set(False)
        provider = get_manager().get_or_build_provider("srv", MCP_URL, None)
        async with httpx.AsyncClient(auth=provider) as client:
            with pytest.raises(BaseException):
                await client.post(MCP_URL, json={"jsonrpc": "2.0", "id": 0, "method": "ping"})
            # A peer rotates the grant on disk (new refresh AND live access token).
            _write_credentials(tmp_path, refresh_token="fresh-refresh", access_token="fresh-access",
                               expires_at=time.time() + 3600)
            return await client.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    response = asyncio.run(scenario())
    assert response.status_code == 200
    assert [s for s in seen if s[1].startswith(MCP_URL)][-1][2] == "Bearer fresh-access"
