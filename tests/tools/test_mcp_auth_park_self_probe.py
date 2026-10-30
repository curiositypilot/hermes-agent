"""The timed self-probe of a server parked on an OAuth failure must not replay the rejected grant.

Regression for intake t_9dc6370b: ``_park()`` woke every ``_PARKED_RETRY_INTERVAL`` and rebuilt the
transport, which re-ran OAuth with the same dead refresh token (one /token 400 every 5 minutes, for
weeks). Only the discovery nudge honoured ``_auth_park``. A re-authorization (new credential stamp)
or an explicit reconnect still revives; a park for any other reason keeps probing.
"""

import asyncio
import json

import pytest

from tests.waiting import async_wait_until

CFG = {"url": "https://mcp.example.test/mcp", "auth": "oauth"}


@pytest.fixture
def token_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools import mcp_tool_config as _config
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"cloud": dict(CFG)})
    path = HermesTokenStorage("cloud")._tokens_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"access_token": "a", "token_type": "Bearer", "refresh_token": "dead"}))
    return path


def _parked_task(failure: BaseException):
    from tools.mcp_tool import MCPServerTask

    class _Task(MCPServerTask):
        builds = 0
        self_probes = 0

        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _run_http(self, config):
            type(self).builds += 1
            raise failure

        async def _wait_for_reconnect_or_shutdown(self, timeout=None):
            outcome = await super()._wait_for_reconnect_or_shutdown(timeout)
            if outcome == "self-probe":
                type(self).self_probes += 1
            return outcome

    return _Task("cloud")


async def _start(task):
    run = asyncio.ensure_future(task.run(dict(CFG)))
    await async_wait_until(lambda: task._was_parked, what="server parked")
    return run


async def _stop(task, run):
    task._shutdown_event.set()
    await asyncio.wait_for(run, 5)


def _auth_failure():
    from tools.mcp_oauth import OAuthNonInteractiveError
    return OAuthNonInteractiveError("refresh token rejected; browser authorization required")


def test_auth_parked_server_does_not_self_probe_while_credentials_are_unchanged(token_file):
    task = _parked_task(_auth_failure())

    async def scenario():
        run = await _start(task)
        assert task._auth_park is not None
        await async_wait_until(lambda: task.self_probes >= 5, what="five self-probe wakes")
        assert task.builds == 1, f"self-probe replayed the rejected grant {task.builds - 1}x"
        await _stop(task, run)

    asyncio.run(scenario())


def test_rewritten_tokens_file_revives_on_the_next_wake(token_file):
    task = _parked_task(_auth_failure())

    async def scenario():
        run = await _start(task)
        await async_wait_until(lambda: task.self_probes >= 2, what="two self-probe wakes")
        assert task.builds == 1
        token_file.write_text(json.dumps({"access_token": "fresh", "token_type": "Bearer",
                                          "refresh_token": "fresh-refresh"}))  # `hermes mcp login`
        await async_wait_until(lambda: task.builds == 2, what="revival after re-login")
        await _stop(task, run)

    asyncio.run(scenario())


def test_explicit_reconnect_revives_even_with_unchanged_credentials(token_file):
    task = _parked_task(_auth_failure())

    async def scenario():
        run = await _start(task)
        await async_wait_until(lambda: task.self_probes >= 2, what="two self-probe wakes")
        assert task.builds == 1
        task._reconnect_event.set()  # manual /mcp refresh
        await async_wait_until(lambda: task.builds == 2, what="revival on explicit reconnect")
        await _stop(task, run)

    asyncio.run(scenario())


def test_non_auth_park_still_self_probes_every_interval(token_file):
    """Must-not-match: a park with no ``_auth_park`` (timeout, outage) keeps probing."""
    task = _parked_task(TimeoutError("backend down"))

    async def scenario():
        run = await _start(task)
        assert task._auth_park is None
        await async_wait_until(lambda: task.builds >= 4, what="repeated self-probe revivals")
        await _stop(task, run)

    asyncio.run(scenario())
