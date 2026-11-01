"""A managed worker (every Kanban run) builds its agent on the credential pool its route resolved
from, as the in-process turn does (``_runtime_agent_kwargs``). Without it the agent's pool recovery
returns at ``pool is None``: an xAI spending-limit 403 never benches the row, so auth.json keeps it
``ok`` and the routing gates keep pinning work to a dead credential (intake t_1ad2d997: 198 failed
runs, row still ``last_status=ok``)."""
import json
import secrets

from tests.gateway.test_managed_worker_route_key import _constructed_agent, _routed_frame

XAI_URL = 'https://api.x.ai/v1'
# The 403 body the pinned Grok critics got, 2026-11-01 08:31:09 UTC (held_critics_events.json).
SPENDING_LIMIT = {'reason': 'personal-team-blocked:spending-limit',
                  'message': 'You have run out of credits or need a Grok subscription. Add credits at '
                             'https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok.'}


def _xai_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'auth.json').write_text(json.dumps({'version': 1, 'providers': {}, 'credential_pool': {'xai-oauth': [{
        'id': 'bc0708', 'label': 'xai-test', 'auth_type': 'oauth', 'priority': 0, 'source': 'device_code',
        'access_token': 'fake-xai-access-' + secrets.token_hex(8),
        'refresh_token': 'fake-xai-refresh-' + secrets.token_hex(8),
        'base_url': XAI_URL, 'last_status': 'ok'}]}}))
    return home


def _xai_frame(tmp_path, monkeypatch, api_key=None):
    return _routed_frame(tmp_path, monkeypatch, 'grok-4.7', {'provider': 'xai-oauth'}, api_key=api_key)


def _worker_route(frame, construct=None):
    from gateway.session_policy import restore_policy
    from gateway.session_worker_construct import worker_route
    return worker_route(restore_policy(frame['policy']), {'api_key': frame['api_key'], **(construct or {})})


def test_keyless_worker_route_carries_its_providers_pool(tmp_path, monkeypatch):
    from agent.credential_pool import CredentialPool
    _xai_home(tmp_path, monkeypatch)
    route = _worker_route(_xai_frame(tmp_path, monkeypatch))
    assert isinstance(route['credential_pool'], CredentialPool)
    assert route['credential_pool'].provider == 'xai-oauth'
    selected = route['credential_pool'].current()
    assert selected is not None and route['api_key'] == selected.access_token


def test_bound_launch_key_absent_from_the_pool_attaches_no_pool(tmp_path, monkeypatch):
    _xai_home(tmp_path, monkeypatch)
    launch = 'fake-launch-' + secrets.token_hex(8)
    route = _worker_route(_xai_frame(tmp_path, monkeypatch, api_key=launch))
    assert route['api_key'] == launch and route['credential_pool'] is None


def test_bound_key_never_borrows_a_pool_the_route_resolved_for_another_key(tmp_path, monkeypatch):
    """Even when resolution hands back a pool, a bound key it did not select keeps the pool out."""
    _xai_home(tmp_path, monkeypatch)
    sentinel = object()
    monkeypatch.setattr('gateway.session_worker_construct.route_runtime', lambda *a, **k: (
        {'provider': 'xai-oauth', 'api_key': 'fake-pool-entry', 'base_url': XAI_URL, 'credential_pool': sentinel}, None))
    frame = _xai_frame(tmp_path, monkeypatch, api_key='fake-launch-' + secrets.token_hex(8))
    assert _worker_route(frame)['credential_pool'] is None
    assert _worker_route({**frame, 'api_key': 'fake-pool-entry'})['credential_pool'] is sentinel


def test_fallback_entry_route_stays_empty(tmp_path, monkeypatch):
    _xai_home(tmp_path, monkeypatch)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_with_fallback', lambda config, **kw: (
        {'provider': 'openrouter', 'api_key': 'fake-other-provider'}, {'provider': 'openrouter', 'model': 'x'}))
    assert _worker_route(_xai_frame(tmp_path, monkeypatch)) == {}


def test_worker_spending_limit_403_benches_the_pool_row(tmp_path, monkeypatch):
    """Incident replay: the agent built the way ``managed_worker.execute`` builds it gets the 403
    the critics got, and the auth.json row is benched (fails on the pool-less worker: row stays ok)."""
    from agent.error_classifier import FailoverReason
    home = _xai_home(tmp_path, monkeypatch)
    agent = _constructed_agent(_xai_frame(tmp_path, monkeypatch), monkeypatch)
    assert agent._credential_pool is not None
    agent._recover_with_credential_pool(status_code=403, has_retried_429=False,
                                        classified_reason=FailoverReason.billing, error_context=dict(SPENDING_LIMIT))
    row, = json.loads((home / 'auth.json').read_text())['credential_pool']['xai-oauth']
    assert row['last_status'] == 'exhausted'
    assert row['last_error_code'] == 403
    assert row['billing_streak'] == 1
