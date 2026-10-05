"""A confidential Kanban worker never resolves an auxiliary client outside its data-class policy.

The config mirrors the production leak: every ``auxiliary.<task>`` is pinned to ``antigravity`` (Gemini)
while the main model runs on ``anthropic``. The behaviour contract: with ``HERMES_DATA_CLASS=confidential``
no auxiliary route (per-task provider, either fallback chain, auto discovery, vision, main-agent net)
lands on a provider outside ``[anthropic, openai-codex]``, and ``same_provider_or_fail`` pins side calls to
the main provider or raises ``ProviderDenied``. ``internal`` keeps today's routing.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.provider_policy import ProviderDenied

# The five tasks named by the card's Goal, plus a few more that production pins to antigravity.
GOAL_TASKS = ("compression", "approval", "web_extract", "title_generation", "mcp")
OTHER_PINNED_TASKS = ("vision", "skills_hub", "tts_audio_tags", "triage_specifier", "kanban_decomposer")

_CONFIG = """
model:
  default: claude-opus-5-5
  provider: anthropic
kanban:
  data_policies:
    default: internal
    tenant_defaults:
      work: confidential
    classes:
      internal:
        providers: any
        fallback: true
        auxiliary: any
      confidential:
        providers: [anthropic, openai-codex]
        fallback: false
        auxiliary: same_provider_or_fail
      allowlist_only:
        providers: [anthropic, openai-codex]
        fallback: false
        auxiliary: any
auxiliary:
{aux}
fallback_providers:
  - provider: antigravity
    model: gemini-3.8-flash
  - provider: openai-codex
    model: gpt-6-luna
"""


def _write_config(home, *, chain=None, main_provider="anthropic"):
    aux = []
    for task in GOAL_TASKS + OTHER_PINNED_TASKS:
        aux.append(f"  {task}:\n    provider: antigravity\n    model: gemini-3.8-flash-tiered")
        if chain and task == "compression":
            aux.append("    fallback_chain:")
            aux.extend(f"      - {{provider: {p}, model: {m}}}" for p, m in chain)
    text = _CONFIG.format(aux="\n".join(aux)).replace("provider: anthropic\n", f"provider: {main_provider}\n", 1)
    (home / "config.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def policy_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_KANBAN_TASK", "HERMES_DATA_CLASS", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    ac._aux_denials_seen.clear()
    ac._reset_aux_unhealthy_cache()
    yield home
    ac._aux_denials_seen.clear()


@pytest.fixture
def confidential(policy_home, monkeypatch):
    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")
    return policy_home


class _FakeClients:
    """Replaces every client constructor so the test sees which provider a route ends on."""

    def __init__(self, monkeypatch):
        self.built: list[str] = []
        anthropic = SimpleNamespace(base_url="https://api.anthropic.com", _hermes_test_provider="anthropic")

        def _try_anthropic(explicit_api_key=None):
            self.built.append("anthropic")
            return anthropic, "claude-sonnet-5"

        def _named_custom(req):
            if req.provider == "antigravity":
                self.built.append("antigravity")
                return (SimpleNamespace(base_url="https://antigravity.example", _hermes_test_provider="antigravity"),
                        "gemini-3.8-flash-tiered")
            return None

        def _codex(req):
            self.built.append("openai-codex")
            return SimpleNamespace(base_url="https://codex.example", _hermes_test_provider="openai-codex"), "gpt-6-luna"

        def _discovered(name):
            def _try(*_a, **_k):
                self.built.append(name)
                return SimpleNamespace(base_url=f"https://{name}.example", _hermes_test_provider=name), f"{name}-model"
            return _try

        monkeypatch.setattr(ac, "_try_anthropic", _try_anthropic)
        monkeypatch.setattr(ac, "_resolve_named_custom_branch", _named_custom)
        monkeypatch.setitem(ac._EXPLICIT_PROVIDER_BRANCHES, "openai-codex", _codex)
        for label in ("_try_openrouter", "_try_nous", "_try_custom_endpoint", "_resolve_api_key_provider"):
            monkeypatch.setattr(ac, label, _discovered(label.strip("_")))

    @staticmethod
    def provider_of(client) -> str:
        return getattr(client, "_hermes_test_provider", "")


@pytest.fixture
def fakes(monkeypatch):
    return _FakeClients(monkeypatch)


# --- per-task provider resolution ---------------------------------------------------------------------

def test_confidential_compression_never_resolves_an_antigravity_client(confidential, fakes):
    client, model = ac.get_text_auxiliary_client("compression")

    assert fakes.provider_of(client) == "anthropic"
    assert "antigravity" not in fakes.built
    # The replaced provider's model must not leak onto the main provider.
    assert model != "gemini-3.8-flash-tiered"


@pytest.mark.parametrize("task", GOAL_TASKS + OTHER_PINNED_TASKS)
def test_confidential_task_route_is_rewritten_onto_the_main_provider(confidential, fakes, task):
    provider, model, base_url, api_key, api_mode = ac._resolve_task_provider_model(task)

    assert provider == "anthropic"
    assert (model, base_url, api_key, api_mode) == (None, None, None, None)


@pytest.mark.parametrize("task", GOAL_TASKS)
def test_confidential_call_llm_per_goal_task_only_calls_the_main_provider(confidential, fakes, task):
    """End to end through ``call_llm``: the request is served by the anthropic client, never antigravity."""
    served: list[str] = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None, model="m")

    def _create(**_kwargs):
        served.append("anthropic")
        return response

    anthropic = SimpleNamespace(
        base_url="https://api.anthropic.com", _hermes_test_provider="anthropic",
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    with patch.object(ac, "_try_anthropic", lambda explicit_api_key=None: (anthropic, "claude-sonnet-5")), \
            patch.object(ac, "_validate_llm_response", lambda resp, *_a, **_k: resp), \
            patch.object(ac, "_relay_sync_completion",
                         lambda client, kwargs, **_k: client.chat.completions.create(**kwargs)):
        ac.call_llm(task=task, messages=[{"role": "user", "content": "hi"}])

    assert served == ["anthropic"]
    assert "antigravity" not in fakes.built


def test_caller_named_denied_provider_raises_instead_of_rerouting(confidential, fakes):
    with pytest.raises(ProviderDenied, match="antigravity"):
        ac._resolve_task_provider_model("compression", provider="antigravity")
    with pytest.raises(ProviderDenied):
        ac._resolve_task_provider_model("compression", provider="custom", base_url="https://gemini.example/v1")


def test_provider_alias_cannot_bypass_the_pin(confidential, fakes):
    """``codex`` is an alias of openai-codex: allowlisted, yet not the main provider (anthropic)."""
    with pytest.raises(ProviderDenied, match="pinned to the main provider"):
        ac._resolve_task_provider_model("approval", provider="codex")


def test_confidential_denial_is_recorded_on_the_kanban_card(confidential, fakes, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_KANBAN_DB", str(confidential / "kanban.db"))
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="confidential card", data_class="confidential")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    ac._resolve_task_provider_model("compression")
    ac._resolve_task_provider_model("compression")  # repeated aux calls must not flood the card

    with kbc.connect_closing() as conn:
        denials = [e for e in kb.list_events(conn, task_id) if e.kind == "provider_denied"]
    assert [d.payload["provider"] for d in denials] == ["antigravity"]
    assert denials[0].payload["data_class"] == "confidential"


# --- same_provider_or_fail ----------------------------------------------------------------------------

def test_same_provider_or_fail_raises_when_main_provider_is_unknown(policy_home, monkeypatch, fakes):
    _write_config(policy_home, main_provider="auto")
    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")

    with pytest.raises(ProviderDenied, match="no main provider"):
        ac._resolve_task_provider_model("compression")


def test_same_provider_or_fail_raises_when_main_provider_is_outside_the_class(policy_home, monkeypatch, fakes):
    _write_config(policy_home, main_provider="antigravity")
    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")

    with pytest.raises(ProviderDenied, match="antigravity"):
        ac._resolve_task_provider_model("compression")
    with pytest.raises(ProviderDenied):
        ac.get_text_auxiliary_client("title_generation")


def test_live_runtime_provider_wins_over_the_configured_main_provider(confidential, fakes):
    with ac.scoped_runtime_main({"provider": "openai-codex", "model": "gpt-6-luna"}):
        provider, *_ = ac._resolve_task_provider_model("compression")

    assert provider == "openai-codex"


def test_main_runtime_argument_selects_the_pin(confidential, fakes):
    client, _ = ac.get_text_auxiliary_client("approval", main_runtime={"provider": "openai-codex", "model": "gpt-6-luna"})

    assert fakes.provider_of(client) == "openai-codex"
    assert "antigravity" not in fakes.built


def test_allowlist_only_class_raises_on_a_configured_denied_provider(policy_home, monkeypatch, fakes):
    """``auxiliary: any`` narrows nothing beyond the allowlist, but the allowlist still binds."""
    monkeypatch.setenv("HERMES_DATA_CLASS", "allowlist_only")

    with pytest.raises(ProviderDenied, match="antigravity"):
        ac._resolve_task_provider_model("compression")


# --- the two chain resolvers + the other fallback paths ---------------------------------------------

def test_configured_fallback_chain_skips_denied_entries(policy_home, monkeypatch, fakes):
    _write_config(policy_home, chain=[("antigravity", "gemini-3.8-flash"), ("anthropic", "claude-sonnet-5")])
    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")

    client, model, label = ac._try_configured_fallback_chain("compression", "openai-codex")

    assert fakes.provider_of(client) == "anthropic"
    assert label == "fallback_chain[1](anthropic)"
    assert "antigravity" not in fakes.built


def test_configured_fallback_chain_exhausts_rather_than_use_a_denied_provider(policy_home, monkeypatch, fakes):
    _write_config(policy_home, chain=[("antigravity", "gemini-3.8-flash")])
    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")

    assert ac._try_configured_fallback_chain("compression", "anthropic") == (None, None, "")
    assert "antigravity" not in fakes.built


def test_configured_fallback_chain_is_unchanged_for_internal(policy_home, monkeypatch, fakes):
    _write_config(policy_home, chain=[("antigravity", "gemini-3.8-flash")])

    client, _, label = ac._try_configured_fallback_chain("compression", "anthropic")

    assert fakes.provider_of(client) == "antigravity"
    assert label == "fallback_chain[0](antigravity)"


def test_main_fallback_chain_skips_denied_entries(confidential, fakes):
    with patch.object(ac, "_is_provider_unhealthy", return_value=False):
        client, model, provider = ac._try_main_fallback_chain("compression", "anthropic")

    # fallback_providers = [antigravity, openai-codex]; confidential pins to the main provider
    # (anthropic), so neither entry is eligible.
    assert client is None
    assert "antigravity" not in fakes.built
    assert "openai-codex" not in fakes.built


def test_main_fallback_chain_is_unchanged_for_internal(policy_home, fakes):
    with patch.object(ac, "_is_provider_unhealthy", return_value=False), \
            patch.object(ac, "_context_too_small", return_value=None):
        client, model, provider = ac._try_main_fallback_chain("compression", "anthropic")

    assert fakes.provider_of(client) == "antigravity"


def test_discovery_and_payment_fallback_are_closed_for_confidential(confidential, fakes):
    assert ac._discovery_chain_allowed("auto", "compression") is False
    assert ac._try_payment_fallback("anthropic", "compression", main_runtime={"provider": "auto"}) == (None, None, "")
    assert fakes.built == []


def test_discovery_chain_still_open_for_internal_with_no_main_provider(policy_home, fakes):
    assert ac._discovery_chain_allowed("auto", "compression") is True


def test_main_agent_model_fallback_refuses_a_denied_main_provider(policy_home, monkeypatch, fakes):
    _write_config(policy_home, main_provider="antigravity")
    monkeypatch.setenv("HERMES_DATA_CLASS", "allowlist_only")

    assert ac._try_main_agent_model_fallback("openai-codex", "compression") == (None, None, "")
    assert "antigravity" not in fakes.built


def test_auto_route_fails_closed_when_the_main_provider_is_denied(policy_home, monkeypatch, fakes):
    _write_config(policy_home, main_provider="antigravity")
    monkeypatch.setenv("HERMES_DATA_CLASS", "confidential")

    with pytest.raises(ProviderDenied, match="antigravity"):
        ac._resolve_auto_route(task="compression")
    assert fakes.built == []


def test_auto_route_uses_the_main_provider_for_confidential(confidential, fakes):
    client, _, provider = ac._resolve_auto_route(task="compression")

    assert fakes.provider_of(client) == "anthropic"
    assert provider == "anthropic"


def test_vision_auto_route_skips_aggregators_for_confidential(confidential, fakes):
    with patch.object(ac, "_vision_main_provider_client", return_value=(None, None)):
        assert ac._vision_auto_route({"provider": "anthropic", "model": "claude-opus-5-5"}, None, None, False) \
            == (None, None, None)
    assert fakes.built == []


def test_router_backstop_refuses_a_denied_provider_from_any_caller(confidential, fakes):
    with pytest.raises(ProviderDenied, match="antigravity"):
        ac.resolve_provider_client("antigravity", "gemini-3.8-flash")
    assert fakes.built == []


# --- internal / unclassified behaviour is unchanged ------------------------------------------------

def test_internal_class_keeps_the_configured_antigravity_route(policy_home, fakes):
    provider, model, *_ = ac._resolve_task_provider_model("compression")
    client, _ = ac.get_text_auxiliary_client("compression")

    assert (provider, model) == ("antigravity", "gemini-3.8-flash-tiered")
    assert fakes.provider_of(client) == "antigravity"


def test_unknown_data_class_fails_closed(policy_home, monkeypatch, fakes):
    monkeypatch.setenv("HERMES_DATA_CLASS", "secret-ish")

    with pytest.raises(ProviderDenied, match="unknown data class"):
        ac._resolve_task_provider_model("compression")


def test_non_kanban_sessions_resolve_as_internal(policy_home, fakes):
    """No HERMES_DATA_CLASS / HERMES_KANBAN_TASK: the policy default (internal) applies."""
    assert ac._aux_policy_context() is None
