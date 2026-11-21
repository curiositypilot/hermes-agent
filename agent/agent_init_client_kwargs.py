"""OpenAI-client kwargs for a primary with explicit CLI/gateway credentials (``agent_init``'s
``_init_openai_client`` explicit branch): URL query split, timeout, ACP launch args, and the
host -> default_headers table."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse, urlunparse

from utils import base_url_host_matches


def _explicit_client_kwargs(agent, api_key, base_url, _provider_timeout) -> dict[str, Any]:
    """OpenAI-client kwargs from explicit CLI/gateway credentials (auth already resolved)."""
    _parsed_url = urlparse(base_url)
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if _parsed_url.query:
        client_kwargs["base_url"] = urlunparse(_parsed_url._replace(query=""))
        client_kwargs["default_query"] = {k: v[0] for k, v in parse_qs(_parsed_url.query).items()}
    if _provider_timeout is not None:
        client_kwargs["timeout"] = _provider_timeout
    # ACP/subprocess providers take launch kwargs instead of HTTP credentials. Keyed on the profile's
    # auth_type, not a vendor slug: every external_process plugin gets this launch path (#102421).
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    if _is_external_process_provider(agent.provider):
        client_kwargs["command"] = agent.acp_command
        client_kwargs["args"] = agent.acp_args
    from agent.provider_route_hooks import route_answer
    _headers_for = _host_default_headers_factory(base_url)
    if _headers_for is not None:
        client_kwargs["default_headers"] = _headers_for(api_key, base_url)
    elif _route_headers := route_answer("client_headers", provider=str(agent.provider or ""), base_url=base_url, scope=agent,
                                        api_key=api_key if isinstance(api_key, str) else ""):
        client_kwargs["default_headers"] = _route_headers
    elif "default_headers" not in client_kwargs:
        # Static profile.default_headers (Vercel AI Gateway attribution, Kimi UA on non-kimi.com endpoints).
        with suppress(Exception):
            from providers import get_provider_profile as _gpf
            _ph = _gpf(agent.provider)
            if _ph and _ph.default_headers:
                client_kwargs["default_headers"] = dict(_ph.default_headers)
    return client_kwargs


def _lazy_headers(module: str, name: str, pass_key: bool = False, pass_base: bool = False):
    """Header factory ``(api_key, base_url) -> dict`` importing ``module.name`` at call time.
    ``pass_key`` forwards ``(key, base_url=base)``; ``pass_base`` forwards ``(base)``."""
    def factory(key, base):
        import importlib
        fn = getattr(importlib.import_module(module), name)
        if pass_key:
            return fn(key, base_url=base)
        return fn(base) if pass_base else fn()
    return factory


# Host → default_headers factory for explicit base_url client construction. Ordered: first
# host match wins; no match falls back to the provider profile's declared headers.
_HOST_DEFAULT_HEADERS: list[tuple[str, Callable[[Any, str], dict[str, str]]]] = [
    ("openrouter.ai", _lazy_headers("agent.auxiliary_client", "build_or_headers")),
    ("integrate.api.nvidia.com",
     _lazy_headers("agent.auxiliary_client", "build_nvidia_nim_headers", pass_base=True)),
    ("api.routermint.com", _lazy_headers("agent.client_lifecycle", "_routermint_headers")),
    ("api.kimi.com", lambda _k, _b: {"User-Agent": "claude-code/0.1.0"}),
    ("portal.qwen.ai", _lazy_headers("agent.client_lifecycle", "_qwen_portal_headers")),
    ("chatgpt.com", _lazy_headers("agent.codex_headers", "codex_cloudflare_headers", pass_key=True)),
    ("x.ai", _lazy_headers("tools.xai_http", "hermes_xai_default_headers")),
]


def _host_default_headers_factory(base_url: str):
    for host, factory in _HOST_DEFAULT_HEADERS:
        if base_url_host_matches(base_url, host):
            return factory
    return None
