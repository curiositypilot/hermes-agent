"""Memory-provider tool-schema plumbing: the gate that decides whether external memory-provider
tools are exposed on an agent's tool surface, and the injection of their schemas."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_tool_schema(schema: Any) -> Optional[dict[str, Any]]:
    """Return a bare function-tool dict with a resolvable top-level ``name``, else None.

    Providers should return ``{"name", "description", "parameters"}`` but some return the
    wrapped OpenAI form; wrapping that twice yields a nameless ``function`` and strict
    providers (DeepSeek) reject the ENTIRE request, so both shapes are normalized here.
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    name = schema.get("name", "")
    return schema if name and isinstance(name, str) else None


def memory_provider_tools_enabled(enabled_toolsets: Optional[list[str]], disabled_toolsets: Optional[list[str]] = None,
                                  *, memory_tool_present: bool = False) -> bool:
    """Return whether external memory-provider tools should be exposed."""
    if disabled_toolsets and "memory" in disabled_toolsets:
        return False
    if memory_tool_present or enabled_toolsets is None:
        return True
    if not enabled_toolsets:
        return False
    if "memory" in enabled_toolsets:
        return True
    try:
        from toolsets import resolve_toolset

        return any("memory" in resolve_toolset(name) for name in enabled_toolsets)
    except Exception:
        logger.debug("Failed to resolve enabled toolsets for memory-provider tools", exc_info=True)
        return False


def _tool_name(tool: Any) -> Any:
    return tool.get("function", {}).get("name") if isinstance(tool, dict) else None


def memory_provider_tools_exposed(agent: Any) -> bool:
    """Whether external memory-provider tools are exposed on ``agent``.

    Same gate as ``inject_memory_provider_tools`` so a provider's ``system_prompt_block()``
    never advertises tools absent from the tool surface.
    """
    tools = getattr(agent, "tools", None)
    present = isinstance(tools, (list, tuple)) and any(_tool_name(t) == "memory" for t in tools)
    enabled, disabled = getattr(agent, "enabled_toolsets", None), getattr(agent, "disabled_toolsets", None)
    return memory_provider_tools_enabled(enabled, disabled, memory_tool_present=present)


def memory_provider_tool_defs(agent: Any) -> list[dict[str, Any]]:
    """Gated, normalized, wrapped memory-provider schemas not already on ``agent.tools``;
    also stored as ``agent._memory_tool_defs`` (``[]`` when gated off or absent)."""
    agent._memory_tool_defs = defs = []
    memory_manager = getattr(agent, "_memory_manager", None)
    tools = getattr(agent, "tools", None)
    if not memory_manager or tools is None:
        return defs

    if not memory_provider_tools_exposed(agent):
        # Say so once: a silent 0 leaves the provider looking "half on" with no clue which
        # config key (platform_toolsets / disabled_toolsets) gated it.
        # See #81014.
        _providers = [p for p in getattr(memory_manager, "providers", None) or []
                      if getattr(p, "name", "") != "builtin"]
        if _providers:
            logger.info(
                "Memory provider(s) %s configured but the 'memory' toolset is "
                "gated off for this session (platform_toolsets / "
                "agent.disabled_toolsets) — provider tools and system-prompt "
                "block are both withheld.",
                [getattr(p, "name", type(p).__name__) for p in _providers],
            )
        return defs

    get_schemas = getattr(memory_manager, "get_all_tool_schemas", None)
    if not callable(get_schemas):
        return defs

    existing_tool_names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    for raw_schema in get_schemas():
        schema = normalize_tool_schema(raw_schema)
        if schema is None:
            logger.warning(
                "Memory provider returned a tool schema with no resolvable "
                "name; skipping to avoid poisoning the request (%r)", raw_schema,
            )
        elif schema["name"] not in existing_tool_names:
            defs.append({"type": "function", "function": schema})
            existing_tool_names.add(schema["name"])
    return defs


def _defers_memory_tool(defs: list[dict[str, Any]]) -> bool:
    """Whether tool_search is on and names one of ``defs`` in its defer set (incl.
    ``kanban.worker_defer_tools`` inside a dispatcher-owned worker)."""
    try:
        from tools.tool_search import load_config
        cfg = load_config()
    except Exception:
        logger.debug("tool_search config unavailable; memory tools stay eager", exc_info=True)
        return False
    return cfg.enabled != "off" and any(_tool_name(td) in cfg.effective_defer_tools for td in defs)


def assembled_memory_tool_defs(agent: Any) -> Optional[list[dict[str, Any]]]:
    """Memory-provider defs to pass as ``get_tool_definitions(extra_tool_defs=...)`` (surface
    rebuilds and the tool_search bridge catalog), or None when the agent appends them after
    assembly. None keeps a session that defers no memory tool on today's bytes and memo."""
    return getattr(agent, "_memory_tool_defs", None) if memory_tools_assembled(agent) else None


def deferred_memory_tool_names(agent: Any) -> frozenset:
    """Memory-provider tools this agent keeps behind tool_call (empty unless any are deferred)."""
    names = getattr(agent, "_deferred_memory_tool_names", None)
    return names if isinstance(names, frozenset) else frozenset()


def memory_tools_assembled(agent: Any) -> bool:
    """True once ``inject_memory_provider_tools`` routed the memory schemas through assembly
    (``is True``: a Mock agent's auto-attribute must not count)."""
    return getattr(agent, "_memory_tools_assembled", False) is True


def inject_memory_provider_tools(agent: Any) -> int:
    """Put external memory-provider tools on an agent tool surface; return how many.

    No memory tool named in the tool_search defer set: plain append after assembly (the
    historical bytes and order). Otherwise the surface is rebuilt with the memory schemas
    run through assembly, so the named ones move behind tool_search/tool_describe/tool_call
    and are listed in ``agent._deferred_memory_tool_names``.
    """
    defs = memory_provider_tool_defs(agent)
    if not defs:
        return 0
    if getattr(agent, "valid_tool_names", None) is None:
        agent.valid_tool_names = set()
    if not _defers_memory_tool(defs):
        for td in defs:
            agent.tools.append(td)
            agent.valid_tool_names.add(_tool_name(td))
        return len(defs)
    import model_tools
    from toolsets import agent_tool_surface
    agent.tools = agent_tool_surface(agent, model_tools.get_tool_definitions(
        enabled_toolsets=getattr(agent, "enabled_toolsets", None),
        disabled_toolsets=getattr(agent, "disabled_toolsets", None),
        quiet_mode=True, extra_tool_defs=defs) or [])
    agent.valid_tool_names = {_tool_name(t) for t in agent.tools}
    agent._memory_tools_assembled = True
    agent._deferred_memory_tool_names = frozenset(
        name for name in map(_tool_name, defs) if name not in agent.valid_tool_names)
    return len(defs)
