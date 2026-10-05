"""Data-class policy for Kanban workers' main-model provider routes.

This module needs only the standard library plus the repository's config loader
(PyYAML underneath), so standalone factory scripts can import it with the
repository on PYTHONPATH.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from hermes_cli.config_defaults import DEFAULT_CONFIG

_LOG = logging.getLogger(__name__)
_DEFAULT_POLICIES = DEFAULT_CONFIG["kanban"]["data_policies"]


class ProviderDenied(ValueError):
    """Raised when a provider route is not allowed for a data class."""


def _task_value(task_row: Any, name: str) -> Any:
    if task_row is None:
        return None
    if isinstance(task_row, Mapping):
        return task_row.get(name)
    try:
        return task_row[name]
    except (KeyError, TypeError, IndexError):
        return getattr(task_row, name, None)


def _load_policies() -> dict[str, Any]:
    """Load the profile-scoped policy through the canonical config loader.

    ``load_config_readonly`` deep-merges DEFAULT_CONFIG (spec defaults) with the user file,
    the managed-scope overlay and ``${VAR}`` expansion. Any read failure fails closed.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception as exc:
        raise ProviderDenied(f"could not read data-class policy: {exc}") from exc
    kanban = config.get("kanban", {}) or {}
    if not isinstance(kanban, Mapping):
        raise ProviderDenied("kanban config must be a mapping")
    raw = kanban.get("data_policies", _DEFAULT_POLICIES) or {}
    if not isinstance(raw, Mapping):
        raise ProviderDenied("kanban.data_policies must be a mapping")
    tenant_defaults = raw.get("tenant_defaults", {}) or {}
    if not isinstance(tenant_defaults, Mapping):
        raise ProviderDenied("kanban.data_policies.tenant_defaults must be a mapping")
    classes = raw.get("classes", {}) or {}
    if not isinstance(classes, Mapping):
        raise ProviderDenied("kanban.data_policies.classes must be a mapping")
    for name, value in classes.items():
        if not isinstance(value, Mapping):
            raise ProviderDenied(f"policy for data class {name!r} must be a mapping")
    return {
        "default": raw.get("default"),
        "tenant_defaults": dict(tenant_defaults),
        "classes": {str(name).strip().lower(): dict(value) for name, value in classes.items()},
    }


def _class_policy(data_class: str) -> dict[str, Any]:
    label = str(data_class or "").strip().lower()
    policies = _load_policies()
    classes = policies["classes"]
    policy = classes.get(label)
    if not label or not isinstance(policy, Mapping):
        raise ProviderDenied(f"unknown data class {label or '(empty)'!r}")
    return dict(policy)


def allowed_providers(data_class: str) -> list[str] | str:
    """Return a provider allowlist, or ``"any"`` for an unrestricted class."""
    raw = _class_policy(data_class).get("providers")
    if isinstance(raw, str) and raw.strip().lower() == "any":
        return "any"
    if isinstance(raw, (list, tuple, set)) and all(isinstance(item, str) for item in raw):
        return [item.strip().lower() for item in raw if item.strip()]
    raise ProviderDenied(f"invalid providers policy for data class {data_class!r}")


def fallback_allowed(data_class: str) -> bool:
    """Whether the class permits switching to a configured fallback provider."""
    return _class_policy(data_class).get("fallback") is True


def assert_fallback_allowed(provider: str, data_class: str) -> None:
    """Raise and record a refusal when the class disables fallback switching."""
    if not fallback_allowed(data_class):
        _deny(
            str(provider or "").strip().lower(), data_class,
            reason="fallback is disabled for this data class", phase="fallback",
        )


def _record_denial(provider: str, data_class: str, *, reason: str, phase: str, task_id: str | None = None) -> None:
    """Persist one ``provider_denied`` event for the current Kanban card, best-effort."""
    task_id = task_id or (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return
    payload = {
        "provider": str(provider or ""),
        "data_class": str(data_class or ""),
        "reason": reason,
        "phase": phase,
    }
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
        try:
            run_id = int(raw_run_id) if raw_run_id else None
        except ValueError:
            run_id = None
        with kbc.connect_closing() as conn:
            if not conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone():
                return
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "provider_denied", payload, run_id=run_id)
    except Exception as exc:  # A policy refusal remains effective if event persistence fails.
        _LOG.warning("Could not persist provider_denied event for Kanban task %s: %s", task_id, exc)


def _deny(provider: str, data_class: str, *, reason: str, phase: str, task_id: str | None = None) -> None:
    _record_denial(provider, data_class, reason=reason, phase=phase, task_id=task_id)
    message = f"provider {provider!r} denied for data class {data_class!r}: {reason}"
    _LOG.warning(message)
    raise ProviderDenied(message)


def assert_provider_allowed(provider: str, data_class: str, *, phase: str = "main") -> None:
    """Raise ``ProviderDenied`` unless provider is permitted by the class policy."""
    allowed = allowed_providers(data_class)
    candidate = str(provider or "").strip().lower()
    if allowed != "any" and candidate not in allowed:
        _deny(candidate, data_class, reason="provider is not in the class allowlist", phase=phase)


def resolve_data_class(task_row: Any) -> str:
    """Resolve explicit task class, then tenant default, then policy default.

    Every resolved label is validated against ``classes``; there is no
    unknown-or-missing-to-unrestricted path.
    """
    explicit = _task_value(task_row, "data_class")
    tenant = _task_value(task_row, "tenant")
    policies = _load_policies()
    label = str(explicit or "").strip().lower()
    if not label:
        tenant_defaults = policies["tenant_defaults"]
        label = str(tenant_defaults.get(str(tenant or "").strip(), "") or "").strip().lower()
    if not label:
        label = str(policies.get("default") or "").strip().lower()
    if label not in policies["classes"]:
        _record_denial(
            "(unresolved)", label, reason="unknown data class", phase="resolution",
            task_id=str(_task_value(task_row, "id") or "") or None,
        )
        raise ProviderDenied(f"unknown data class {label or '(empty)'!r}")
    return label


def is_restricted(data_class: str) -> bool:
    """A class is restricted when it narrows providers or disables fallback."""
    return allowed_providers(data_class) != "any" or not fallback_allowed(data_class)


def inherit_data_class(child_row: Any, parent_rows: list[Any]) -> str | None:
    """Return the class to persist on a new card, never looser than its parents.

    Without parents the explicit value is validated and kept (NULL stays NULL so the
    tenant/default resolution applies at dispatch). With parents the result is always an
    explicit label: the child's own class when it matches or the parents are unrestricted,
    else the parents' restricted class. An explicit unrestricted child under a restricted
    parent, or two different restricted classes, raise ``ProviderDenied``.
    """
    explicit = str(_task_value(child_row, "data_class") or "").strip().lower() or None
    own = resolve_data_class(child_row)
    if not parent_rows:
        return explicit
    result = own
    for inherited in {resolve_data_class(parent) for parent in parent_rows}:
        if inherited == result or not is_restricted(inherited):
            continue
        if is_restricted(result):
            raise ProviderDenied(
                f"child data class {result!r} conflicts with parent data class {inherited!r}"
            )
        if explicit is not None:
            raise ProviderDenied(
                f"child data class {explicit!r} would loosen parent data class {inherited!r}"
            )
        result = inherited
    return result


def current_data_class() -> str:
    """Resolve and validate the current worker's policy class.

    Kanban workers must receive the dispatcher-resolved environment value. A
    missing value on a worker retries resolution from its durable task row;
    ordinary non-Kanban sessions retain the ``internal`` default.
    """
    env_class = (os.environ.get("HERMES_DATA_CLASS") or "").strip()
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if env_class:
        try:
            _class_policy(env_class)
        except ProviderDenied as exc:
            _record_denial(
                "(unresolved)", env_class, reason=str(exc), phase="resolution", task_id=task_id or None,
            )
            raise
        return env_class.lower()
    if task_id:
        try:
            from hermes_cli import kanban_db as kb
            from hermes_cli import kanban_db_connect as kbc

            with kbc.connect_closing() as conn:
                task = kb.get_task(conn, task_id)
            if task is None:
                raise ProviderDenied(f"Kanban worker task {task_id!r} is missing")
            return resolve_data_class(task)
        except ProviderDenied:
            raise
        except Exception as exc:
            raise ProviderDenied(f"could not resolve data class for Kanban task {task_id!r}: {exc}") from exc
    return resolve_data_class(None)
