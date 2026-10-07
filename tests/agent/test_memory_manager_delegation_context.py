"""MemoryManager.delegation_context: parent-side recall for delegate_task children (t_2354171e)."""
import threading
import time

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _Provider(MemoryProvider):
    def __init__(self, name, fn=None):
        self._name = name
        self._fn = fn or (lambda q: "")

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def delegation_context(self, query):
        return self._fn(query)


def _manager(provider, timeout=8.0):
    m = MemoryManager(external_prefetch_timeout=timeout)
    m.add_provider(provider)
    return m


def test_wraps_external_result_without_authoritative_note():
    m = _manager(_Provider("ext", lambda q: f"- fact about {q}"))
    out = m.delegation_context("goal")
    assert out.startswith("<memory-context>") and out.endswith("</memory-context>")
    assert "- fact about goal" in out
    assert "authoritative" not in out.lower()
    assert "verify" in out.lower()


def test_builtin_is_skipped():
    called = []
    m = _manager(_Provider("builtin", lambda q: called.append(q) or "x"))
    assert m.delegation_context("goal") == ""
    assert called == []


def test_base_provider_default_is_empty():
    class Plain(_Provider):
        delegation_context = MemoryProvider.delegation_context
    assert _manager(Plain("ext")).delegation_context("goal") == ""


def test_error_returns_empty():
    def boom(_q):
        raise RuntimeError("down")
    assert _manager(_Provider("ext", boom)).delegation_context("goal") == ""


def test_timeout_returns_empty_and_does_not_block_parent_prefetch():
    release = threading.Event()

    def slow(_q):
        release.wait(5)
        return "- late"
    m = _manager(_Provider("ext", slow), timeout=0.2)
    start = time.monotonic()
    assert m.delegation_context("goal") == ""
    assert time.monotonic() - start < 1.0
    # A stuck child recall must not register as a running prefetch (the parent would skip its next turn).
    assert m._external_prefetch_threads == {}
    release.set()


def test_explicit_timeout_overrides_default():
    def slow(_q):
        time.sleep(0.5)
        return "- late"
    m = _manager(_Provider("ext", slow), timeout=8.0)
    start = time.monotonic()
    assert m.delegation_context("goal", timeout=0.1) == ""
    assert time.monotonic() - start < 0.4
