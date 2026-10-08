"""Tests for the on_session_switch hook and session_id propagation.

Covers #6672: memory providers must be notified when AIAgent.session_id
rotates mid-process (via /resume, /branch, /reset, /new, or context
compression). Without the notification, providers that cache per-session
state in initialize() (Hindsight, and any plugin that stores session_id
for scoped writes) keep writing into the old session's record.
"""



import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Provider that records every lifecycle call for assertion."""

    def __init__(self, name="rec"):
        self._name = name
        self.switch_calls: list[dict] = []
        self.sync_calls: list[dict] = []
        self.queue_calls: list[dict] = []
        self.initialize_calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:  # pragma: no cover - unused
        return True

    def initialize(self, session_id, **kwargs):
        self.initialize_calls.append({"session_id": session_id, **kwargs})

    def get_tool_schemas(self):
        return []

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        self.sync_calls.append(
            {"user": user_content, "asst": assistant_content, "session_id": session_id}
        )

    def queue_prefetch(self, query, *, session_id=""):
        self.queue_calls.append({"query": query, "session_id": session_id})

    def on_session_switch(
        self,
        new_session_id,
        *,
        parent_session_id="",
        reset=False,
        **kwargs,
    ):
        self.switch_calls.append(
            {
                "new": new_session_id,
                "parent": parent_session_id,
                "reset": reset,
                "extra": kwargs,
            }
        )


# ---------------------------------------------------------------------------
# MemoryManager.on_session_switch — fan-out
# ---------------------------------------------------------------------------


def test_manager_fans_out_to_all_providers():
    mm = MemoryManager()
    # Only one external provider is allowed; use the builtin slot for p1.
    p1 = _RecordingProvider(name="builtin")
    p2 = _RecordingProvider(name="hindsight")
    mm.add_provider(p1)
    mm.add_provider(p2)

    mm.on_session_switch("new-sid", parent_session_id="old-sid", reset=False, reason="resume")

    assert len(p1.switch_calls) == 1
    assert len(p2.switch_calls) == 1
    for call in (p1.switch_calls[0], p2.switch_calls[0]):
        assert call["new"] == "new-sid"
        assert call["parent"] == "old-sid"
        assert call["reset"] is False
        assert call["extra"] == {"reason": "resume"}


def test_manager_isolates_provider_failures():
    """A provider that raises must not block other providers."""

    class _Broken(_RecordingProvider):
        def on_session_switch(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    mm = MemoryManager()
    # MemoryManager rejects a second external provider, so pair broken
    # (builtin slot) with a good external one.
    broken = _Broken(name="builtin")
    good = _RecordingProvider(name="good")
    mm.add_provider(broken)
    mm.add_provider(good)

    # Must not raise — exceptions in one provider are swallowed + logged
    mm.on_session_switch("new-sid", parent_session_id="old-sid")
    assert len(good.switch_calls) == 1
    assert good.switch_calls[0]["new"] == "new-sid"


# ---------------------------------------------------------------------------
# MemoryManager.sync_all / queue_prefetch_all — session_id propagation
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Hindsight reference implementation — state-flush semantics
# ---------------------------------------------------------------------------


def _make_hindsight_provider():
    """Build a bare HindsightMemoryProvider that skips network setup.

    HindsightMemoryProvider.__init__ is pure config and sets up default
    attributes without I/O or background threads. We instantiate normally
    and override only the session state and stubs needed for testing
    on_session_switch.
    """
    hindsight_mod = pytest.importorskip("plugins.memory.hindsight")
    provider = hindsight_mod.HindsightMemoryProvider()
    provider._session_id = "old-sid"
    provider._parent_session_id = ""
    provider._document_id = "old-sid-20260101_000000_000000"
    provider._session_turns = ["turn-1", "turn-2"]
    provider._turn_counter = 2
    provider._turn_index = 2
    provider._retain_context = "test-context"
    provider._retain_async = False
    provider._bank_id = "test-bank"
    # Writer queue infra the flush-on-switch path enqueues onto. We stub
    # _ensure_writer / _register_atexit so no real thread is spawned;
    # tests exercising flush delivery live in
    # tests/plugins/memory/test_hindsight_provider.py where the full
    # writer-queue wiring is in place.
    provider._atexit_registered = True
    provider._ensure_writer = lambda: None
    provider._register_atexit = lambda: None
    # Stub _resolve_retain_target so tests don't actually probe the API
    # (_mode keeps its __init__ default). Real probe behavior is
    # exercised by tests in tests/plugins/memory/test_hindsight_provider.py.
    provider._resolve_retain_target = lambda fb: (fb, None)
    # Stub the network-touching helper so any enqueued flush closure is
    # a no-op if ever drained in a unit test.
    provider._run_hindsight_operation = lambda _op: None
    return provider


def test_hindsight_on_session_switch_updates_session_id_and_mints_fresh_doc():
    provider = _make_hindsight_provider()
    old_doc = provider._document_id

    provider.on_session_switch(
        "new-sid", parent_session_id="old-sid", reset=False, reason="resume"
    )

    assert provider._session_id == "new-sid"
    assert provider._parent_session_id == "old-sid"
    # Document id MUST be fresh — else next retain overwrites old session doc
    assert provider._document_id != old_doc
    assert provider._document_id.startswith("new-sid-")


def test_hindsight_on_session_switch_clears_turn_buffers():
    """Accumulated _session_turns must not leak into the next session.

    Hindsight batches turns under a single _document_id. If the buffer
    isn't cleared on switch, the next retain under the new _document_id
    flushes turns that belong to the previous session.
    """
    provider = _make_hindsight_provider()
    provider.on_session_switch("new-sid", parent_session_id="old-sid")
    assert provider._session_turns == []
    assert provider._turn_counter == 0
    assert provider._turn_index == 0






