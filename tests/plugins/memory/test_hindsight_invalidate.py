"""hindsight_invalidate and hindsight_retain(supersedes=...) retire a recalled fact (t_337039ce).

Regression for the 2026-11-24 23:44:40 incident: the chat agent retained "This supersedes memories
saying control.curiositypilot.com was taken offline" and told the user it had marked the old notes
as replaced, but the tool surface had no invalidate path and recall showed no ids, so the stale fact
stayed live. These drive the real provider against a fake bank that records call order.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider

STALE = "4b2c5e98-9709-46ba-a5ea-0abb512b512f"
FINDING = "11111111-2222-4333-8444-555555555555"
DECISION = "393b5e02-c037-4190-ba5c-d8ba4ee26827"
OBSERVATION = "aaaaaaaa-0000-4000-8000-000000000001"
DONE = "bbbbbbbb-0000-4000-8000-000000000002"
INCIDENT_TEXT = (
    "CORRECTION (2026-11-24, verified by MB): control.curiositypilot.com is ONLINE and reachable "
    "behind Cloudflare Access. This supersedes memories saying control.curiositypilot.com was "
    "taken offline on 2026-11-22."
)


class FakeBank:
    """The slice of the Hindsight client the tools use; every call lands in ``calls``."""

    def __init__(self):
        self.calls = []
        self.facts = {
            STALE: {"id": STALE, "type": "world", "state": "valid", "entities": [],
                    "text": "control.curiositypilot.com and the public site were taken offline on 22 November 2026"},
            FINDING: {"id": FINDING, "type": "world", "state": "valid",
                      "entities": ["status:finding", "project:movis"], "text": "a finding"},
            DECISION: {"id": DECISION, "type": "world", "state": "valid",
                       "entities": ["status:decision", "project:movis"], "text": "a decision"},
            OBSERVATION: {"id": OBSERVATION, "type": "observation", "state": "valid", "entities": [],
                          "text": "an observation"},
            DONE: {"id": DONE, "type": "world", "state": "invalidated", "entities": [], "text": "gone"},
        }
        self.fail_patch = False
        self.memory = SimpleNamespace(get_memory=self._get, update_memory=self._patch)
        self.arecall = AsyncMock(side_effect=self._recall)
        self.aretain_batch = AsyncMock(side_effect=self._retain)
        self.aclose = AsyncMock()

    async def _get(self, bank_id, memory_id):
        self.calls.append(("get", memory_id))
        if memory_id not in self.facts:
            raise LookupError(f"404 {memory_id}")
        return dict(self.facts[memory_id])

    async def _patch(self, bank_id, memory_id, request):
        self.calls.append(("patch", memory_id, request))
        if self.fail_patch:
            raise ConnectionError("hindsight down")
        self.facts[memory_id]["state"] = request["state"]
        return {"success": True}

    async def _recall(self, **kwargs):
        self.calls.append(("recall",))
        return SimpleNamespace(results=[SimpleNamespace(id=f["id"], text=f["text"], type=f["type"],
                                                        mentioned_at=None, scores=None)
                                        for f in self.facts.values()])

    async def _retain(self, **kwargs):
        self.calls.append(("retain", kwargs["items"][0]["content"]))
        return SimpleNamespace(operation_id=None)

    def patches(self):
        return [c for c in self.calls if c[0] == "patch"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    path = tmp_path / "hindsight" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"mode": "cloud", "apiKey": "k", "api_url": "http://localhost:9999", "bank_id": "b",
                                "auto_retain": False, "recall_sync": True}))
    p = HindsightMemoryProvider()
    p.initialize(session_id="sess-1", hermes_home=str(tmp_path), platform="cli")
    bank = FakeBank()
    p._client = p._reflect_client = bank
    return p, bank


def _call(p, tool, args):
    return json.loads(p.handle_tool_call(tool, args))


def _recall(p):
    return _call(p, "hindsight_recall", {"query": "control.curiositypilot.com"})["result"]


def test_tool_recall_shows_id8_and_auto_recall_does_not(setup):
    p, _ = setup
    assert f"1. {{{STALE[:8]}}} control.curiositypilot.com" in _recall(p)
    injected = p.prefetch("control.curiositypilot.com")
    assert "control.curiositypilot.com" in injected and STALE[:8] not in injected


def test_id8_from_recall_resolves_and_patches_with_agent_reason(setup):
    p, bank = setup
    _recall(p)
    out = _call(p, "hindsight_invalidate", {"ids": [STALE[:8]], "reason": "control is online"})["result"]
    assert out.splitlines()[0].startswith(f"invalidated {STALE[:8]}: control.curiositypilot.com")
    assert "reversible: PATCH state=valid" in out
    [(_, memory_id, request)] = bank.patches()
    assert memory_id == STALE and request["state"] == "invalidated"
    assert request["reason"] == "agent-correction: control is online (session sess-1)"


def test_full_uuid_works_without_recall(setup):
    p, bank = setup
    out = _call(p, "hindsight_invalidate", {"ids": [STALE], "reason": "r"})["result"]
    assert out.startswith(f"invalidated {STALE[:8]}") and [c[1] for c in bank.patches()] == [STALE]


def test_unknown_id8_is_refused_without_any_bank_call(setup):
    p, bank = setup
    out = _call(p, "hindsight_invalidate", {"ids": ["deadbeef"], "reason": "r"})["result"]
    assert "refused deadbeef" in out and "run hindsight_recall first" in out
    assert bank.calls == []


def test_observation_and_decision_are_refused_finding_and_unlabelled_are_patched(setup):
    p, bank = setup
    out = _call(p, "hindsight_invalidate", {"ids": [OBSERVATION, DECISION, FINDING, STALE], "reason": "r"})["result"]
    lines = out.splitlines()
    assert lines[0].startswith(f"refused {OBSERVATION[:8]}: observation")
    assert lines[1].startswith(f"refused {DECISION[:8]}: status:decision")
    assert f"memory-system retire {DECISION}" in lines[1]
    assert lines[2].startswith(f"invalidated {FINDING[:8]}") and lines[3].startswith(f"invalidated {STALE[:8]}")
    assert [c[1] for c in bank.patches()] == [FINDING, STALE]
    assert all(c[2]["reason"].startswith("agent-correction:") for c in bank.patches())


def test_missing_reason_is_a_tool_error(setup):
    p, bank = setup
    out = _call(p, "hindsight_invalidate", {"ids": [STALE]})
    assert "error" in out and "reason" in out["error"] and bank.calls == []


def test_retain_supersedes_patches_before_retaining(setup):
    """Incident replay: the 23:44:40 correction with supersedes on the recalled stale echo."""
    p, bank = setup
    _recall(p)
    out = _call(p, "hindsight_retain", {"content": INCIDENT_TEXT, "supersedes": [STALE[:8]]})["result"]
    order = [c[0] for c in bank.calls if c[0] in ("patch", "retain")]
    assert order == ["patch", "retain"] and bank.facts[STALE]["state"] == "invalidated"
    assert out.startswith("Memory stored successfully.") and f"invalidated {STALE[:8]}" in out


@pytest.mark.parametrize("ref,fail_patch,outcome", [
    (STALE, True, "failed"),          # PATCH error: the stale fact stays live
    (DECISION, False, "refused"),     # protected decision: the merge could rewrite it
    ("deadbeef", False, "refused"),   # unknown id8
])
def test_retain_is_skipped_unless_every_superseded_id_is_invalidated(setup, ref, fail_patch, outcome):
    p, bank = setup
    bank.fail_patch = fail_patch
    out = _call(p, "hindsight_retain", {"content": INCIDENT_TEXT, "supersedes": [ref]})["result"]
    assert out.startswith("Memory NOT stored") and f"{outcome} {ref[:8]}" in out
    assert not [c for c in bank.calls if c[0] == "retain"]


def test_already_invalidated_supersedes_id_counts_so_a_retry_retains(setup):
    p, bank = setup
    out = _call(p, "hindsight_retain", {"content": "x", "supersedes": [DONE]})["result"]
    assert out.startswith("Memory stored successfully.") and "(already)" in out
    assert bank.patches() == [] and bank.calls[-1] == ("retain", "x")


def test_session_switch_forgets_recalled_ids(setup):
    p, bank = setup
    _recall(p)
    p.on_session_switch("sess-2")
    out = _call(p, "hindsight_invalidate", {"ids": [STALE[:8]], "reason": "r"})["result"]
    assert "run hindsight_recall first" in out and bank.patches() == []
