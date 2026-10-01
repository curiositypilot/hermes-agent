"""MemoryManager strips slash-skill scaffolding for every provider.

When a user invokes a /skill or /bundle, Hermes expands the turn into a
model-facing message that embeds the full skill body. Feeding that verbatim to
memory providers pollutes their stores/embeddings with prompt scaffolding
instead of what the user actually asked. The strip lives once in MemoryManager
so it covers the whole provider fan-out — not per backend.

See: agent.skill_commands.extract_user_instruction_from_skill_message and
MemoryManager._strip_skill_scaffolding.
"""

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.skill_commands import extract_user_instruction_from_skill_message


_SINGLE_SKILL_TURN = (
    '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]\n\n"
    "# Skill Creator\n\n"
    "Large skill body that must not be searched or embedded.\n\n"
    "The user has provided the following instruction alongside the skill invocation: "
    "make a skill for release triage"
)

_BUNDLE_TURN = (
    '[IMPORTANT: The user has invoked the "backend-dev" skill bundle, '
    "loading 2 skills together. Treat every skill below as active guidance for this turn.]\n\n"
    "Bundle: backend-dev\n"
    "Skills loaded: test-driven-development, code-review\n\n"
    "User instruction: fix the failing retrieval test\n\n"
    '[Loaded as part of the "backend-dev" skill bundle.]\n\n'
    "Large bundled skill body that must not be searched or embedded."
)

_BARE_SKILL_TURN = (
    '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
    "you to follow its instructions. The full skill content is loaded below.]\n\n"
    "# Skill Creator\n\n"
    "Large skill body, no user instruction."
)


class _RecordingProvider(MemoryProvider):
    """Captures exactly what user text each fan-out method received."""

    _name = "recording"

    def __init__(self):
        self.prefetched = []
        self.queued = []
        self.synced = []

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        self.prefetched.append(query)
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        self.queued.append(query)

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.synced.append(user_content)

    def get_tool_schemas(self):
        return []


def _manager_with_recorder():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


class TestExtractUserInstruction:
    def test_non_string_returns_none(self):
        assert extract_user_instruction_from_skill_message(None) is None
        assert extract_user_instruction_from_skill_message(123) is None
        assert extract_user_instruction_from_skill_message([{"text": "hi"}]) is None



    def test_bundle_with_instruction(self):
        assert (
            extract_user_instruction_from_skill_message(_BUNDLE_TURN)
            == "fix the failing retrieval test"
        )




class TestMemoryManagerStripsScaffolding:

    def test_prefetch_all_skips_bare_skill(self):
        mgr, provider = _manager_with_recorder()
        result = mgr.prefetch_all(_BARE_SKILL_TURN)
        assert result == ""
        assert provider.prefetched == []

    def test_queue_prefetch_all_strips_bundle(self):
        mgr, provider = _manager_with_recorder()
        mgr.queue_prefetch_all(_BUNDLE_TURN)
        mgr.flush_pending(timeout=5.0)
        assert provider.queued == ["fix the failing retrieval test"]



    def test_sync_all_skips_bare_skill(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(_BARE_SKILL_TURN, "Done.")
        mgr.flush_pending(timeout=5.0)
        assert provider.synced == []



# ---------------------------------------------------------------------------
# Auto-loaded skill blocks (gateway topic binding / skills.auto_load)
# ---------------------------------------------------------------------------

_CRON_BRIEF_REPLY = (
    '[Replying to: "Cronjob Response: kanban-orchestrator\n(job_id: 580e25b75ad8)\n-------------\n\n'
    "2 decisions needed\n1. Fork·land (t_bc22f8eb): commit the routing code? Rec: yes.\n"
    "Reply e.g. '1 yes, 2 no'.\"]"
)
_SKILL_BODY = (
    "---\nname: kanban-dispatch\ndescription: \"Use when distributing work.\"\n---\n\n"
    "# Kanban dispatch\n\nGateway-restarting work is interactive: a worker cannot restart it.\n"
    "Config changes apply on the next tick, no restart."
)


def _auto_load_block(tmp_path, name="kanban-dispatch", body=_SKILL_BODY, *, with_files=True, setup=False):
    """A real gateway auto-load block, built with the production builders."""
    from agent.skill_commands import _build_skill_message, auto_load_activation_note

    skill_dir = tmp_path / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    if with_files:
        (skill_dir / "scripts" / "dispatch.py").write_text("print('x')\n")
    loaded = {"name": name, "content": body}
    if setup:
        loaded["setup_skipped"] = True
    return _build_skill_message(loaded, skill_dir, auto_load_activation_note(name))


def _gateway_turn(blocks, user_text):
    """``_hmwa_auto_load_skills`` shape: blocks then the user's (reply-prefixed) text."""
    return "\n\n".join([*blocks, user_text])


class TestAutoLoadedSkillStrip:
    def test_first_turn_keeps_cron_quote_and_user_text(self, tmp_path):
        from agent.skill_commands import extract_user_text_for_memory

        user_text = f"{_CRON_BRIEF_REPLY}\n\n1. please do this yourself\n2. no"
        turn = _gateway_turn([_auto_load_block(tmp_path)], user_text)
        assert "Gateway-restarting" in turn  # the model-facing message carries the skill

        cleaned = extract_user_text_for_memory(turn)
        assert cleaned == user_text
        assert "Kanban dispatch" not in cleaned and "Skill directory" not in cleaned

    def test_bundle_of_blocks_all_stripped(self, tmp_path):
        from agent.skill_commands import extract_user_text_for_memory

        blocks = [
            _auto_load_block(tmp_path, "kanban-dispatch"),
            _auto_load_block(tmp_path, "sdlc-review", "# Review\n\nAlways verify.", with_files=False, setup=True),
        ]
        assert extract_user_text_for_memory(_gateway_turn(blocks, "any blockers?")) == "any blockers?"

    def test_block_with_no_user_text_is_skipped(self, tmp_path):
        from agent.skill_commands import extract_user_text_for_memory

        assert extract_user_text_for_memory(_auto_load_block(tmp_path)) is None

    def test_config_auto_load_note_is_recognised(self, tmp_path):
        from agent.skill_commands import _build_skill_message, strip_auto_loaded_skill_blocks

        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        note = ('[IMPORTANT: The "s" skill is auto-loaded via config (skills.auto_load). '
                "Treat its instructions as active guidance for the duration of this session unless "
                "the user overrides them.]")
        block = _build_skill_message({"name": "s", "content": "body"}, skill_dir, note)
        assert strip_auto_loaded_skill_blocks(block + "\n\nhello") == "hello"

    def test_plain_text_and_mentions_are_untouched(self):
        from agent.skill_commands import extract_user_text_for_memory

        text = 'why does it say [IMPORTANT: The "x" skill is auto-loaded] in my prompt?'
        assert extract_user_text_for_memory(text) == text
        assert extract_user_text_for_memory("hello") == "hello"

    def test_memory_manager_fan_out_strips_auto_load(self, tmp_path):
        mgr, provider = _manager_with_recorder()
        user_text = f"{_CRON_BRIEF_REPLY}\n\n1. yes"
        turn = _gateway_turn([_auto_load_block(tmp_path)], user_text)

        mgr.sync_all(turn, "Done.")
        mgr.queue_prefetch_all(turn)
        mgr.prefetch_all(turn)
        mgr.flush_pending(timeout=5.0)

        assert provider.synced == [user_text]
        assert provider.queued == [user_text]
        assert provider.prefetched == [user_text]
