"""SOUL.md sections fenced by ``<!-- chat-only -->`` / ``<!-- /chat-only -->`` reach chat sessions with the
marker lines removed (byte-identical to the unmarked file) and are dropped for the dispatcher-owned Kanban
worker, whose prompt carries ~75k API calls a month."""

from types import SimpleNamespace

from agent import system_prompt
from agent.delegation_context import non_dispatcher_owned_context

PLAIN = (
    "# Me\n\nBe kind.\n\n"
    "## Voice\n\nFirst line = next action.\n\n"
    "## Tests\n\n- Simple retrieval: do it directly.\n- Fix root causes.\n\n"
    "## Absolute\n\n- Private things stay private.\n"
)
MARKED = (
    "# Me\n\nBe kind.\n\n"
    "<!-- chat-only -->\n## Voice\n\nFirst line = next action.\n<!-- /chat-only -->\n\n"
    "## Tests\n\n<!-- chat-only -->\n- Simple retrieval: do it directly.\n<!-- /chat-only -->\n- Fix root causes.\n\n"
    "## Absolute\n\n- Private things stay private.\n"
)
SLIM = "# Me\n\nBe kind.\n\n## Tests\n\n- Fix root causes.\n\n## Absolute\n\n- Private things stay private."


def _identity(tmp_path, soul: str) -> str:
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    (home / "SOUL.md").write_text(soul, encoding="utf-8")
    agent = SimpleNamespace(load_soul_identity=False, skip_context_files=False,
                            _session_db=SimpleNamespace(db_path=str(home / "state.db")))
    parts, loaded = system_prompt._identity_parts(agent, None)
    assert loaded
    return parts[0]


def test_chat_session_gets_full_soul_byte_identical(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _identity(tmp_path, MARKED) == _identity(tmp_path, PLAIN) == PLAIN.strip()


def test_kanban_worker_gets_slim_soul(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    assert _identity(tmp_path, MARKED) == SLIM
    assert _identity(tmp_path, MARKED) == SLIM  # same bytes on every rebuild: the cached prefix holds


def test_unmarked_soul_unchanged_for_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    assert _identity(tmp_path, PLAIN) == PLAIN.strip()


def test_non_owner_inside_worker_keeps_full_soul(tmp_path, monkeypatch):
    """A cron run inside a worker process inherits the env var but does not own the task."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    with non_dispatcher_owned_context():
        assert _identity(tmp_path, MARKED) == PLAIN.strip()


def test_unbalanced_markers_keep_every_section(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    broken = PLAIN.replace("## Voice\n", "<!-- chat-only -->\n## Voice\n")
    assert _identity(tmp_path, broken) == PLAIN.strip()
