"""patch must not write a display mask over the secret it masks (intake t_fd0d9703).

read_file shows secret-shaped literals masked (the file-read sentinel, ``***``, a head/tail
mask). A model that copies a masked block into ``patch`` still gets a fuzzy match on the real
lines, and the replacement used to write the mask onto disk in place of the real value.
Fixture secrets and sentinels are built at runtime so this source holds no secret-shaped literal.
"""

import json

import pytest

from agent.redact import redact_sensitive_text

KEY = "sk-" + "ant-" + "api03-" + "abcdefghij" * 3
OAT = "sk-" + "ant-" + "oat01-" + "abcdef"
LITERAL_SENTINEL = "\u00abredacted:ghp_\u2026\u00bb"

SOURCE = (
    "class TestIsOAuthToken:\n"
    "    def test_setup_token(self):\n"
    f'        assert _is_oauth_token("{OAT}") is True\n'
    "\n"
    "    def test_api_key(self):\n"
    f'        assert _is_oauth_token("{KEY}") is False\n'
    "\n"
    "\n"
    "def test_next():\n"
    "    pass\n"
)


@pytest.fixture(autouse=True)
def _redaction_on(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.fixture
def victim(tmp_path):
    path = tmp_path / "victim_test.py"
    path.write_text(SOURCE)
    return path


def _patch(**kwargs):
    from tools.file_tools import patch_tool
    return json.loads(patch_tool(task_id="t_fd0d9703", **kwargs))


def _shown_lines(path):
    """The lines exactly as read_file hands them to the model."""
    from tools.file_tools import read_file_tool
    result = json.loads(read_file_tool(str(path), task_id="t_fd0d9703"))
    return [line.split("|", 1)[1] for line in result["content"].split("\n") if "|" in line]


def _masked_block(path):
    """Shown lines 5-9: the masked secret line in the middle, first and last lines exact."""
    old = "\n".join(_shown_lines(path)[4:9])
    assert KEY not in old and "redacted" in old
    return old, old.replace("def test_next():", "def test_next_renamed():")


class TestRefusesMaskWriteBack:
    def test_replace_with_sentinel_in_middle_line(self, victim):
        old, new = _masked_block(victim)
        result = _patch(mode="replace", path=str(victim), old_string=old, new_string=new)
        assert not result.get("success")
        assert "will not write the mask" in result["error"]
        assert victim.read_text() == SOURCE

    def test_v4a_update_hunk(self, victim):
        shown = _shown_lines(victim)
        hunk = "\n".join(f" {line}" for line in shown[4:8])
        patch = (f"*** Begin Patch\n*** Update File: {victim}\n@@\n{hunk}\n"
                 "-def test_next():\n+def test_next_renamed():\n*** End Patch")
        result = _patch(mode="patch", patch=patch)
        assert not result.get("success")
        assert "will not write the mask" in result["error"]
        assert victim.read_text() == SOURCE

    def test_head_tail_tool_result_mask(self, victim):
        view = redact_sensitive_text(SOURCE).split("\n")
        old = "\n".join(view[4:9])
        assert KEY not in old and "..." in old
        new = old.replace("def test_next():", "def test_next_renamed():")
        result = _patch(mode="replace", path=str(victim), old_string=old, new_string=new)
        assert not result.get("success")
        assert "will not write the mask" in result["error"]
        assert victim.read_text() == SOURCE

    def test_masked_line_moved_elsewhere_in_the_edit(self):
        from tools.file_operations_common import display_placeholder_error
        original = "a = 1\n" f'TOKEN = "{KEY}"\n' "b = 2\nc = 3\n"
        masked = redact_sensitive_text(original, file_read=True).split("\n")[1]
        new_content = "a = 1\nb = 2\n" + masked + "\nc = 3\n"
        assert display_placeholder_error(original, new_content, masked, "/w/conf.py")


class TestAllowsLegitimateEdits:
    def test_deleting_the_secret_line(self, victim):
        old, new = _masked_block(victim)
        new = "\n".join(line for line in new.split("\n") if "redacted" not in line)
        result = _patch(mode="replace", path=str(victim), old_string=old, new_string=new)
        assert result.get("success"), result
        text = victim.read_text()
        assert "def test_next_renamed():" in text and "redacted" not in text

    def test_literal_sentinel_in_file_without_secrets(self, tmp_path):
        path = tmp_path / "test_sentinels.py"
        path.write_text(f'A = "{LITERAL_SENTINEL}"\nB = 1\n')
        result = _patch(mode="replace", path=str(path), old_string="B = 1",
                        new_string=f'B = "{LITERAL_SENTINEL}"')
        assert result.get("success"), result
        assert path.read_text().count(LITERAL_SENTINEL) == 2

    @pytest.mark.parametrize("old, new", [
        ("Some *bold* text", "Some ***bold italic*** text"),
        ("see a..b", "see foo...bar"),
    ])
    def test_markdown_masks_in_file_without_secrets(self, tmp_path, old, new):
        path = tmp_path / "notes.md"
        path.write_text(f"# Notes\n\n{old}\n")
        result = _patch(mode="replace", path=str(path), old_string=old, new_string=new)
        assert result.get("success"), result
        assert new in path.read_text()

    def test_markdown_stars_in_file_whose_secret_renders_as_stars(self, tmp_path):
        bearer = "Authorization: Bearer " + "sk-" + "live-" + "abcdefghij" * 3
        path = tmp_path / "api.md"
        path.write_text(f"# API\n\n{bearer}\n\nSome *bold* text\n")
        assert "***" in redact_sensitive_text(path.read_text(), file_read=True)
        result = _patch(mode="replace", path=str(path), old_string="Some *bold* text",
                        new_string="Some ***bold italic*** text")
        assert result.get("success"), result
        text = path.read_text()
        assert bearer in text and "Some ***bold italic*** text" in text

    def test_redaction_disabled(self, victim, monkeypatch):
        old, new = _masked_block(victim)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        result = _patch(mode="replace", path=str(victim), old_string=old, new_string=new)
        assert result.get("success"), result

    def test_real_key_verbatim_is_kept(self, victim):
        real = "\n".join(SOURCE.split("\n")[4:9])
        result = _patch(mode="replace", path=str(victim), old_string=real,
                        new_string=real.replace("def test_next():", "def test_next_renamed():"))
        assert result.get("success"), result
        text = victim.read_text()
        assert KEY in text and "def test_next_renamed():" in text
