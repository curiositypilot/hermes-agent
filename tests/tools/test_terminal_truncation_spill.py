"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import json
import os
from pathlib import Path

import pytest

from tools.terminal_tool import terminal_tool


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    }})
    return tmp_path


class TestTruncationSpill:
    @pytest.mark.platforms("linux")
    def test_truncated_output_has_metadata_and_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('marker_head'); [print(f'row_{i}', 'x'*80) for i in range(200)]; print('marker_tail')\"",
            task_id="t-spill-1"))
        assert r["exit_code"] == 0
        assert "OUTPUT TRUNCATED" in r["output"]
        assert r["output_total_chars"] > 2000
        p = Path(r["full_output_path"])
        assert p.exists()
        full = p.read_text()
        assert "marker_head" in full and "marker_tail" in full
        # The spill contains rows that were cut from the visible window.
        assert "row_100 " in full
        assert "read_file" in r["truncation_note"]

    def test_small_output_has_no_metadata(self, small_cap):
        r = json.loads(terminal_tool("echo tiny", task_id="t-spill-2"))
        assert r["exit_code"] == 0
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    @pytest.mark.platforms("linux")
    def test_spill_is_redacted(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('sk-proj-' + 'a1B2c3D4e5F6g7H8i9J0' * 3); [print('pad', 'y'*90) for i in range(200)]\"",
            task_id="t-spill-3"))
        p = Path(r["full_output_path"])
        full = p.read_text()
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in full

    def test_old_spills_cleaned(self, small_cap, tmp_path):
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        stale = spill_dir / "out-1-2-dead.log"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        json.loads(terminal_tool(
            "python3 -c \"[print('z'*90) for i in range(200)]\"", task_id="t-spill-4"))
        assert not stale.exists()

    @pytest.mark.platforms("linux")
    def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()


    def test_truncated_json_is_detectable_and_full_spill_parses(self, small_cap):
        """execute_code's terminal() helper returns this result: a caller detects the head+tail
        cut by ``full_output_path``, and that file holds exactly the command's output (no
        cwd marker), so a big JSON dump parses from it while ``output`` does not."""
        payload = [{"id": i, "body": "line\nnext " + "x" * 60} for i in range(200)]
        script = "import json,sys; json.dump(json.load(sys.stdin), sys.stdout)"
        r = json.loads(terminal_tool(
            f"printf '%s' '{json.dumps(payload)}' | python3 -c '{script}'", task_id="t-spill-json"))
        assert r["exit_code"] == 0
        with pytest.raises(json.JSONDecodeError):
            json.loads(r["output"])
        assert "OUTPUT TRUNCATED" in r["output"]
        assert json.loads(Path(r["full_output_path"]).read_text()) == payload

    def test_execute_code_terminal_doc_names_the_recovery_field(self, small_cap):
        """The execute_code schema tells scripts which result key carries the full output."""
        from tools.code_execution_tool import _TOOL_DOC_LINES
        doc = dict(_TOOL_DOC_LINES)["terminal"]
        r = json.loads(terminal_tool("python3 -c \"[print('q'*90) for i in range(200)]\"",
                                     task_id="t-spill-doc"))
        recovery = [k for k in r if k not in ("output", "exit_code", "error") and k in doc]
        assert recovery == ["full_output_path"]
