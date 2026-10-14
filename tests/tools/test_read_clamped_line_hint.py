"""The per-line clamp names the clamped lines and the command that reads the rest.

Evidence case (2026-10-14, t_67dedc8f): a 3,100-char one-line deal log was clamped at
2,000 chars with a bare ``truncated_lines: true``; the model guessed ``cut -c2400-`` to
resume and skipped the verdict sitting in chars 2,000-2,400. The hint must say which
lines were clamped, at what width, and give a character-exact recipe (GNU ``cut -c``
counts bytes, so it drifts on non-ASCII text such as "€" and "ó").
"""

import json
import shlex
import subprocess
import sys

import pytest

from tools.file_tools import _add_clamped_lines_hint, _clamped_line_numbers, read_file_tool
from tools.tool_output_limits import _reset_tool_output_limits_cache, get_max_line_length


def _long_line(width: int) -> str:
    # Multibyte text past the clamp so byte- and char-offsets differ (the bug's shape).
    head = ("€ó" * (width // 2))[:width]
    return head + " Verdict: genuine Honda OEM, fits FK3 (7J ET55). Next: tracking code."


class TestClampedLineHint:
    def test_hint_names_lines_width_and_recipe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        _reset_tool_output_limits_cache()
        width = get_max_line_length()
        p = tmp_path / "deals.md"
        p.write_text("short\n" + _long_line(width) + "\nalso short\n" + _long_line(width) + "\n",
                     encoding="utf-8")
        result = json.loads(read_file_tool(str(p)))
        assert result.get("truncated_lines") is True
        assert result.get("clamped_lines") == [2, 4]
        assert result.get("clamped_at_chars") == width
        hint = result.get("hint") or ""
        assert "Line(s) 2, 4" in hint
        assert f"{width:,}" in hint
        assert "offset/limit cannot reach" in hint
        assert "python3 -c" in hint and shlex.quote(str(p)) in hint

    def test_recipe_reads_the_exact_remainder(self, tmp_path, monkeypatch):
        """Running the suggested command returns exactly what the clamp cut off."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        _reset_tool_output_limits_cache()
        width = get_max_line_length()
        p = tmp_path / "deals.md"
        line = _long_line(width)
        p.write_text("x\n" + line + "\n", encoding="utf-8")
        result = json.loads(read_file_tool(str(p)))
        hint = result["hint"]
        recipe = hint.split("terminal ", 1)[1]
        argv = shlex.split(recipe)
        argv[0] = sys.executable
        out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
        assert out.rstrip("\n") == line[width:]
        assert "Verdict: genuine Honda OEM" in out

    def test_no_hint_when_nothing_clamped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "s.md"
        p.write_text("a\nb\n", encoding="utf-8")
        result = json.loads(read_file_tool(str(p)))
        assert not result.get("truncated_lines")
        assert "clamped_lines" not in result
        assert "clamped" not in (result.get("hint") or "")

    def test_pagination_hint_is_kept_in_front(self):
        result = {"content": "7|" + "x" * 5 + "... [truncated]", "hint": "Use offset=9 to continue",
                  "truncated_lines": True}
        _add_clamped_lines_hint(result, "/tmp/a b.md")
        assert result["hint"].startswith("Use offset=9 to continue ")
        assert "Line(s) 7" in result["hint"]
        assert "'/tmp/a b.md'" in result["hint"]

    def test_marker_inside_text_is_not_a_clamp(self):
        content = "1|plain ... [truncated] not at end\n2|real... [truncated]\nnogutter... [truncated]"
        assert _clamped_line_numbers(content) == [2]

    def test_tilde_path_recipe_reads_the_remainder(self, tmp_path, monkeypatch):
        """A '~/x' path is shell-quoted in the hint, so the recipe must expand it itself."""
        monkeypatch.setenv("HOME", str(tmp_path))
        _reset_tool_output_limits_cache()
        width = get_max_line_length()
        line = _long_line(width)
        (tmp_path / "deals.md").write_text("x\n" + line + "\n", encoding="utf-8")
        result = {"content": "1|x\n2|" + line[:width] + "... [truncated]", "truncated_lines": True}
        _add_clamped_lines_hint(result, "~/deals.md")
        argv = shlex.split(result["hint"].split("terminal ", 1)[1])
        argv[0] = sys.executable
        out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
        assert out.rstrip("\n") == line[width:]

    def test_extracted_document_hint_has_no_raw_file_recipe(self, tmp_path, monkeypatch):
        """Extracted-text line numbers do not map onto the raw file, so no raw-file recipe."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        _reset_tool_output_limits_cache()
        width = get_max_line_length()
        p = tmp_path / "nb.ipynb"
        p.write_text(json.dumps({
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# T\n", _long_line(width)]}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
        result = json.loads(read_file_tool(str(p)))
        assert result.get("extracted_document") is True
        assert result.get("truncated_lines") is True
        assert result.get("clamped_lines")
        assert result.get("clamped_at_chars") == width
        hint = result.get("hint") or ""
        assert "extracted document text" in hint
        assert "python3 -c" not in hint

    @pytest.mark.parametrize("codec, text_of", [
        # BOM + CRLF, clamped line 1: the recipe must drop the BOM and fold CRLF like the read.
        ("utf-16", lambda line: line + "\r\nshort\r\n"),
        # No BOM: the read detects the byte order by zero-byte parity; the recipe must use it.
        ("utf-16-be", lambda line: "x\n" + line + "\n"),
    ])
    def test_utf16_recipe_decodes_like_the_transcoding_read(self, tmp_path, monkeypatch, codec, text_of):
        """A UTF-16 file is transcoded for display; a UTF-8 recipe printed NUL-interleaved bytes
        from about half the clamp width instead of the remainder (t_d5555cdf)."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        _reset_tool_output_limits_cache()
        width = get_max_line_length()
        line = _long_line(width)
        p = tmp_path / "u16.txt"
        p.write_bytes(text_of(line).encode(codec))
        result = json.loads(read_file_tool(str(p)))
        assert result["hint"].startswith("Transcoded from UTF-16")
        clamped = result["clamped_lines"]
        assert clamped == [1 if codec == "utf-16" else 2]
        argv = shlex.split(result["hint"].split("terminal ", 1)[1])
        argv[0] = sys.executable
        out = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", check=True).stdout
        assert out.rstrip("\n") == line[width:]
