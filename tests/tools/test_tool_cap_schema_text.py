"""Schema text for read_file and execute_code tracks the configured output cap.

The stock strings (~100K, Over 50KB) stay when the config keys are at their
defaults, so a stock prompt surface is unchanged. A non-default cap is rendered
into the description the model plans with.
"""

from __future__ import annotations


def test_read_file_custom_cap_renders_without_mutating_schema(monkeypatch):
    import tools.file_tools as ft

    before_desc = ft.READ_FILE_SCHEMA["description"]
    before_limit = ft.READ_FILE_SCHEMA["parameters"]["properties"]["limit"]["description"]
    monkeypatch.setattr(ft, "_get_max_read_chars", lambda: 40000)
    monkeypatch.setattr("tools.read_extract.hosted_ocr_available", lambda: False)

    overrides = ft._read_file_schema_overrides()

    desc = overrides["description"]
    limit = overrides["parameters"]["properties"]["limit"]["description"]
    assert "~40K characters" in desc
    assert "~100K" not in desc
    assert "~40K-character" in limit
    assert overrides["parameters"] is not ft.READ_FILE_SCHEMA["parameters"]
    assert ft.READ_FILE_SCHEMA["description"] == before_desc
    assert ft.READ_FILE_SCHEMA["parameters"]["properties"]["limit"]["description"] == before_limit


def test_read_file_default_cap_and_ocr_off_returns_empty(monkeypatch):
    import tools.file_tools as ft

    monkeypatch.setattr(ft, "_get_max_read_chars", lambda: ft._DEFAULT_MAX_READ_CHARS)
    monkeypatch.setattr("tools.read_extract.hosted_ocr_available", lambda: False)

    assert ft._read_file_schema_overrides() == {}


def test_read_file_custom_cap_and_ocr_apply_together(monkeypatch):
    import tools.file_tools as ft

    monkeypatch.setattr(ft, "_get_max_read_chars", lambda: 40000)
    monkeypatch.setattr("tools.read_extract.hosted_ocr_available", lambda: True)

    desc = ft._read_file_schema_overrides()["description"]

    assert "~40K characters" in desc
    assert "PDF (scanned or text)" in desc


def test_read_file_non_round_cap_uses_grouped_digits(monkeypatch):
    import tools.file_tools as ft

    monkeypatch.setattr(ft, "_get_max_read_chars", lambda: 12345)
    monkeypatch.setattr("tools.read_extract.hosted_ocr_available", lambda: False)

    assert "~12,345 characters" in ft._read_file_schema_overrides()["description"]


def test_execute_code_custom_byte_cap_rewrites_terminal_line_only(monkeypatch):
    from tools.code_execution_tool import build_execute_code_schema

    # build_execute_code_schema imports get_max_bytes from this module at call time.
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 20000)

    desc = build_execute_code_schema()["description"]

    assert "Over 20KB, output is head+tail cut" in desc
    assert "Stdout over 50KB" in desc


def test_execute_code_default_byte_cap_keeps_stock_sentence():
    from tools.code_execution_tool import build_execute_code_schema
    from tools.tool_output_limits import DEFAULT_MAX_BYTES, get_max_bytes

    assert get_max_bytes() == DEFAULT_MAX_BYTES
    desc = build_execute_code_schema()["description"]
    assert "Over 50KB, output is head+tail cut" in desc
