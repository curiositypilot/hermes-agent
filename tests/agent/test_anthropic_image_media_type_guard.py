"""Inline images whose bytes disagree with the declared media_type must never reach Anthropic:
one such block in history 400s every later turn of the session (poisoned session)."""
import base64

from agent.anthropic_message_convert import _image_block_from_openai_url

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def test_jpeg_bytes_labelled_png_are_relabelled():
    block = _image_block_from_openai_url(_url("image/png", JPEG))
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/jpeg"


def test_matching_label_is_kept():
    block = _image_block_from_openai_url(_url("image/png", PNG))
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"] == base64.b64encode(PNG).decode()


def test_garbage_base64_becomes_text_placeholder():
    block = _image_block_from_openai_url("data:image/png;base64,!!!not-base64!!!")
    assert block["type"] == "text"
    assert "image omitted" in block["text"]


def test_unrecognised_bytes_become_text_placeholder():
    block = _image_block_from_openai_url(_url("image/jpeg", b"hello world, not an image" * 4))
    assert block["type"] == "text"


def test_remote_url_untouched():
    block = _image_block_from_openai_url("https://example.com/a.png")
    assert block["source"] == {"type": "url", "url": "https://example.com/a.png"}


def test_browser_exec_attaches_jpeg_with_its_real_mime(tmp_path, monkeypatch):
    from tools import browser_use_cli, vision_tools
    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)
    photo = tmp_path / "s3_0.jpg"
    photo.write_bytes(JPEG)
    out = browser_use_cli._native_screenshot_result({"success": True}, str(photo))
    assert out is not None
    assert out["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
