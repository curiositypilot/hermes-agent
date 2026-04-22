from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from tools.audio_redaction.models import MatchedSpan, Window, WindowClassification
from tools.audio_redaction.prompts import DEFAULT_CLASSIFIER_PROMPT


class LocalOpenAICompatibleClassifier:
    def __init__(self, config: dict[str, Any], prompt_text: str | None = None):
        classification_cfg = config["classification"]
        self.base_url = classification_cfg["api_base"].rstrip("/")
        self.api_key = classification_cfg.get("api_key", "")
        self.model = classification_cfg["model"]
        self.temperature = classification_cfg.get("temperature", 0)
        self.timeout_seconds = classification_cfg.get("timeout_seconds", 120)
        self.max_retries = classification_cfg.get("max_retries", 2)
        self.max_tokens = classification_cfg.get("max_tokens", 512)
        self.extra_body = classification_cfg.get("extra_body", {}) or {}
        self.prompt_text = prompt_text or DEFAULT_CLASSIFIER_PROMPT

    def classify_window(self, window: Window) -> WindowClassification:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.prompt_text},
                {"role": "user", "content": json.dumps(asdict(window), ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.extra_body:
            payload.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                data = _parse_json_object(content)
                matched_spans = [MatchedSpan(**item) for item in data.get("matched_spans", [])]
                severity = int(data.get("severity", 0))
                return WindowClassification(
                    window_id=data.get("window_id", window.window_id),
                    unsafe_for_work=bool(data.get("unsafe_for_work", False)),
                    severity=severity,
                    action=_normalize_action(severity, data.get("action", "keep")),
                    reasons=list(data.get("reasons", [])),
                    matched_spans=matched_spans,
                    notes=data.get("notes"),
                )
            except Exception as exc:  # pragma: no cover - network/runtime path
                last_error = exc
        raise RuntimeError(f"Classification failed for {window.window_id}: {last_error}")


def _normalize_action(severity: int, action: str) -> str:
    if severity >= 3:
        return "redact"
    if severity == 2:
        return "review"
    return "keep"


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("<|channel>") and "<channel|>" in text:
        text = text.split("<channel|>", 1)[1].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def load_prompt_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")
