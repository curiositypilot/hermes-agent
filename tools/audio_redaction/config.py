from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HERMES_CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": "./input",
    "output_dir": "./output",
    "transcription": {
        "engine": "faster-whisper",
        "model": "large-v3",
        "language": None,
        "device": "auto",
        "compute_type": "auto",
        "word_timestamps": True,
    },
    "windowing": {
        "target_seconds": 16,
        "overlap_seconds": 3,
    },
    "policy": {
        "redact_threshold": 3,
        "review_threshold": 2,
        "pre_pad_ms": 180,
        "post_pad_ms": 220,
        "merge_gap_sec": 0.4,
    },
    "classification": {
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "",
        "model": "qwen2.5:14b-instruct",
        "temperature": 0,
        "timeout_seconds": 120,
        "max_retries": 2,
        "max_tokens": 512,
        "extra_body": {},
    },
    "redaction": {
        "mode": "mute",
        "output_format": "wav",
    },
    "batch": {
        "recursive": False,
        "skip_existing": True,
        "supported_extensions": [".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv"],
    },
}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _merge_dicts(DEFAULT_CONFIG, loaded)


def load_hermes_topic_override(topic_key: str, hermes_config_path: str | Path = DEFAULT_HERMES_CONFIG_PATH) -> dict[str, Any]:
    path = Path(hermes_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Hermes config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    topic_models = loaded.get("topic_models", {})
    override = topic_models.get(topic_key)
    if not isinstance(override, dict):
        raise KeyError(f"Topic model override not found for {topic_key}")
    return deepcopy(override)


def apply_topic_model_override(
    config: dict[str, Any],
    topic_key: str,
    hermes_config_path: str | Path = DEFAULT_HERMES_CONFIG_PATH,
) -> dict[str, Any]:
    updated = deepcopy(config)
    override = load_hermes_topic_override(topic_key, hermes_config_path=hermes_config_path)
    classification = updated.setdefault("classification", {})

    if override.get("base_url"):
        classification["api_base"] = override["base_url"]
    if "api_key" in override:
        classification["api_key"] = override["api_key"]
    if override.get("model"):
        classification["model"] = override["model"]
    if "max_tokens" in override:
        classification["max_tokens"] = override["max_tokens"]
    if isinstance(override.get("extra_body"), dict):
        classification["extra_body"] = deepcopy(override["extra_body"])

    return updated
