#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tools.audio_redaction.classify import load_prompt_file
from tools.audio_redaction.config import DEFAULT_HERMES_CONFIG_PATH, apply_topic_model_override, load_config
from tools.audio_redaction.pipeline import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch contextual NSFW redaction for local audio/video files")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--input", dest="input_dir", default=None, help="Override input directory")
    parser.add_argument("--output", dest="output_dir", default=None, help="Override output directory")
    parser.add_argument("--mode", choices=["review", "auto", "strict"], default="auto")
    parser.add_argument("--prompt-file", default=None, help="Optional classifier prompt text file")
    parser.add_argument("--topic-key", default=None, help="Read classification model/base_url/api_key from a Hermes topic_models entry")
    parser.add_argument("--hermes-config", default=str(DEFAULT_HERMES_CONFIG_PATH), help="Hermes config path used with --topic-key")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.topic_key:
        config = apply_topic_model_override(config, args.topic_key, hermes_config_path=args.hermes_config)
    prompt_text = load_prompt_file(args.prompt_file)
    summary = run_batch(config=config, mode=args.mode, prompt_text=prompt_text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
