from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.audio_redaction.align import align_phrase_to_window
from tools.audio_redaction.chunking import build_windows
from tools.audio_redaction.classify import LocalOpenAICompatibleClassifier
from tools.audio_redaction.io_utils import discover_media_files, ensure_directory, output_stem
from tools.audio_redaction.models import FlagSpan, Transcript, WindowClassification, merge_flag_spans
from tools.audio_redaction.redact import redact_audio
from tools.audio_redaction.report import write_flags_csv, write_flags_json
from tools.audio_redaction.transcribe import normalize_media, save_transcript, transcribe_file


def _classification_to_spans(
    transcript: Transcript,
    window,
    classification: WindowClassification,
    config: dict[str, Any],
) -> list[FlagSpan]:
    policy = config["policy"]
    spans: list[FlagSpan] = []
    for matched in classification.matched_spans:
        span = align_phrase_to_window(
            transcript=transcript,
            window=window,
            phrase=matched.text,
            severity=matched.severity,
            category=matched.category,
            action=classification.action,
            confidence=matched.confidence,
            reason=matched.reason,
            pre_pad_ms=policy["pre_pad_ms"],
            post_pad_ms=policy["post_pad_ms"],
        )
        if span is not None:
            spans.append(span)
    return spans


def process_file(source_file: str | Path, config: dict[str, Any], mode: str = "auto", prompt_text: str | None = None) -> dict[str, Any]:
    output_root = ensure_directory(config["output_dir"])
    temp_dir = ensure_directory(output_root / "temp")
    transcript_dir = ensure_directory(output_root / "transcripts")
    report_dir = ensure_directory(output_root / "reports")
    cleaned_dir = ensure_directory(output_root / "cleaned")

    normalized = normalize_media(source_file, temp_dir)
    transcript = transcribe_file(normalized, config)
    stem = output_stem(source_file)
    save_transcript(transcript, transcript_dir / f"{stem}.transcript.json")

    classifier = LocalOpenAICompatibleClassifier(config=config, prompt_text=prompt_text)
    windows = build_windows(
        transcript,
        target_seconds=float(config["windowing"]["target_seconds"]),
        overlap_seconds=float(config["windowing"]["overlap_seconds"]),
    )
    spans: list[FlagSpan] = []
    review_count = 0
    for window in windows:
        classification = classifier.classify_window(window)
        if classification.action == "review":
            review_count += 1
        spans.extend(_classification_to_spans(transcript, window, classification, config))

    merged = merge_flag_spans(spans, float(config["policy"]["merge_gap_sec"]))
    write_flags_json(merged, report_dir / f"{stem}.flags.json")
    write_flags_csv(merged, report_dir / f"{stem}.flags.csv")

    cleaned_output = None
    if mode != "review":
        threshold = int(config["policy"]["redact_threshold"])
        if mode == "strict":
            threshold = int(config["policy"]["review_threshold"])
        selected = [span for span in merged if span.severity >= threshold and span.action == "redact"]
        cleaned_output = cleaned_dir / f"{stem}.cleaned.{config['redaction']['output_format']}"
        redact_audio(normalized, selected, config["redaction"]["mode"], cleaned_output)

    return {
        "source_file": str(source_file),
        "normalized_file": str(normalized),
        "flagged_spans": len(merged),
        "review_windows": review_count,
        "cleaned_output": str(cleaned_output) if cleaned_output else None,
    }


def run_batch(config: dict[str, Any], mode: str = "auto", prompt_text: str | None = None) -> list[dict[str, Any]]:
    batch_cfg = config["batch"]
    files = discover_media_files(
        input_dir=config["input_dir"],
        supported_extensions=batch_cfg["supported_extensions"],
        recursive=bool(batch_cfg.get("recursive", False)),
    )
    return [process_file(file, config=config, mode=mode, prompt_text=prompt_text) for file in files]
