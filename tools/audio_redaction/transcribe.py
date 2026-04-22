from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools.audio_redaction.models import Transcript, TranscriptSegment, TranscriptWord

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def normalize_media(source_file: str | Path, output_dir: str | Path) -> Path:
    source = Path(source_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = out_dir / f"{source.stem}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(normalized),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return normalized


def _load_model(model_name: str, device: str, compute_type: str):
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Install hermes-agent[voice] or pip install faster-whisper")
    cache_key = (model_name, device, compute_type)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _MODEL_CACHE[cache_key]


def transcribe_file(audio_file: str | Path, config: dict[str, Any]) -> Transcript:
    transcription_cfg = config["transcription"]
    model = _load_model(
        transcription_cfg["model"],
        transcription_cfg.get("device", "auto"),
        transcription_cfg.get("compute_type", "auto"),
    )
    segments, info = model.transcribe(
        str(audio_file),
        language=transcription_cfg.get("language"),
        word_timestamps=bool(transcription_cfg.get("word_timestamps", True)),
        vad_filter=True,
    )
    transcript_segments: list[TranscriptSegment] = []
    for index, segment in enumerate(segments, start=1):
        words = [
            TranscriptWord(word=word.word.strip(), start=round(word.start, 3), end=round(word.end, 3))
            for word in (segment.words or [])
            if word.word.strip()
        ]
        transcript_segments.append(
            TranscriptSegment(
                segment_id=index,
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text.strip(),
                words=words,
            )
        )
    duration_sec = transcript_segments[-1].end if transcript_segments else 0.0
    return Transcript(
        source_file=Path(audio_file).name,
        language=getattr(info, "language", None),
        duration_sec=duration_sec,
        segments=transcript_segments,
    )


def save_transcript(transcript: Transcript, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(transcript.to_dict(), handle, ensure_ascii=False, indent=2)
