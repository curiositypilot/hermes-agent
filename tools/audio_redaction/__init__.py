"""Local audio NSFW redaction pipeline."""

from tools.audio_redaction.align import align_phrase_to_window
from tools.audio_redaction.chunking import build_windows
from tools.audio_redaction.models import (
    FlagSpan,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    Window,
    merge_flag_spans,
)

__all__ = [
    "align_phrase_to_window",
    "build_windows",
    "FlagSpan",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
    "Window",
    "merge_flag_spans",
]
