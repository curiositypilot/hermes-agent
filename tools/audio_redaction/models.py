from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Action = Literal["keep", "review", "redact"]


@dataclass(slots=True)
class TranscriptWord:
    word: str
    start: float
    end: float


@dataclass(slots=True)
class TranscriptSegment:
    segment_id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass(slots=True)
class Transcript:
    source_file: str
    language: str | None
    duration_sec: float
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Window:
    window_id: str
    start: float
    end: float
    segment_ids: list[int]
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MatchedSpan:
    text: str
    category: str
    severity: int
    confidence: float
    reason: str


@dataclass(slots=True)
class WindowClassification:
    window_id: str
    unsafe_for_work: bool
    severity: int
    action: Action
    reasons: list[str]
    matched_spans: list[MatchedSpan] = field(default_factory=list)
    notes: str | None = None


@dataclass(slots=True)
class FlagSpan:
    source_file: str
    window_id: str
    start: float
    end: float
    action: Action
    severity: int
    category: str
    categories: list[str]
    confidence: float
    matched_text: str
    reason: str
    reasons: list[str]
    alignment_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _action_priority(action: Action) -> int:
    return {"keep": 0, "review": 1, "redact": 2}[action]


def merge_flag_spans(spans: list[FlagSpan], merge_gap_sec: float) -> list[FlagSpan]:
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[FlagSpan] = [ordered[0]]

    for span in ordered[1:]:
        current = merged[-1]
        if span.start - current.end > merge_gap_sec:
            merged.append(span)
            continue

        current.end = max(current.end, span.end)
        if span.severity > current.severity:
            current.severity = span.severity
        if _action_priority(span.action) > _action_priority(current.action):
            current.action = span.action
        current.confidence = max(current.confidence, span.confidence)
        current.alignment_confidence = max(current.alignment_confidence, span.alignment_confidence)
        if span.category not in current.categories:
            current.categories.append(span.category)
        if span.category != current.category:
            current.category = current.categories[0]
        if span.reason not in current.reasons:
            current.reasons.append(span.reason)
        if span.matched_text and span.matched_text not in current.matched_text:
            current.matched_text = "; ".join(filter(None, [current.matched_text, span.matched_text]))

    return merged
