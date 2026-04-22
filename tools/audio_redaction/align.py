from __future__ import annotations

import re
import string
from difflib import SequenceMatcher

from tools.audio_redaction.models import FlagSpan, Transcript, TranscriptWord, Window

_WORD_STRIP = string.punctuation + "“”‘’«»"


def _normalize_token(token: str) -> str:
    return re.sub(r"\s+", "", token.casefold().strip(_WORD_STRIP))


def _window_words(transcript: Transcript, window: Window) -> list[TranscriptWord]:
    segment_ids = set(window.segment_ids)
    words: list[TranscriptWord] = []
    for segment in transcript.segments:
        if segment.segment_id in segment_ids:
            words.extend(segment.words)
    return words


def _match_phrase(words: list[TranscriptWord], phrase: str) -> tuple[int, int, float] | None:
    normalized_words = [_normalize_token(word.word) for word in words]
    phrase_tokens = [_normalize_token(token) for token in phrase.split() if _normalize_token(token)]
    if not phrase_tokens:
        return None

    for start_index in range(0, len(normalized_words) - len(phrase_tokens) + 1):
        candidate = normalized_words[start_index : start_index + len(phrase_tokens)]
        if candidate == phrase_tokens:
            return start_index, start_index + len(phrase_tokens) - 1, 1.0

    joined_phrase = " ".join(phrase_tokens)
    best: tuple[int, int, float] | None = None
    for start_index in range(len(normalized_words)):
        for end_index in range(start_index, len(normalized_words)):
            candidate = " ".join(token for token in normalized_words[start_index : end_index + 1] if token)
            score = SequenceMatcher(None, joined_phrase, candidate).ratio()
            if score >= 0.82 and (best is None or score > best[2]):
                best = (start_index, end_index, round(score, 3))
    return best


def align_phrase_to_window(
    transcript: Transcript,
    window: Window,
    phrase: str,
    severity: int,
    category: str,
    action: str,
    confidence: float,
    reason: str,
    pre_pad_ms: int = 180,
    post_pad_ms: int = 220,
) -> FlagSpan | None:
    words = _window_words(transcript, window)
    if not words:
        return None

    match = _match_phrase(words, phrase)
    if match is None:
        return None

    start_index, end_index, alignment_confidence = match
    start = max(window.start, words[start_index].start - (pre_pad_ms / 1000.0))
    end = min(window.end, words[end_index].end + (post_pad_ms / 1000.0))
    return FlagSpan(
        source_file=transcript.source_file,
        window_id=window.window_id,
        start=round(start, 3),
        end=round(end, 3),
        action=action,
        severity=severity,
        category=category,
        categories=[category],
        confidence=confidence,
        matched_text=phrase,
        reason=reason,
        reasons=[reason],
        alignment_confidence=alignment_confidence,
    )
