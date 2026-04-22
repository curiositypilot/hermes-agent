from pathlib import Path

from tools.audio_redaction.align import align_phrase_to_window
from tools.audio_redaction.chunking import build_windows
from tools.audio_redaction.classify import _normalize_action, _parse_json_object
from tools.audio_redaction.config import apply_topic_model_override, load_config
from tools.audio_redaction.models import Transcript, TranscriptSegment, TranscriptWord, Window, merge_flag_spans


def _sample_transcript() -> Transcript:
    return Transcript(
        source_file="sample.mp3",
        language="uk",
        duration_sec=12.0,
        segments=[
            TranscriptSegment(
                segment_id=1,
                start=0.0,
                end=3.0,
                text="This is casual banter",
                words=[
                    TranscriptWord("This", 0.0, 0.3),
                    TranscriptWord("is", 0.3, 0.4),
                    TranscriptWord("casual", 0.4, 0.8),
                    TranscriptWord("banter", 0.8, 1.3),
                ],
            ),
            TranscriptSegment(
                segment_id=2,
                start=3.0,
                end=6.0,
                text="very explicit sexual phrase here",
                words=[
                    TranscriptWord("very", 3.0, 3.2),
                    TranscriptWord("explicit", 3.2, 3.7),
                    TranscriptWord("sexual", 3.7, 4.1),
                    TranscriptWord("phrase", 4.1, 4.5),
                    TranscriptWord("here", 4.5, 4.8),
                ],
            ),
            TranscriptSegment(
                segment_id=3,
                start=6.0,
                end=9.0,
                text="more neutral content",
                words=[
                    TranscriptWord("more", 6.0, 6.2),
                    TranscriptWord("neutral", 6.2, 6.7),
                    TranscriptWord("content", 6.7, 7.1),
                ],
            ),
        ],
    )


def test_build_windows_groups_segments_with_overlap_context():
    transcript = _sample_transcript()

    windows = build_windows(transcript, target_seconds=4.0, overlap_seconds=1.0)

    assert [window.segment_ids for window in windows] == [[1, 2], [2, 3]]
    assert windows[0].text == "This is casual banter very explicit sexual phrase here"
    assert windows[1].start == 3.0
    assert windows[1].end == 9.0


def test_align_phrase_to_window_uses_word_timestamps_and_padding():
    transcript = _sample_transcript()
    window = Window(
        window_id="w1",
        start=0.0,
        end=6.0,
        segment_ids=[1, 2],
        text="This is casual banter very explicit sexual phrase here",
    )

    span = align_phrase_to_window(
        transcript=transcript,
        window=window,
        phrase="explicit sexual phrase",
        severity=3,
        category="sexual_explicit",
        action="redact",
        confidence=0.91,
        reason="explicit sexual description",
        pre_pad_ms=100,
        post_pad_ms=200,
    )

    assert span is not None
    assert span.start == 3.1
    assert span.end == 4.7
    assert span.matched_text == "explicit sexual phrase"
    assert span.alignment_confidence == 1.0


def test_merge_flag_spans_combines_adjacent_ranges_and_unions_metadata():
    spans = [
        align_phrase_to_window(
            transcript=_sample_transcript(),
            window=Window(
                window_id="w1",
                start=0.0,
                end=6.0,
                segment_ids=[1, 2],
                text="This is casual banter very explicit sexual phrase here",
            ),
            phrase="explicit sexual",
            severity=2,
            category="sexual_explicit",
            action="review",
            confidence=0.7,
            reason="borderline",
            pre_pad_ms=0,
            post_pad_ms=0,
        ),
        align_phrase_to_window(
            transcript=_sample_transcript(),
            window=Window(
                window_id="w1",
                start=0.0,
                end=6.0,
                segment_ids=[1, 2],
                text="This is casual banter very explicit sexual phrase here",
            ),
            phrase="phrase here",
            severity=3,
            category="extreme_insult",
            action="redact",
            confidence=0.9,
            reason="definitely not safe",
            pre_pad_ms=0,
            post_pad_ms=0,
        ),
    ]

    merged = merge_flag_spans([span for span in spans if span is not None], merge_gap_sec=0.5)

    assert len(merged) == 1
    assert merged[0].severity == 3
    assert merged[0].action == "redact"
    assert set(merged[0].categories) == {"sexual_explicit", "extreme_insult"}
    assert set(merged[0].reasons) == {"borderline", "definitely not safe"}


def test_apply_topic_model_override_reads_local_topic_route(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
classification:
  api_base: http://example.invalid/v1
  api_key: old
  model: old-model
topic_models:
  telegram:-1003882011045:1720:
    model: gemma-local
    base_url: http://127.0.0.1:8080/v1
    api_key: local
    max_tokens: 4096
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    updated = apply_topic_model_override(config, "telegram:-1003882011045:1720", hermes_config_path=config_path)

    assert updated["classification"]["model"] == "gemma-local"
    assert updated["classification"]["api_base"] == "http://127.0.0.1:8080/v1"
    assert updated["classification"]["api_key"] == "local"
    assert updated["classification"]["max_tokens"] == 4096


def test_parse_json_object_strips_channel_wrapper():
    parsed = _parse_json_object("<|channel>thought\n<channel|>{\"ok\": true}")

    assert parsed == {"ok": True}


def test_normalize_action_matches_severity_thresholds():
    assert _normalize_action(0, "redact") == "keep"
    assert _normalize_action(2, "redact") == "review"
    assert _normalize_action(3, "review") == "redact"
