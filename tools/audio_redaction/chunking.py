from __future__ import annotations

from tools.audio_redaction.models import Transcript, Window


def build_windows(transcript: Transcript, target_seconds: float = 16.0, overlap_seconds: float = 3.0) -> list[Window]:
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds cannot be negative")
    if not transcript.segments:
        return []

    windows: list[Window] = []
    segments = transcript.segments
    start_index = 0
    window_index = 1

    while start_index < len(segments):
        group: list = []
        group_start = segments[start_index].start
        group_end = group_start
        index = start_index

        while index < len(segments):
            group.append(segments[index])
            group_end = segments[index].end
            index += 1
            if group_end - group_start >= target_seconds:
                break

        windows.append(
            Window(
                window_id=f"w_{window_index:05d}",
                start=group[0].start,
                end=group[-1].end,
                segment_ids=[segment.segment_id for segment in group],
                text=" ".join(segment.text.strip() for segment in group if segment.text.strip()).strip(),
            )
        )
        window_index += 1

        if index >= len(segments):
            break

        next_start_index = start_index + 1
        overlap_boundary = group[-1].end - overlap_seconds
        while next_start_index < len(segments) and segments[next_start_index].end <= overlap_boundary:
            next_start_index += 1
        if next_start_index <= start_index:
            next_start_index = start_index + 1
        start_index = next_start_index

    return windows
