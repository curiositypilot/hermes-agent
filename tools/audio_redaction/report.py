from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.audio_redaction.models import FlagSpan


def write_flags_json(spans: list[FlagSpan], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([span.to_dict() for span in spans], handle, ensure_ascii=False, indent=2)


def write_flags_csv(spans: list[FlagSpan], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "window_id",
                "start",
                "end",
                "action",
                "severity",
                "category",
                "categories",
                "confidence",
                "matched_text",
                "reason",
                "reasons",
                "alignment_confidence",
            ],
        )
        writer.writeheader()
        for span in spans:
            row = span.to_dict()
            row["categories"] = "|".join(span.categories)
            row["reasons"] = "|".join(span.reasons)
            writer.writerow(row)
