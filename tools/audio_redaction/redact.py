from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from tools.audio_redaction.models import FlagSpan


def redact_audio(source_file: str | Path, spans: Iterable[FlagSpan], mode: str, output_file: str | Path) -> Path:
    source = Path(source_file)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    spans = list(spans)
    if not spans:
        subprocess.run(["ffmpeg", "-y", "-i", str(source), "-c", "copy", str(output)], check=True, capture_output=True, text=True)
        return output
    if mode != "mute":
        raise NotImplementedError("Only mute mode is implemented in the MVP")

    expressions = [f"between(t,{span.start:.3f},{span.end:.3f})" for span in spans]
    enabled = "+".join(expressions)
    filter_complex = f"volume=enable='{enabled}':volume=0"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-af", filter_complex, str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output
