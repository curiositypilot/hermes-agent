"""Fork: second-pass STT for local faster-whisper results that look wrong.

Triggers (only when ``stt.second_pass.provider`` is set):
  * auto-detected language outside ``stt.local.allowed_languages`` (e.g. uk/ru speech detected as lv);
  * a repetition loop (the same 1-3 word chunk repeated ``repetition_threshold``+ times in a row).

Recovery: re-transcribe with Deepgram; if that fails, re-run local whisper forced to the most likely
allowed language (from faster-whisper's ``all_language_probs``). Config::

    stt:
      local: {allowed_languages: [en, uk, ru]}
      second_pass: {provider: deepgram, model: nova-3, language: multi, smart_format: true}
      # language_hint defaults to allowed_languages (Deepgram nova-3 multi biasing).
"""

from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tools.transcription_common import _error_result, _ok_result

logger = logging.getLogger(__name__)

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
_DEFAULT_REPETITION_THRESHOLD = 6


def allowed_languages(stt_config: Dict[str, Any]) -> List[str]:
    raw = (stt_config.get("local") or {}).get("allowed_languages") or []
    return [str(code).strip().lower() for code in raw if str(code).strip()]


def has_repetition_loop(text: str, threshold: int = _DEFAULT_REPETITION_THRESHOLD) -> bool:
    """True when a 1-3 word chunk repeats ``threshold``+ times back to back ("da da da da da da")."""
    if threshold < 2 or not text:
        return False
    pattern = r"\b(\w+(?:\W+\w+){0,2})(?:\W+\1\b){%d,}" % (threshold - 1)
    return re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE) is not None


def second_pass_reason(stt_config: Dict[str, Any], detected: Optional[str], transcript: str,
                       language_was_forced: bool) -> Optional[str]:
    """Why the local result needs a second pass, or None (also None when the feature is off)."""
    sp_cfg = stt_config.get("second_pass") or {}
    if not isinstance(sp_cfg, dict) or not sp_cfg.get("provider"):
        return None
    allowed = allowed_languages(stt_config)
    if allowed and not language_was_forced and (detected or "").lower() not in allowed:
        return f"detected language {detected!r} not in allowed {allowed}"
    try:
        threshold = int(sp_cfg.get("repetition_threshold") or _DEFAULT_REPETITION_THRESHOLD)
    except (TypeError, ValueError):
        threshold = _DEFAULT_REPETITION_THRESHOLD
    if has_repetition_loop(transcript, threshold):
        return f"repetition loop ({threshold}+ repeats)"
    return None


def best_allowed_language(allowed: Sequence[str], probs: Optional[Sequence[Tuple[str, float]]]) -> Optional[str]:
    """Most probable allowed language per the detector; first allowed entry when probs are missing."""
    if not allowed:
        return None
    scores = {str(code).lower(): float(p) for code, p in (probs or [])}
    return max(allowed, key=lambda code: (scores.get(code, -1.0), -list(allowed).index(code)))


def transcribe_deepgram(file_path: str, sp_cfg: Dict[str, Any], hints: Sequence[str] = ()) -> Dict[str, Any]:
    """Deepgram pre-recorded transcription (raw-bytes POST). *hints* become ``language_hint`` params
    (nova-3 ``language=multi`` only; they bias detection, they don't restrict it)."""
    import requests
    from tools.tool_backend_helpers import resolve_provider_secret

    api_key = resolve_provider_secret("DEEPGRAM_API_KEY", "deepgram")
    if not api_key:
        return _error_result("DEEPGRAM_API_KEY not set", provider="deepgram")
    params = {"model": str(sp_cfg.get("model") or "nova-3"),
              "language": str(sp_cfg.get("language") or "multi"),
              "smart_format": str(bool(sp_cfg.get("smart_format", True))).lower()}
    query = list(params.items()) + ([("language_hint", h) for h in hints] if params["language"] == "multi" else [])
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    try:
        with open(file_path, "rb") as fh:
            resp = requests.post(DEEPGRAM_LISTEN_URL, params=query, data=fh,
                                 headers={"Authorization": f"Token {api_key}", "Content-Type": content_type},
                                 timeout=float(sp_cfg.get("timeout") or 60))
        if resp.status_code >= 400:
            return _error_result(f"Deepgram HTTP {resp.status_code}: {resp.text[:300]}", provider="deepgram")
        alt = resp.json()["results"]["channels"][0]["alternatives"][0]
        transcript = (alt.get("transcript") or "").strip()
    except Exception as exc:  # network, JSON shape, file errors
        return _error_result(f"Deepgram request failed: {exc}", provider="deepgram")
    if not transcript:
        return _error_result("Deepgram returned an empty transcript", provider="deepgram")
    logger.info("Second-pass transcribed %s via Deepgram (%s, language=%s, langs=%s, %d chars)",
                Path(file_path).name, params["model"], params["language"], alt.get("languages"), len(transcript))
    return _ok_result(transcript, "deepgram")


_SECOND_PASS_PROVIDERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "deepgram": transcribe_deepgram}


def run_second_pass(file_path: str, stt_config: Dict[str, Any], reason: str, detected: Optional[str],
                    probs: Optional[Sequence[Tuple[str, float]]],
                    rerun_local: Callable[[str], str], original: str) -> Dict[str, Any]:
    """Cloud second pass, else local re-run forced to the best allowed language, else the original."""
    sp_cfg = stt_config.get("second_pass") or {}
    provider = str(sp_cfg.get("provider")).strip().lower()
    logger.warning("STT second pass for %s: %s; trying %s", Path(file_path).name, reason, provider)
    handler = _SECOND_PASS_PROVIDERS.get(provider)
    allowed = allowed_languages(stt_config)
    hints = sp_cfg.get("language_hint", allowed)  # default: bias toward the allowed list; [] = none
    hints = [hints] if isinstance(hints, str) else list(hints or [])
    result = handler(file_path, sp_cfg, hints) if handler else _error_result(f"unknown second-pass provider {provider!r}")
    if result.get("success"):
        return result
    logger.warning("STT second pass via %s failed: %s", provider, result.get("error"))
    forced = best_allowed_language(allowed, probs)
    if forced and forced != (detected or "").lower():
        try:
            transcript = rerun_local(forced)
            logger.info("STT second pass: local re-run forced to %s for %s (%d chars)",
                        forced, Path(file_path).name, len(transcript))
            return _ok_result(transcript, "local")
        except Exception as exc:
            logger.error("STT second pass: forced local re-run (%s) failed: %s", forced, exc)
    return _ok_result(original, "local")
