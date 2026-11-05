"""A failed script's notice quotes the END of its stderr, where the cause sits (t_69291d34).

A no_agent script that logs progress to stderr ends with its cause; the notice used to keep the
first 177 chars and showed only progress lines. Fixture: the marketplace-watch failure of
2026-11-05 09:11:31, ``Script exited with code 2`` to EOF of its run output (cause: rrr HTTP 403).
"""

from pathlib import Path

from cron.scheduler import _summarize_cron_failure_for_delivery

JOB = {"id": "d3592b9ef30c", "name": "marketplace-watch", "no_agent": True}
FIXTURE = Path(__file__).parent / "fixtures" / "marketplace_watch_2026-11-05.txt"


def _notice(error: str, job: dict = JOB) -> str:
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "...." not in msg, msg
    return msg


def _progress(n: int) -> str:
    return "\n".join(f"rrr part-{i}--zq5x: {i} (site says {i})" for i in range(n))


def test_replayed_marketplace_watch_failure_shows_its_cause():
    msg = _notice(FIXTURE.read_text())
    assert "HTTP 403" in msg, msg
    assert "rrr:" in msg, msg
    assert "trunk-boot-mat-liner" not in msg, msg
    assert "Script exited with code 2, last stderr lines: …" in msg, msg


def test_cause_after_more_than_2000_chars_of_progress_survives():
    error = f"Script exited with code 2\nstderr:\n{_progress(80)}\n- olx: offers api: HTTP 500"
    assert len(error) > 2000
    msg = _notice(error)
    assert "- olx: offers api: HTTP 500" in msg, msg
    assert "part-0--" not in msg, msg


def test_tail_comes_from_stderr_when_both_streams_exist():
    error = ("Script exited with code 1\nstderr:\nloading\nstderr cause: disk full\n"
             "stdout:\nrow 1\nstdout last row")
    msg = _notice(error)
    assert "last stderr lines:" in msg, msg
    assert "stderr cause: disk full" in msg, msg
    assert "stdout last row" not in msg, msg


def test_stdout_only_failure_quotes_the_stdout_tail():
    msg = _notice("Script exited with code 3\nstdout:\nfetched 10\nno rows matched the filter")
    assert "last stdout lines:" in msg, msg
    assert "no rows matched the filter" in msg, msg


def test_short_stderr_is_shown_whole_without_ellipsis():
    msg = _notice("Script exited with code 1\nstderr:\nboom: config.yaml not found")
    assert "Script exited with code 1, last stderr lines: boom: config.yaml not found. " in msg, msg
    assert "…" not in msg, msg


def test_one_overlong_last_line_keeps_its_end_at_a_word_boundary():
    line = " ".join(f"word{i}" for i in range(120)) + " FINAL-CAUSE"
    msg = _notice(f"Script exited with code 1\nstderr:\n{line}")
    assert "FINAL-CAUSE" in msg, msg
    quoted = msg.split("last stderr lines: ", 1)[1].split(". See the full run", 1)[0]
    assert quoted.startswith("…word"), quoted
    assert len(quoted) <= 301, len(quoted)


def test_script_exit_without_a_stream_keeps_the_bare_line():
    msg = _notice("Script exited with code 1")
    assert "Script exited with code 1" in msg, msg
    assert "last stderr lines" not in msg, msg


def test_non_script_error_keeps_head_truncation():
    msg = _notice("RuntimeError: " + "a" * 200 + " TAILMARK " + "b" * 190)
    assert "failed: aaaa" in msg, msg
    assert "TAILMARK" not in msg, msg
    assert "…. See the full run" in msg, msg


def test_script_timeout_keeps_its_own_copy():
    msg = _notice("Script timed out after 5s: x.py")
    assert "its script timed out" in msg, msg


def test_script_exit_phrase_not_at_the_start_keeps_head_truncation():
    error = "wrapper failed: Script exited with code 2\nstderr:\n" + _progress(10) + "\nREAL CAUSE"
    msg = _notice(error)
    assert "failed: wrapper failed: Script exited with code 2" in msg, msg
    assert "REAL CAUSE" not in msg, msg
    assert "last stderr lines" not in msg, msg
