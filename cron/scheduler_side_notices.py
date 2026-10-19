"""Side notices of a successful cron run go to a card, not into the report (t_2cd8f62d).

Two notices used to ride inside the delivered text of a run that SUCCEEDED, putting raw error text
in front of the reader: the agent's file-mutation verifier footer (``agent/turn_finalizer.py``)
and the pre-agent provider-fallback line (#74349). With ``cron.failure_route`` configured each one
is handed to the router first (``cron/failure_route.py``, one signature per job and notice kind);
exit 0 means the router owns it and the notice is left out of the report. Any other outcome
(router unset, non-zero exit, timeout) re-attaches it exactly as before, so with no router the
delivered text is unchanged and a notice is never dropped without a card.
"""

from __future__ import annotations

from typing import Optional


def defer_verifier_footer(agent) -> None:
    """Make the agent store its verifier footer instead of appending it to the response."""
    agent._defer_file_mutation_footer = True
    agent._deferred_file_mutation_footer = ""


def _deferred_footer(agent) -> str:
    footer = getattr(agent, "_deferred_file_mutation_footer", "")
    return footer if isinstance(footer, str) else ""


def final_response_with_side_notices(
    result: dict, job: dict, job_id: str, job_name: str, agent, fallback_notice: Optional[str],
    AIAgent,
) -> str:
    """``_final_response_from_result`` plus the run's side notices, each routed to a card when
    the router takes it and otherwise attached where it used to be."""
    from cron import scheduler as _sched
    from cron.failure_route import route_failure

    footer = _deferred_footer(agent)
    try:
        final_response = _sched._final_response_from_result(result, job_id, job_name, AIAgent)
    except RuntimeError as exc:
        if not footer:
            raise
        # A failed run's notice is already routed; the footer rides in it.
        raise RuntimeError(f"{exc}\n\n{footer}") from exc
    # The finalizer only stores a footer for a non-empty, uninterrupted response (its own gate).
    # output_file=None on both routes: they fire before this run's report is saved, so the job's
    # newest file is the PREVIOUS run's report, the wrong evidence for this notice.
    if footer and route_failure(job, footer, signature=f"cron-verifier:{job_id}",
                                output_file=None) is None:
        # Not routed: rebuild the response the finalizer would have produced, footer included.
        raw = result.get("final_response") or ""
        final_response = _sched._final_response_from_result(
            dict(result, final_response=raw.rstrip() + "\n\n" + footer), job_id, job_name, AIAgent)
    if (fallback_notice and final_response.strip()
            and not _sched._is_cron_silence_response(final_response)
            and _sched._cron_failure_marker_error(final_response) is None
            and route_failure(job, fallback_notice, signature=f"cron-provider-fallback:{job_id}",
                              output_file=None) is None):
        # Pre-agent provider switch (#74349) rides with the delivered report; silence and the
        # agent-declared failure marker keep their first-line/whole-response contract.
        final_response = f"{fallback_notice}\n\n{final_response}"
    return final_response
