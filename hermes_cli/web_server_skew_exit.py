"""Code-skew retirement for backends nothing on the host restarts when the install moves.

Two owners use it. A Desktop-owned ``hermes serve --isolated`` backend reached over SSH belongs to a
Desktop on another machine: the host's updater never restarts it (only the remote client holds its
token and owner nonce), and the idle watchdog (``web_server_idle_exit``) only fires once every
client has left, which a connected Desktop never does. A supervised dashboard (``hermes dashboard``
under systemd ``Restart=always`` with ``dashboard.retire_on_code_skew: true``) has the same gap:
nothing restarts it when a git landing moves the checkout. Either way it keeps serving the code it
imported at startup against a tree that has moved on, and lazily imports new modules into an old
process.

This watchdog polls ``gateway.code_skew`` (the boot revision the serve lifespan records, compared
with the checkout on disk) and, once no update is in flight, no caller hold is active (open PTY
shells on a supervised dashboard) and the retirement fence proves the backend idle (and closes
admission so no turn can start mid-teardown), exits cleanly. Its owner (the client's reconnect
path, or the service supervisor) respawns it from the new code. Every uncertain read (unknown
revision, busy or indeterminate ledgers, a live update) keeps the process up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Iterable, Optional

if TYPE_CHECKING:
    from hermes_cli.pty_session import PtySessionRegistry

_log = logging.getLogger(__name__)

DEFAULT_SKEW_POLL_S = 30.0
# Two consecutive observations: a checkout mid-rewrite (ref file replaced, packed-refs repacked) can
# briefly read differently; a real update reads the new revision on every poll after it finishes.
_CONFIRMATIONS = 2

# ``(boot_rev, disk_rev)`` when the checkout drifted since boot, else None (``detect_code_skew``).
Skew = Optional[tuple[str, str]]


def should_retire_for_skew(*, skew: Skew, update_in_progress: bool) -> bool:
    """Retire only on a proven code change, and never while the tree may still be mid-swap."""
    return bool(skew) and not update_in_progress


def _update_in_progress() -> bool:
    try:
        from hermes_cli.update_lock import update_in_progress

        return update_in_progress()
    except Exception:
        return True  # cannot prove the swap is over: wait


def pty_shell_hold(registries: Callable[[], Iterable[PtySessionRegistry]]) -> Callable[[], Optional[str]]:
    """Retirement hold for a supervised dashboard: ``"pty:<n>"`` while any PTY shell is open.

    ``registries`` yields the registries that exist right now. An absent one (the dashboard surface
    never sets ``app.state.host_terminals``) is not yielded, so it never holds. A registry that
    exists but cannot be read holds as ``"pty:unknown"``: fail closed, a shell may be live.
    """
    def _hold() -> Optional[str]:
        try:
            live = sum(registry.live_count() for registry in registries())
        except (AttributeError, TypeError, RuntimeError):
            return "pty:unknown"
        return f"pty:{live}" if live else None

    return _hold


def _run_retirement_watchdog(server, *, observe: Callable[[], Optional[str]], fence, poll_s: float,
                             max_polls: Optional[int], thread_name: str) -> threading.Thread:
    """Daemon thread shared by the retirement watchdogs: ``observe`` returns the log
    reason while retirement is warranted (else None); after ``_CONFIRMATIONS`` consecutive reasons
    the backend exits only through the retirement fence (proven idle, admission closed first)."""
    if fence is None:
        from hermes_cli.backend_retirement import retirement

        fence = retirement

    def _loop() -> None:
        seen = polls = 0
        while not getattr(server, "should_exit", False) and (max_polls is None or polls < max_polls):
            polls += 1
            reason = observe()
            seen = seen + 1 if reason else 0
            if seen >= _CONFIRMATIONS:
                permit = fence.prepare()
                if permit.get("ok") and fence.commit(permit.get("token")).get("ok"):
                    _log.warning("%s", reason)
                    server.should_exit = True
                    return
            time.sleep(poll_s)

    thread = threading.Thread(target=_loop, daemon=True, name=thread_name)
    thread.start()
    return thread


def start_code_skew_watchdog(server, *, skew_fn: Optional[Callable[[], Skew]] = None,
                             update_probe: Callable[[], bool] = _update_in_progress,
                             fence=None, poll_s: float = DEFAULT_SKEW_POLL_S,
                             max_polls: Optional[int] = None,
                             label: str = "SSH-isolated backend",
                             respawner: str = "its client",
                             thread_name: str = "ssh-isolated-skew-watchdog",
                             hold_fn: Optional[Callable[[], Optional[str]]] = None) -> threading.Thread:
    """Daemon thread that sets ``server.should_exit`` once the loaded code is provably stale and
    the backend is provably idle. While ``hold_fn`` returns a reason (e.g. ``"pty:1"``) a stale
    backend stays up and the confirmation count restarts. ``max_polls`` bounds the loop for tests."""
    if skew_fn is None:
        from gateway.code_skew import detect_code_skew

        skew_fn = detect_code_skew
    read_skew: Callable[[], Skew] = skew_fn
    last_hold: list[Optional[str]] = [None]

    def _held() -> bool:
        hold = hold_fn() if hold_fn is not None else None
        if hold and hold != last_hold[0]:
            _log.info("%s is stale but holds retirement: %s", label, hold)
        last_hold[0] = hold or None
        return bool(hold)

    def _observe() -> Optional[str]:
        skew = read_skew()
        if not (skew and should_retire_for_skew(skew=skew, update_in_progress=update_probe())):
            return None
        if _held():
            return None
        return ("{} loaded {} but the install is now at {}; "
                "retiring so {} respawns it on the new code.".format(label, *skew, respawner))

    return _run_retirement_watchdog(server, observe=_observe, fence=fence, poll_s=poll_s,
                                    max_polls=max_polls, thread_name=thread_name)
