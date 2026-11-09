"""Kanban section of DEFAULT_CONFIG, moved out so config_defaults.py stops growing.

Imported back and spliced into DEFAULT_CONFIG by config_defaults.py.
"""

KANBAN_DEFAULTS = {
        # Default data handling and main-model provider boundaries for Kanban workers. Unknown
        # classes fail closed; tenant defaults are applied only when a task has no explicit class.
        "data_policies": {
            "default": "internal",
            "tenant_defaults": {"work": "confidential"},
            "classes": {
                "internal": {"providers": "any", "fallback": True, "auxiliary": "any"},
                "confidential": {
                    "providers": ["anthropic", "openai-codex"],
                    "fallback": False,
                    "auxiliary": "same_provider_or_fail",
                },
            },
        },
        # Auto-subscribe the originating gateway/TUI session to completion + block events when
        # kanban_create is called from a session with a persistent delivery channel. Disable for
        # profiles that prefer explicit kanban_notify-subscribe calls per task.
        "auto_subscribe_on_create": True,
        # Poll and deliver Kanban subscriptions from this gateway. Disable on profiles that do
        # not own notification subscriptions to avoid an idle five-second board probe.
        "notify_in_gateway": True,
        # Run the dispatcher inside the gateway process (~300µs per idle tick). False only if you
        # run it as a separate unit or don't want the gateway spawning workers.
        "dispatch_in_gateway": True,
        # Auto-claim tasks in the review column and spawn the assigned profile with the bundled
        # sdlc-review skill. Disable where every review is done manually from the dashboard.
        "review_dispatch": True,
        # Seconds between dispatcher ticks. Lower = snappier pickup; higher = less SQL pressure.
        "dispatch_interval_seconds": 60,
        # Auto-block after this many consecutive non-success attempts (spawn_failed, timed_out,
        # crashed) for the same task/profile. Reassignment resets the streak.
        "failure_limit": 2,
        # Worker stdout/stderr log rotation at spawn time (2 MiB + one backup). Raise to keep more
        # early failure evidence from long-running workers.
        "worker_log_rotate_bytes": 2 * 1024 * 1024,
        "worker_log_backup_count": 1,
        # Toolsets a dispatcher-spawned worker never gets, subtracted from the assignee's CLI
        # selection at spawn and again at schema assembly. Workers run headless: `clarify` has
        # nobody to answer and the browser vault needs a user-side unlock/consent prompt. MCP
        # server names work too ("granola" meeting notes: 8k schema chars per worker turn, no
        # worker use); a name the profile does not enable is a no-op.
        "worker_disabled_toolsets": ["clarify", "browser_vault", "granola"],
        # Extra tools deferred only inside a dispatcher-owned worker
        # (tool_search.py effective_defer_tools, gated on owned_kanban_task()); [] turns it off.
        "worker_defer_tools": ["delegate_task", "session_search"],
        # Worker sandbox (hermes_cli/kanban_sandbox.py): "off" | "bwrap" (bubblewrap, Linux), read at
        # every spawn. Run `hermes kanban sandbox-selftest` on the host before turning it on.
        "sandbox": "off",
        # Tenants the sandbox applies to; empty = every tenant.
        "sandbox_tenants": [],
        # Cards with any of these skills run unsandboxed (exact name), e.g. hindsight-ops.
        "sandbox_exempt_skills": [],
        # Profile for the root/orchestration task after Triage decomposition; "" = default profile.
        # Does not control the decomposer LLM path (see auxiliary.kanban_decomposer).
        "orchestrator_profile": "",
        # Assignee when the orchestrator can't match one to an installed profile; "" = default
        # profile. A task never ends up with assignee=None.
        "default_assignee": "",
        # Global concurrency cap (#33488): when set to a positive int, the HOST never has more than N tasks
        # in 'running' at once — counted across every active board and across both the ready and review
        # dispatch lanes (workers are OS processes sharing one machine's memory, so the cap bounds the
        # machine, not each board; OOF-30). Unset (None) means "derive from system memory" (OOF-30/OOF-77):
        # the dispatcher caps concurrency at roughly MemTotal / 512 MiB, clamped to [2, 8] — e.g. 2 workers
        # on a 1 GiB VM. On hosts where total memory can't be read (macOS/Windows), unset falls back to no
        # cap. Set an explicit value to override the derived default in either direction.
        "max_in_progress": None,
        # Per-profile cap: positive int = no profile runs more than N workers; excess defers a tick. Unset
        # (None) means "no per-profile cap" — backward-compatible with existing installs. Useful for fan-out
        # workflows that would otherwise saturate one profile's local model / API quota / browser pool. #21582.
        "max_in_progress_per_profile": None,
        "worker_memory_max_mb": None,  # Worker scope MemoryMax (MiB); lifts the 4 GiB default, still <= RAM/2
        # Per-home claim allowlist for boards shared across Hermes homes (#110995): profile names
        # this home's dispatcher may claim (list or comma-separated string). None = any existing
        # profile is claimable. Set = fail-closed (an empty list claims nothing). Every home has a
        # root profile named "default", so on a shared kanban.db every home can otherwise claim
        # default-assigned cards.
        "dispatch_profiles": None,
        # Auto-run the decomposer on Triage tasks every tick. False = manual via `hermes kanban
        # decompose <id>` or the dashboard's Decompose button.
        "auto_decompose": True,
        # Max triage tasks decomposed per tick, bounding the aux-LLM burst from a bulk load. Excess
        # defers to the next tick.
        "auto_decompose_per_tick": 3,
        # Complexity-tier model routing (hermes_cli/kanban_routing.py). A ready card with an S/M/L
        # label and no pinned model gets its worker model from tiers.<label>: the first candidate
        # whose provider is not rate-limited (credential pool exhausted/benched, or a worker on this
        # board exited rate_limited on it within cooldown_seconds). Precedence: --model pin > tier >
        # profile default. Re-read every tick. Example tiers entry:
        #   S: [{model: gemini-flash-latest, provider: antigravity}]
        #   L: [{model: claude-opus-5-5, provider: anthropic, reasoning: high}]
        "routing": {
            "enabled": False,
            "tiers": {},
            # Ordered candidates (same shape as a tier) for review-lane runs (sdlc-review). When
            # set they replace the card's pinned model and reasoning for the reviewer; empty =
            # reviewers keep the card pin / profile model. on_exhausted applies here too.
            "review": [],
            # Unlabeled cards: "profile" (assignee's own model) or a tier name (S/M/L).
            "unlabeled": "profile",
            # When every candidate of the requested tier is unavailable, try the next larger tier.
            "escalate": True,
            # When nothing is available: "wait" holds the card (guard reason tier_exhausted, no
            # failure counted) or "profile" spawns on the assignee's own model.
            "on_exhausted": "wait",
            # How long a candidate is skipped after a routed worker exited rate_limited on it.
            "cooldown_seconds": 900,
            # Estimate S/M/L with the auxiliary kanban_estimator model for unlabeled ready/todo
            # cards each tick (one aux call per card, up to auto_label_per_tick).
            "auto_label": False,
            "auto_label_per_tick": 3,
            # Quota gate: hold low-priority cards off a provider whose live quota is low. Reads the
            # quota-snapshot file and the Anthropic quota mode each tick (no network). anthropic:
            # mode.json's min_priority (closed 4, critical 3, tight 2, normal/open 0); every other
            # provider, and anthropic without a fresh mode: the fixed level gate (binding headroom
            # < 10% needs P2, < 3% needs P3; kanban_routing.LEVEL_GATE). A gated candidate is
            # skipped and the walk moves on; if nothing is left the card waits as quota_hold (no
            # failure counted; on_exhausted: profile does not override it). Unknown headroom
            # (missing/stale/unreadable file, provider absent, a kept reading older than
            # max_age_seconds) = gate open.
            "quota_gate": {
                "enabled": False,
                # "" = <HERMES_HOME>/data/quota/headroom.json
                "headroom_file": "",
                # Older file or reading = headroom unknown = gate open; older mode.json = level gate.
                "max_age_seconds": 1800,
                # Pay-as-you-go providers a card may NOT fall through to once a gate skip happened
                # in its walk (a plain rate-limit fallthrough still reaches them).
                "payg_providers": ["openrouter"],
                # Anthropic quota mode written by scripts/quota_mode.py ("" =
                # <HERMES_HOME>/data/quota/mode.json). "open" = Opus first in every tier + review
                # lane and quota-parked cards woken.
                "mode_file": "",
            },
            # Force the quota mode until the week resets (closed|critical|tight|normal|open, or
            # null). Read by scripts/quota_mode.py, not by the dispatcher.
            "quota_mode_override": None,
        },
        # Running tasks with no heartbeat (last_heartbeat_at) for this many seconds are reclaimed to
        # ready on the next tick; a still-running local worker is terminated first. 0 = off.
        "dispatch_stale_timeout_seconds": 14400,
        # A `scheduled` card with no date (no --until, no "until YYYY-MM-DD" in its reason) wakes
        # as a sticky `blocked` ("ask") this many days after it was parked. Dated cards wake on
        # their date into ready (--then start) or blocked (--then ask). 0 = undated cards never wake.
        "schedule_recheck_days": 7,
        # Each tick, requeue 'running' cards with broken claim bookkeeping (claim_lock or
        # claim_expires NULL with a dead worker) that TTL/crash/stale recovery can't see. False
        # keeps orphans frozen for manual forensics.
        "reconcile_orphans": True,
        # Notify subscriptions survive `done` (completion is reversible) and are removed on archive.
        # On boards that never archive, the notifier GC purges subscriptions for tasks done with no
        # activity for this many days so stale rows aren't scanned forever. 0 = off.
        "done_sub_retention_days": 30,
    }
