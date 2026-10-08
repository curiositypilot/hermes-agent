# Upstream sync 2026-10-08 — NousResearch/hermes-agent → MB's fork

Branch `sync/upstream-20261008` (worktree `.worktrees/t_91ce4a22`), NOT merged to main.
Upstream head merged: f549953a38 (2026-10-08 03:43 -0700); merge-base with fork main: 95f20517c2
(2026-09-22). Upstream brought 10,453 commits; the fork carried 131 commits on top of the base, plus
16 more that landed on main while this sync ran (merged in f5d39b01e5).

Branch head: see `git log --oneline -1 sync/upstream-20261008` (last commit of this card is listed in
the card's completion metadata). First-parent history of the sync:

```
e04016f43e Merge upstream/main (2026-10-08) into fork main            (41 conflicts resolved)
c50d1fcaa5 test(cron): quota-hold day-2 alert follows upstream signature; hermes_yaml
cc195246b1 test(agent): import HISTORICAL_TASK_HEADING (summary-prefix test)
06c4b7998b test(gateway): queued-followup fakes carry reply_expected
14c2e6313e test(tools): kanban record_fallback tests import os
95d71fde14 fix(hindsight): client dependency through pm.extras; drop runtime auto-upgrade
f5d39b01e5 Merge fork main (86344e4466) into the sync branch           (9 conflicts resolved)
5841fc87ad fix(merge): pinned-kanban fast-fail uses provider_retry_after_seconds; data-class test
6353314ac1 test: fork tests use platforms("linux") (upstream retired linux_only)
```

## 1. Conflicts and resolutions

### 1a. Upstream merge (e04016f43e) — 41 conflicted files

Verified against the merge result: 31 files carry both sides, 3 are the fork's version unchanged
(hindsight plugin files + one test), 1 is upstream's version (`main_install_repair.py`), 4 are deleted
(upstream). Per file:

| File | Resolution |
|---|---|
| `agent/account_usage.py` | Fork percent semantics (callers always pass 0-100; upstream's `fraction` scaling dropped) on upstream's Codex window labels. |
| `agent/agent_init.py` | Fork fallback-chain init (chat_only filter, kanban pinned route) moved into new sibling `agent/agent_init_fallback.py` to respect upstream's facade/sibling split. |
| `agent/auxiliary_client.py` | Upstream `_resolve_auto_route`; fork data-class provider policy (`agent/provider_policy.py`: `_AuxPolicy`, `auxiliary_permits`, same-provider pin) kept. |
| `agent/chat_completion_helpers.py` | Upstream version; fork entitlement no-bench behaviour kept (`agent/credential_pool*`, `fallback_cooldown`). |
| `agent/context_compressor.py` | Upstream SUMMARY_PREFIX; fork no-built-in-memory clause via `.replace()` kept (`HISTORICAL_TASK_HEADING` is upstream's name). |
| `agent/conversation_loop.py` | Upstream loop (now has `title_user_message` natively); fork `memory_query` plumbing kept. |
| `agent/prompt_builder.py` | Upstream assembly + fork no-built-in-memory compaction note. |
| `agent/skill_commands.py` | Upstream; fork `auto_load_activation_note` / `strip_auto_loaded_skill_blocks` (memory-path scaffolding strip) kept. |
| `agent/turn_api_error.py` | Upstream phase file; fork `_pinned_kanban_rate_limit_fast_fail` kept, now calls `agent.retry_utils.provider_retry_after_seconds` (5841fc87ad). |
| `agent/turn_facade.py` | Upstream facade; fork `memory_query`/`title_user_message` parameters kept. |
| `apps/desktop/src/plugins/kanban/{drawer.tsx,i18n.ts,types.ts}` | Fork scheduled-wake UI (acd566cbf3) kept on upstream's kanban drawer. |
| `cron/AGENTS.md` | Upstream text + fork notes (quota hold, kanban routing). |
| `cron/jobs.py` | Upstream store; fork quota-hold fields kept. |
| `cron/quota_hold.py` | Rewritten: fork 24h cap + route stamp merged with upstream's recovery fire; day-2 alert follows upstream's duration-masked incident signature (c50d1fcaa5). |
| `cron/scheduler.py` | Upstream `_job_fallback_chain` (pinned jobs never borrow the global chain, #100437) with fork `drop_chat_only_entries` folded inside; fork cron memory_query kept. |
| `gateway/run_inbound.py`, `gateway/run_turn.py` | Upstream turn pipeline; fork synthetic-prompt recall skip and reply-quote `memory_query` kept. |
| `hermes_cli/doctor.py` | Fork declared-deps check kept, body moved to new `hermes_cli/declared_deps.py` (no `--fix`: upstream deleted the uv repair path). |
| `hermes_cli/kanban_decompose.py` | Upstream decomposition; fork `list_triage_ids(exclude_finished=)` (never re-plan finished work) kept. |
| `hermes_cli/main_install_repair.py` | Upstream (PM owns repair); fork uv-repair hook dropped (PM replaces it). |
| `hermes_cli/update_cmd_deps.py` | Deleted by upstream (updater rewritten on PM); fork venv-health probe superseded by `hermes pm doctor` + `declared_deps.py`. |
| `hermes_state_timeline.py` | Upstream; synthetic-prompt regex now imported from fork `agent.memory_provider.SYNTHETIC_PROMPT_RE` (one owner) instead of a local copy. |
| `plugins/memory/hindsight/README.md`, `__init__.py`, `settings.py` | Upstream deleted the bundled plugin (moved to `plugin-catalog/hindsight.yaml`); fork keeps the bundled provider with all fork controls (retain_*, recall_*, reply-quote recovery, scaffolding strip). Dependency path moved from retired `tools.lazy_deps` to `pm.extras` (95d71fde14). |
| `plugins/memory/openviking/__init__.py` | Deleted by upstream (catalog plugin); fork's scaffolding-strip edit there dropped with it. |
| `pyproject.toml` | Upstream (PM-era, `[tool.uv] environments = python>=3.14`); fork `hindsight` extra kept; `uv.lock` relocked with it. |
| `scripts/run_tests_parallel.py` | Upstream runner + fork cgroup-v2 memory-limit worker sizing (`_detect_cgroup_v2_memory_limit`). |
| `tests/agent/test_memory_agent_context.py`, `test_memory_session_switch.py` | Fork tests adapted to upstream fixtures. |
| `tests/cron/test_cron_failure_alert_remediation_hint.py` | Fork version kept as-is. |
| `tests/cron/test_quota_hold.py` | Fork tests on the merged quota-hold design (50/50 after c50d1fcaa5). |
| `tests/hermes_cli/test_doctor.py` | Fork declared-deps cases on upstream's doctor. |
| `tests/hermes_cli/test_update_venv_health.py`, `tests/plugins/memory/test_mem0_setup.py` | Deleted with their subjects (upstream). |
| `tests/plugins/memory/test_hindsight_provider.py` | Fork suite kept; dependency tests rewritten for `pm.extras` (210 pass). |
| `tests/tools/test_kanban_tools.py` | Fork record_fallback tests kept on upstream's tool module. |
| `tools/cronjob_tools.py` | Upstream tool + fork `interpreter`/quota fields. |
| `tools/skills_tool.py` | Upstream; fork `_excluded_skill_location` (archive/backup dirs excluded lexically and after symlink resolution) kept. |

### 1b. Fork main → sync branch (f5d39b01e5) — 9 conflicted files

| File | Resolution |
|---|---|
| `cron/jobs.py`, `hermes_cli/cron.py`, `hermes_cli/subcommands/cron.py`, `tools/cronjob_job_args.py`, `tools/cronjob_tools.py` | Both-sides additive: fork `interpreter` field (sync branch) + main's `data_class` field; union. |
| `cron/scheduler.py` | Upstream pinned-aware `_job_fallback_chain` kept; main's data-class `fallback_permits` filter added inside the walk. Consequence: a cron job with its own `provider`/`model`/`base_url` no longer falls back (upstream #100437); `tests/cron/test_cron_data_class.py` now walks an unpinned job via `cron.default_provider`. |
| `tests/conftest.py` | Upstream split the env filter into `tests/_fixtures/env_filter.py`; main's `HERMES_DATA_CLASS` and `INVOCATION_ID` scrubs ported there. |
| `tests/gateway/test_fallback_chain_reload.py` | Main's pinned-route test kept; upstream's deletion of the source-reading test kept (banned shape). |
| `tests/hermes_cli/test_kanban_default_assignee.py` | Upstream's tmp_path fixture (no module purge needed). |
| `tests/cron/test_recurring_eagain_redispatch.py` (deleted upstream, modified on main) | Kept main's version: it passes on the merged tree. |

## 2. Merge fallout fixed (tests)

- `tests/agent/test_summary_prefix_semantics.py`: import `HISTORICAL_TASK_HEADING` (upstream name).
- `tests/gateway/test_reply_memory_query.py`: queued-followup fakes carry upstream's `reply_expected`.
- `tests/tools/test_kanban_tools.py`: missing `import os`.
- Hindsight: `tools.lazy_deps.ensure/install_specs` are retired upstream (raise). Plugin now uses
  `pm.extras.ensure_import("hindsight")` (anchor `hindsight_client` added to `pm/extras.py`), the
  initialize()-time auto-upgrade is a warning naming `hermes pm install --extra hindsight`, and
  `post_setup` syncs the extra through PM (embedded `hindsight-all` is not a declared extra: told, not installed).
  `tests/pm/test_extras.py` flips one upstream assertion (hindsight IS a core extra in this fork).
- Markers: `@pytest.mark.linux_only` → `platforms("linux")` in `tests/hermes_cli/test_kanban_test_gate.py`,
  `tests/test_hermes_constants.py` (collection error otherwise).

## 3. Upstream changes that affect MB's setup (adopt / act on)

1. **PM-managed environment, Python 3.14 only.** `uv.lock` resolves for `python_version >= '3.14'`;
   PM (`pm/`, `setup-hermes.sh`, `hermes pm install|lock|repair|doctor`) owns the venv; the live
   `~/.hermes/hermes-agent/venv` is Python 3.11 and cannot run this tree. Landing = `setup-hermes.sh
   --runtime-only` (or `hermes pm install`) + systemd unit refresh (`hermes gateway install` is PM-aware)
   + the drain-aware restart. `uv pip install -e .` into the old venv is no longer the procedure; update
   the `hermes-fork-management` skill § After pulling.
2. **`tools.lazy_deps` retired → `pm.extras`.** Any fork code/skill that lazily installs packages must
   use `pm.ensure_import(extra)`; raw pip/uv against the Hermes venv is out of policy.
3. **Pinned cron jobs never borrow `fallback_providers`** (#100437): a job with `provider`/`model`/
   `base_url` fails instead of falling back. Audit MB's cron jobs with pins (`hermes cron list`).
4. **Tests:** `scripts/run_tests.sh` re-execs into the PM test env unless `HERMES_PYTHON` points at an
   interpreter with pytest; `tests/_fixtures/` holds the env filter and platform gating; `platforms(...)`
   is the only OS marker. The home-IO guard whitelists the checkout but not other paths under
   `~/.hermes`, so run suites from a clone outside `~/.hermes` (this card used `/tmp/hermes-t91`).
5. **Memory providers left core:** hindsight/honcho/supermemory/mem0/openviking are catalog plugins
   upstream (`hermes_cli/memory_provider_migration.py` auto-installs for configured homes). The fork
   keeps hindsight bundled; moving MB's controls into the catalog plugin repo would retire the largest
   carried patch. Also upstream now ships `title_user_message` natively (fork plumbing retired there).

## 4. Patches now upstreamed (drop from fork-patches.md)

- `title_user_message` on `run_conversation` (upstream `agent/conversation_loop.py`).
- Venv health probe in the updater (`hermes_cli/update_cmd_deps.py`): superseded by PM (`hermes pm doctor`).
- OpenViking scaffolding strip: provider left core; nothing to carry.

Still fork-only (verified absent upstream): `drop_chat_only_entries`, `is_low_signal_prompt`,
`SYNTHETIC_PROMPT_RE`, `kanban_routing` (complexity tiers, quota gate, pace), `worker_disabled_toolsets`,
dated schedule (`scheduled_until`), Hindsight retain/recall controls, `kanban_test_gate` related-tests
gate, declared-deps doctor check, account-usage percent semantics, cron `interpreter` + `data_class`.

## 5. Test evidence

Results (clone `/tmp/hermes-t91`, py3.14 venv, head a9af8ddbeb + the migration-test fix):

- `pytest tests/hermes_cli -k kanban`: **690 passed, 1 failed, 4 skipped** (baseline 27 failed / 590 passed).
  The one failure, `test_kanban_db.py::test_concurrent_create_with_same_idempotency_key_yields_one_task`
  (main's 671408943c), is untriaged: re-run alone to tell flake from merge fallout.
- Fork-focused related-tests gate (`KANBAN_TEST_BASE=f549953a38`, 450 files, `scripts/run_tests.sh -j 16`):
  5052 passed / 26 failed / 8 files hit the 300 s per-file timeout under -j 16 load
  (`test_error_classifier`, `test_run_agent`, `test_doctor`, 5 `tests/e2e/core/*` files: re-run alone).
  Of the 26: 5 fixed in a9af8ddbeb, 1 fixed after (`test_memory_provider_migration`: hindsight is bundled
  here, honcho is the left-core case), 2 were the host's `SSL_CERT_FILE` pointing at the old venv's certifi
  (pass with it unset), 2 are timing bounds in `tests/tools/test_delegate_memory_context.py` (fork
  t_2354171e; `_FakeManager(timeout=0.2)` build took ~0.95 s = the slow recall ran to completion, so the
  delegation-context timeout may not be honoured on the merged `tools/delegate_tool*` path: CHECK).
  Still open: `test_background_review.py` (3), `test_auxiliary_data_class_policy.py` (8),
  `test_cron_failure_alert_remediation_hint.py` (4), `test_kanban_db.py` (1),
  `tests/tui_gateway/test_kanban_resume_guard.py` (teardown error in `tests/conftest.py` fixture,
  circular import of `tui_gateway.server`). Raw logs: `~/.hermes/cache/scratch/t91_forkgate_run.txt`,
  `t91_kanban_run.txt`.

Environment: clone `/tmp/hermes-t91` of the branch, interpreter `~/.hermes/cache/scratch/sync-venv-t91`
(Python 3.14, upstream lock + pytest). Results are recorded in the card's completion metadata and in
`~/.hermes/cache/scratch/t91_*_run.txt`.

- Completion contract `python3 scripts/kanban_test_gate_related.py` (base `main`): related set is
  4,592 test files / ~41k tests (the diff vs main is the whole upstream merge). Measured pace at
  `-j 8`: ~150 tests/min → ~4.5 h, beyond the gate's 3600 s timeout; the worktree also has no PM test
  env, so the kernel-run gate re-execs into `run-in-hermes-env`. Not satisfiable as configured for a
  sync card; run with `KANBAN_TEST_BASE=f549953a38` (fork-touched files only, 450 test files) instead.
