"""Cron runs recall external memory against the job's own prompt (Retrieval·W2).

The message a cron agent receives is the assembled prompt: skill bodies, the fixed cron hint,
script output and the job prompt. Memory providers truncate their query (Hindsight: 800 chars),
so recalling against that message searched skill text / the hint instead of the job's topic.
``run_job`` now hands ``run_conversation`` a ``memory_query`` built from the job's own prompt.
"""

import pytest


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    skills_dir = hermes_home / "skills"
    skills_dir.mkdir(parents=True)
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(hermes_home / "skill-bundles"))
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    import tools.skills_tool as _skills_tool

    monkeypatch.setattr(_skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(_skills_tool, "HERMES_HOME", hermes_home)
    import agent.skill_bundles as _skill_bundles

    _skill_bundles._bundles_cache = {}
    _skill_bundles._bundles_cache_mtime = None
    import cron.scheduler as scheduler

    return hermes_home, scheduler


def _plant_skill(hermes_home, name, body):
    skill_dir = hermes_home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n\n{body}\n", encoding="utf-8")


class _SessionDB:
    def set_session_title(self, *a, **k):
        pass

    def end_session(self, *a, **k):
        pass

    def close(self):
        pass


def _run_capturing(monkeypatch, scheduler, hermes_home, job, **run_kwargs):
    """Run ``job`` through the real ``run_job`` with a fake agent; return what it received."""
    seen = {}

    class _Agent:
        def __init__(self, *a, **k):
            pass

        def run_conversation(self, prompt, *, task_id=None, **kwargs):
            seen["prompt"], seen["kwargs"] = prompt, kwargs
            return {"completed": True, "failed": False, "final_response": "ok", "turn_exit_reason": ""}

        def close(self):
            pass

    monkeypatch.setattr("hermes_state_registry.acquire", _SessionDB)
    monkeypatch.setattr("run_agent.AIAgent", _Agent)
    monkeypatch.setattr("hermes_constants.resolve_reasoning_config", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_k: {"api_key": "k", "base_url": None, "provider": "p", "api_mode": None,
                      "command": None, "args": None},
    )
    monkeypatch.setattr("tools.mcp_tool_discovery.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(scheduler, "_guard_job_credential_exfil", lambda _job: None)
    success, _doc, _final, error = scheduler.run_job(
        {"schedule_display": "manual", **job}, **run_kwargs)
    assert success is True, error
    return seen


_JOB_PROMPT = "Summarize what changed in the beancount ledger since yesterday."


def test_no_skill_job_recalls_against_its_prompt_not_the_cron_hint(cron_env, monkeypatch):
    home, scheduler = cron_env
    seen = _run_capturing(monkeypatch, scheduler, home,
                          {"id": "j1", "name": "ledger", "prompt": _JOB_PROMPT})
    assert seen["kwargs"]["memory_query"] == _JOB_PROMPT
    # The model still receives the assembled prompt, which leads with the hint: the first
    # 800 chars (Hindsight's default query cap) never reached the job's topic.
    assert _JOB_PROMPT not in seen["prompt"][:800]
    assert seen["prompt"].rstrip().endswith(_JOB_PROMPT)


def test_skill_job_recalls_against_its_prompt_not_the_skill_body(cron_env, monkeypatch):
    home, scheduler = cron_env
    _plant_skill(home, "ledger-skill", "Step one. " * 300)
    seen = _run_capturing(monkeypatch, scheduler, home,
                          {"id": "j2", "name": "ledger", "prompt": _JOB_PROMPT, "skills": ["ledger-skill"]})
    assert "Step one." in seen["prompt"]
    assert seen["kwargs"]["memory_query"] == _JOB_PROMPT


def test_per_fire_extra_prompt_joins_the_query(cron_env, monkeypatch):
    home, scheduler = cron_env
    seen = _run_capturing(monkeypatch, scheduler, home,
                          {"id": "j3", "name": "ledger", "prompt": _JOB_PROMPT},
                          extra_prompt="focus on the Revolut account")
    assert seen["kwargs"]["memory_query"] == f"{_JOB_PROMPT}\n\nfocus on the Revolut account"


def test_skill_only_job_falls_back_to_name_and_skills():
    from cron.scheduler_prompt import _cron_memory_query

    assert _cron_memory_query({"name": "daily digest", "skills": ["morning-digest"]}) == \
        "daily digest morning-digest"
    assert _cron_memory_query({"prompt": "  ", "skill": "git-backup"}) == "git-backup"
    assert _cron_memory_query({}) == ""
