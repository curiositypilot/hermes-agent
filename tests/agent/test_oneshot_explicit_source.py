"""A finite run under an explicit ``--source`` (``hermes chat -q --source tool``) is one-shot (t_f4bcefd3).

The gateway owns the session, so the one-shot fact is the frozen ``oneshot`` create flag, not the source label:
the agent's tool build drops ``clarify`` and its prompt drops the mid-turn steering note. The same source
without the flag is interactive, the control.
"""

from types import SimpleNamespace


def _surface(tmp_path, monkeypatch, **params):
    from agent import system_prompt
    from agent.agent_init import _load_tools
    from gateway.session_local_inventory import _policy_tools
    from gateway.session_policy import build_policy, policy_scope

    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    policy = build_policy({"source": "tool", "cwd": str(tmp_path), "model": "m", **params}, {})
    with policy_scope(policy):
        agent = SimpleNamespace(platform="cli", quiet_mode=True)
        _load_tools(agent, ["hermes-cli"], None)
        inventory = {t["function"]["name"] for t in _policy_tools(policy, {})[1]}
    # Prompt built outside the policy scope: the agent's frozen one-shot answer still decides it.
    agent.__dict__.update(valid_tool_names={"terminal"}, model="m", _tool_use_enforcement=False,
                          _execution_guidance=False, _kanban_worker_guidance="")
    guidance = "\n".join(p for p in system_prompt._guidance_parts(agent) if p)
    return {t["function"]["name"] for t in agent.tools}, inventory, guidance


def test_explicit_source_finite_run_drops_clarify_and_steering_note(tmp_path, monkeypatch):
    tools, inventory, guidance = _surface(tmp_path, monkeypatch, oneshot=True)
    assert "clarify" not in tools and "clarify" not in inventory
    assert "OUT-OF-BAND USER MESSAGE" not in guidance


def test_explicit_source_interactive_run_keeps_clarify_and_steering_note(tmp_path, monkeypatch):
    tools, inventory, guidance = _surface(tmp_path, monkeypatch)
    assert "clarify" in tools and "clarify" in inventory
    assert "OUT-OF-BAND USER MESSAGE" in guidance
