"""A systemd-supervised ``hermes dashboard`` (``Restart=always``) retires itself between turns on code
skew only when ``dashboard.retire_on_code_skew`` is the literal ``true``, and never while a PTY shell
it hosts is open (the supervisor stops the whole cgroup, so the shell would die with it)."""

from __future__ import annotations

import pytest
from starlette.datastructures import State

from hermes_cli.web_server import _supervised_pty_hold, _supervised_skew_retirement


def test_opt_in_arms_only_a_plain_supervised_dashboard():
    assert _supervised_skew_retirement(ssh_isolated=False, desktop_owned=False,
                                       dashboard_cfg={"retire_on_code_skew": True}) is True


@pytest.mark.parametrize("cfg", [{}, {"retire_on_code_skew": False}, {"retire_on_code_skew": "true"},
                                 {"retire_on_code_skew": 1}])
def test_anything_but_a_literal_true_stays_off(cfg):
    assert _supervised_skew_retirement(ssh_isolated=False, desktop_owned=False, dashboard_cfg=cfg) is False


@pytest.mark.parametrize("ssh_isolated,desktop_owned", [(True, False), (False, True)])
def test_owned_backends_keep_their_own_branch(ssh_isolated, desktop_owned):
    assert _supervised_skew_retirement(ssh_isolated=ssh_isolated, desktop_owned=desktop_owned,
                                       dashboard_cfg={"retire_on_code_skew": True}) is False


class _Registry:
    def __init__(self, live=0, error=None):
        self.live, self.error = live, error

    def live_count(self):
        if self.error:
            raise self.error
        return self.live


def _state(host=None):
    state = State()
    if host is not None:
        state.host_terminals = host
    return state


def test_dashboard_state_without_host_terminals_does_not_hold():
    # The dashboard surface never sets app.state.host_terminals: absent is not a hold.
    assert _supervised_pty_hold(_state(), _Registry(0))() is None


def test_no_open_shell_does_not_hold():
    assert _supervised_pty_hold(_state(_Registry(0)), _Registry(0))() is None


def test_an_open_shell_holds():
    assert _supervised_pty_hold(_state(), _Registry(1))() == "pty:1"
    assert _supervised_pty_hold(_state(_Registry(1)), _Registry(0))() == "pty:1"
    assert _supervised_pty_hold(_state(_Registry(1)), _Registry(2))() == "pty:3"


def test_an_unreadable_registry_holds():
    assert _supervised_pty_hold(_state(_Registry(error=RuntimeError("x"))), _Registry(0))() == "pty:unknown"
    assert _supervised_pty_hold(_state(), _Registry(error=RuntimeError("x")))() == "pty:unknown"
