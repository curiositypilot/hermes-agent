"""The PM worker takes the lazy-install policy from its client's request, not from hermes_cli.config.

Importing hermes_cli.config in the worker loads every bundled provider plugin (~2 s per
spawn, and a "solstice: No module named 'httpx'" warning in the stdlib PM runtime). The
client already decides the policy before spawning, so it sends that decision over stdin.
"""
from __future__ import annotations

import json
import sys

import pytest

from pm.package import InstallError
from tests.pm._fixtures import client, isolated_python, worker_toolchain

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = pytest.mark.real_machine_home


@pytest.fixture
def no_override(monkeypatch):
    from pm import install

    monkeypatch.setattr(install, "_lazy_policy_override", None)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    return install


def test_override_answers_without_importing_config(no_override, monkeypatch):
    monkeypatch.setattr(no_override, "_lazy_policy_override", True)
    # A blocked import would answer True (bootstrap) or False; neither is consulted.
    monkeypatch.setitem(sys.modules, "hermes_cli.config", None)
    assert no_override.lazy_installs_allowed() is True
    monkeypatch.setattr(no_override, "_lazy_policy_override", False)
    assert no_override.lazy_installs_allowed() is False


def test_env_disable_beats_an_allowing_override(no_override, monkeypatch):
    monkeypatch.setattr(no_override, "_lazy_policy_override", True)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    assert no_override.lazy_installs_allowed() is False


@pytest.mark.parametrize("allowed", [True, False])
def test_request_carries_the_client_policy(no_override, monkeypatch, allowed):
    from pm import client

    sent = []

    class Stop(Exception):
        pass

    class Popen:
        def __init__(self, *args, **kwargs):
            self.stdin = self
            self.stdout = self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, line):
            sent.append(json.loads(line))

        def flush(self):
            raise Stop

        def poll(self):
            return 0

    monkeypatch.setattr(no_override, "lazy_installs_allowed", lambda: allowed)
    monkeypatch.setattr(client, "runtime_environment", dict)
    monkeypatch.setattr(client, "_worker_command", lambda *a, **kw: ["worker"])
    monkeypatch.setattr(client.subprocess, "Popen", Popen)
    with pytest.raises(Stop):
        client._request("ensure", {"name": "node", "explicit": False})
    assert sent[0]["context"]["lazy_installs_allowed"] is allowed


def test_read_only_request_skips_the_policy_read(no_override, monkeypatch):
    """hermes_bootstrap asks venv_is_current in every process, managed workers included; reading
    the policy there imports hermes_cli.config before the worker binds its frozen config, and the
    worker refuses to start ("worker policy must bind before runtime imports")."""
    from pm import client

    sent = []

    class Stop(Exception):
        pass

    class Popen:
        def __init__(self, *args, **kwargs):
            self.stdin = self
            self.stdout = self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, line):
            sent.append(json.loads(line))

        def flush(self):
            raise Stop

        def poll(self):
            return 0

    def policy_read():
        raise AssertionError("read-only request read the lazy-install policy")

    monkeypatch.setattr(no_override, "lazy_installs_allowed", policy_read)
    monkeypatch.setattr(client, "runtime_environment", dict)
    monkeypatch.setattr(client, "_worker_command", lambda *a, **kw: ["worker"])
    monkeypatch.setattr(client.subprocess, "Popen", Popen)
    with pytest.raises(Stop):
        client._request("venv_is_current", {"extras": [], "plugins": []})
    assert "lazy_installs_allowed" not in sent[0]["context"]


# The worker's config import is sabotaged, so a worker that still read config would decide
# differently from the client: `None` in sys.modules makes it look absent (bootstrap: allowed),
# a raising finder makes it look broken (fail closed: refused).
_CONFIG_ABSENT = "sys.modules['hermes_cli.config'] = None"
_CONFIG_BROKEN = (
    "class _Broken:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'hermes_cli.config':\n"
    "            raise ImportError('worker imported hermes_cli.config')\n"
    "sys.meta_path.insert(0, _Broken())"
)


def _config(tmp_path, allowed: bool):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(f"security:\n  allow_lazy_installs: {str(allowed).lower()}\n", encoding="utf-8")


def test_worker_refuses_when_client_config_disallows(client, tmp_path, monkeypatch, isolated_python):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    _config(tmp_path, allowed=False)
    worker_toolchain(client, monkeypatch, isolated_python, injection=_CONFIG_ABSENT)
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        client._request("ensure", {"name": "node", "explicit": False})


def test_worker_allows_when_client_config_allows(client, tmp_path, monkeypatch, isolated_python):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    _config(tmp_path, allowed=True)
    worker_toolchain(client, monkeypatch, isolated_python, injection=_CONFIG_BROKEN)
    # Past the policy gate: the worker's next refusal is the empty test lockfile.
    with pytest.raises(InstallError, match="not in the lockfile"):
        client._request("ensure", {"name": "node", "explicit": False})
