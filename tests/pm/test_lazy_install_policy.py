"""PM reads boolean install policy from real config, without CLI output."""

import pytest

import pm
from hermes_cli.config import get_config_path


@pytest.mark.parametrize("content,expected", [
    ("{}\n", True),
    ("security:\n  allow_lazy_installs: true\n", True),
    ("security:\n  allow_lazy_installs: false\n", False),
    ("security:\n  allow_lazy_installs: 'false'\n", False),
    ("security: [\n", False),
])
def test_real_policy_is_boolean_and_fails_closed(monkeypatch, capsys, content, expected):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    config = get_config_path()
    config.write_text(content, encoding="utf-8")

    assert pm.lazy_installs_allowed() is expected
    assert capsys.readouterr().out == ""
    assert config.read_text(encoding="utf-8") == content


def test_warm_policy_does_not_reparse(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    get_config_path().write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    assert pm.lazy_installs_allowed() is True

    calls = []
    real_load = config_mod.fast_safe_load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(config_mod, "fast_safe_load", counting_load)
    assert [pm.lazy_installs_allowed() for _ in range(5)] == [True] * 5
    assert calls == []


def test_policy_follows_config_edit(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    config = get_config_path()
    config.write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    assert pm.lazy_installs_allowed() is True

    config.write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")
    assert pm.lazy_installs_allowed() is False


def test_broken_config_fails_closed_every_call(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    config = get_config_path()
    config.write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    assert pm.lazy_installs_allowed() is True

    config.write_text("security: [\n", encoding="utf-8")
    assert pm.lazy_installs_allowed() is False
    assert pm.lazy_installs_allowed() is False


def test_internal_disable_overrides_enabled_config(monkeypatch):
    get_config_path().write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

    assert not pm.lazy_installs_allowed()
