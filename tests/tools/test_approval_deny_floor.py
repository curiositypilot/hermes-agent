"""Regression coverage for operator deny globs as unconditional approval floors."""

import json

import pytest

import hermes_cli.config as hc
from tools import approval as mod
from tools import approval_context


_DENY_CASES = [
    pytest.param("terraform destroy*", "terraform destroy -auto-approve", id="terraform-destroy"),
    pytest.param(
        "terraform -chdir=* destroy*",
        "terraform -chdir=prod destroy -auto-approve",
        id="terraform-destroy-with-chdir",
    ),
    pytest.param(
        "aws cloudformation delete-stack*",
        "aws cloudformation delete-stack --stack-name prod",
        id="cloudformation-delete-stack",
    ),
    pytest.param(
        "aws cloudformation delete-stack --stack-name*",
        "aws cloudformation delete-stack --stack-name production",
        id="cloudformation-delete-stack-name",
    ),
    pytest.param("*drop database *", "psql -c 'DROP DATABASE production;'", id="drop-database"),
    pytest.param(
        "*drop schema * cascade*",
        "psql -c 'DROP SCHEMA production CASCADE;'",
        id="drop-schema-cascade",
    ),
    pytest.param("kubectl delete *", "kubectl delete pod production-api", id="kubectl-delete"),
    pytest.param(
        "kubectl delete namespace*",
        "kubectl delete namespace production",
        id="kubectl-delete-namespace",
    ),
    pytest.param("git push --force*", "git push --force origin main", id="git-force-push"),
    pytest.param("git push -f*", "git push -f origin main", id="git-short-force-push"),
    pytest.param("rm -rf /", "rm -rf /", id="rm-root"),
    pytest.param("rm -rf /*", "rm -rf / --no-preserve-root", id="rm-root-with-option"),
    pytest.param("mkfs*", "mkfs.ext4 /dev/sdb", id="mkfs-filesystem"),
    pytest.param("dd*of=/dev/*", "dd if=/dev/zero of=/dev/sda bs=4M", id="dd-to-device"),
]


@pytest.fixture
def deny_config_home(tmp_path, monkeypatch):
    """Use the real config reader against an isolated Hermes home."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    yield home
    hc._LOAD_CONFIG_CACHE.clear()


@pytest.mark.parametrize(("pattern", "command"), _DENY_CASES)
def test_deny_glob_blocks_with_approvals_off_and_docker_fast_path(
        monkeypatch, deny_config_home, pattern, command):
    """Every configured glob blocks despite approvals.mode=off, including docker's fast path."""
    (deny_config_home / "config.yaml").write_text(
        json.dumps({"approvals": {"mode": "off", "deny": [pattern]}}),
        encoding="utf-8",
    )
    hc._LOAD_CONFIG_CACHE.clear()
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)

    assert approval_context._get_approval_mode() == "off"
    assert mod._match_user_deny_rule(command) == pattern

    for guard in (mod.check_dangerous_command, mod.check_all_command_guards):
        for env_type in ("local", "docker"):
            result = guard(command, env_type)
            assert result["approved"] is False, (guard.__name__, env_type, command, result)
            if env_type == "docker":
                assert result.get("user_deny") is True, (guard.__name__, command, result)
