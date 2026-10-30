"""A key removed from ``~/.hermes/.env`` leaves ``os.environ`` on the next reload.

Regression for t_8d6c7d5a: the gateway reloads ``.env`` per turn and per cron fire, but a reload only
republished the keys the file still defined. An ``ANTHROPIC_API_KEY`` deleted from ``.env`` stayed in
the long-lived gateway's environ until restart and reached every spawned Kanban worker, where it beat
the OAuth pool and failed every request with a billing 400.

The guard is the same as the PATH-growth fix: only a value this process published from that file and
nobody changed since is taken back; shell exports and secret-source values stay.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.env_loader import load_hermes_dotenv


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_env(home: Path, text: str) -> None:
    (home / ".env").write_text(text, encoding="utf-8")


def _load(home: Path) -> None:
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)


def test_key_removed_from_dotenv_is_unset_on_reload(home, monkeypatch):
    monkeypatch.delenv("RMKEY_A_API_KEY", raising=False)
    _write_env(home, "RMKEY_A_API_KEY=x\nRMKEY_A_KEEP=1\n")
    _load(home)
    assert os.environ["RMKEY_A_API_KEY"] == "x"

    _write_env(home, "RMKEY_A_KEEP=1\n")
    _load(home)

    assert "RMKEY_A_API_KEY" not in os.environ
    assert os.environ["RMKEY_A_KEEP"] == "1"


def test_shell_export_absent_from_dotenv_survives_reload(home, monkeypatch):
    monkeypatch.setenv("RMKEY_B_API_KEY", "from-shell")
    _write_env(home, "RMKEY_B_OTHER=1\n")
    _load(home)
    _write_env(home, "RMKEY_B_OTHER=2\n")
    _load(home)

    assert os.environ["RMKEY_B_API_KEY"] == "from-shell"


def test_shell_value_shadowed_by_dotenv_comes_back_when_the_line_goes(home, monkeypatch):
    monkeypatch.setenv("RMKEY_C_API_KEY", "from-shell")
    _write_env(home, "RMKEY_C_API_KEY=from-dotenv\n")
    _load(home)
    assert os.environ["RMKEY_C_API_KEY"] == "from-dotenv"

    _write_env(home, "\n")
    _load(home)

    assert os.environ["RMKEY_C_API_KEY"] == "from-shell"


def test_key_still_defined_keeps_its_new_value(home, monkeypatch):
    monkeypatch.delenv("RMKEY_D_API_KEY", raising=False)
    _write_env(home, "RMKEY_D_API_KEY=old\n")
    _load(home)
    _write_env(home, "RMKEY_D_API_KEY=new\n")
    _load(home)

    assert os.environ["RMKEY_D_API_KEY"] == "new"


def test_value_changed_after_publish_survives_removal(home, monkeypatch):
    """A secret source (or anything else) replaced our published value: it is theirs now."""
    monkeypatch.delenv("RMKEY_E_API_KEY", raising=False)
    _write_env(home, "RMKEY_E_API_KEY=from-dotenv\n")
    _load(home)
    os.environ["RMKEY_E_API_KEY"] = "from-secret-source"

    _write_env(home, "\n")
    _load(home)

    assert os.environ["RMKEY_E_API_KEY"] == "from-secret-source"
    monkeypatch.delenv("RMKEY_E_API_KEY", raising=False)


def test_launch_sweep_drops_removed_key_without_a_full_reload(home, monkeypatch):
    """The Kanban dispatcher's entry point: no reload ran between the edit and the spawn."""
    from hermes_cli.env_loader import unpublish_removed_launch_dotenv_keys

    monkeypatch.delenv("RMKEY_F_API_KEY", raising=False)
    _write_env(home, "RMKEY_F_API_KEY=x\n")
    load_hermes_dotenv(load_external_secrets=False)
    assert os.environ["RMKEY_F_API_KEY"] == "x"

    _write_env(home, "\n")
    unpublish_removed_launch_dotenv_keys()

    assert "RMKEY_F_API_KEY" not in os.environ


def test_other_files_keys_are_not_swept_by_the_user_env(home, tmp_path, monkeypatch):
    """A key the project .env published is not the user .env's to take back."""
    monkeypatch.delenv("RMKEY_G_SETTING", raising=False)
    project_env = tmp_path / "project.env"
    project_env.write_text("RMKEY_G_SETTING=from-project\n", encoding="utf-8")
    _write_env(home, "RMKEY_G_OTHER=1\n")
    load_hermes_dotenv(hermes_home=home, project_env=project_env, load_external_secrets=False)
    assert os.environ["RMKEY_G_SETTING"] == "from-project"

    _write_env(home, "RMKEY_G_OTHER=2\n")
    _load(home)

    assert os.environ["RMKEY_G_SETTING"] == "from-project"
    monkeypatch.delenv("RMKEY_G_SETTING", raising=False)
