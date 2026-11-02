"""Replay of intake t_7a1ad159: profile churn must not change the dependency stamp.

On 2026-11-02 four environment generations were built in twelve minutes with unchanged core
inputs. Cloning a profile copied the default home's ``plugins/hindsight`` byte for byte and
kept ``memory.provider: hindsight``; the stamp keyed members by path, so creating the clone,
clearing its provider and deleting it each forced a full rebuild under the checkout lock.
"""
import shutil

from pm.workspace import enabled_member_dirs, members_stamp

_PYPROJECT = (
    '[project]\nname = "hermes-plugin-hindsight"\nversion = "1.0.1"\nrequires-python = ">=3.11"\n'
    'dependencies = [\n    "hindsight-client>=0.10.1,<1",\n    "hindsight-embed>=0.10.1,<1",\n]\n'
    '[tool.uv]\npackage = false\n[dependency-groups]\ndev = ["pytest>=8.0"]\n'
)


def _stamp():
    return members_stamp(enabled_member_dirs())


def _clone(home, name):
    """What ``hermes profile create --clone`` leaves: config, plugins and an identity marker."""
    profile = home / "profiles" / name
    profile.mkdir(parents=True)
    shutil.copytree(home / "plugins", profile / "plugins")
    (profile / "plugins" / "hindsight" / "__pycache__").mkdir()
    (profile / "plugins" / "hindsight" / "__pycache__" / "__init__.cpython-314.pyc").write_bytes(name.encode())
    (profile / "profile.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (profile / "config.yaml").write_text("memory:\n  provider: hindsight\n", encoding="utf-8")
    return profile


def test_profile_create_edit_delete_keeps_the_stamp(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "hindsight"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (plugin / "plugin.yaml").write_text('name: hindsight\nversion: 1.0.1\npip_dependencies:\n  - "hindsight-client>=0.10.1"\n',
                                        encoding="utf-8")
    (plugin / "__init__.py").write_text("PROVIDER = 'hindsight'\n", encoding="utf-8")
    (home / "config.yaml").write_text("memory:\n  provider: hindsight\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    baseline = _stamp()

    # 17:40:26 profile gtm created with provider hindsight: its copy joins the selection.
    gtm = _clone(home, "gtm")
    assert gtm / "plugins" / "hindsight" in enabled_member_dirs()
    assert _stamp() == baseline

    # 17:42:46 gtm's provider cleared.
    (gtm / "config.yaml").write_text("memory:\n  provider: ''\n", encoding="utf-8")
    assert gtm / "plugins" / "hindsight" not in enabled_member_dirs()
    assert _stamp() == baseline

    # 17:51 test-fork created, 17:51:33 deleted.
    fork = _clone(home, "test-fork")
    assert fork / "plugins" / "hindsight" in enabled_member_dirs()
    assert _stamp() == baseline
    shutil.rmtree(fork)
    assert _stamp() == baseline

    # gtm removed too; and a genuinely different copy still changes the stamp.
    shutil.rmtree(gtm)
    assert _stamp() == baseline
    other = _clone(home, "other")
    (other / "plugins" / "hindsight" / "__init__.py").write_text("PROVIDER = 'edited'\n", encoding="utf-8")
    assert _stamp() != baseline
