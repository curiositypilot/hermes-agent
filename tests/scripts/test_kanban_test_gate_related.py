"""Behaviour of the kanban test gate's related-test selection (scripts/kanban_test_gate_related.py).

Regression for intake t_5401508c: a change to ``pm/install.py`` selected every
``tests/**/test_install*.py`` in the repo, including an e2e suite that hit the per-file timeout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import kanban_test_gate_related as gate  # noqa: E402

TREE = [
    "tests/pm/test_install_x.py",
    "tests/scripts/install/test_install_y.py",
    "tests/e2e/core/upgrade/test_install_fresh.py",
    "tests/agent/test_run_agent.py",
    "tests/tools/test_foo.py",
    "tests/other/test_foo.py",
    "tests/e2e/x/test_lonely.py",
    "tests/docker/test_lonely_image.py",
    "tests/e2e/x/test_a.py",
    "tests/test___init__.py",
    "tests/test_conftest.py",
]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for rel in TREE:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    return tmp_path


def test_mirror_dir_wins_over_global_stem_match(repo):
    assert gate.related_tests({"pm/install.py"}) == ["tests/pm/test_install_x.py"]


def test_root_level_source_walks_up_to_tests_root(repo):
    assert gate.related_tests({"run_agent.py"}) == ["tests/agent/test_run_agent.py"]


def test_missing_mirror_level_falls_back_to_nearest_parent(repo):
    assert gate.related_tests({"tools/sub/foo.py"}) == ["tests/tools/test_foo.py"]


def test_integration_e2e_docker_matches_are_dropped(repo):
    assert gate.related_tests({"lonely.py"}) == []
    assert gate.related_tests({"pkg/lonely.py"}) == []


def test_changed_test_file_is_selected_wherever_it_lives(repo):
    assert gate.related_tests({"tests/e2e/x/test_a.py"}) == ["tests/e2e/x/test_a.py"]


def test_init_and_conftest_stems_select_nothing(repo):
    assert gate.related_tests({"pkg/__init__.py", "tests/conftest.py", "conftest.py"}) == []
