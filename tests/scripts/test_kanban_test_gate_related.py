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

import kanban_test_gate_related as gate

TREE = [
    "tests/pm/test_install_x.py",
    "tests/scripts/install/test_install_y.py",
    "tests/e2e/core/upgrade/test_install_fresh.py",
    "tests/agent/test_run_agent.py",
    "tests/ci/test_prompt_surface.py",
    "tests/tools/test_foo.py",
    "tests/other/test_foo.py",
    "tests/e2e/x/test_lonely.py",
    "tests/docker/test_lonely_image.py",
    "tests/e2e/x/test_a.py",
    "tests/test___init__.py",
    "tests/test_conftest.py",
]


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _write(tmp_path, dict.fromkeys(TREE, ""))
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    return tmp_path


def test_mirror_dir_wins_over_global_stem_match(repo):
    assert gate.related_tests({"pm/install.py"}) == ["tests/pm/test_install_x.py"]


def test_root_level_source_walks_up_to_tests_root(repo):
    assert gate.related_tests({"run_agent.py"}) == [
        "tests/agent/test_run_agent.py",
        "tests/ci/test_prompt_surface.py",
    ]


def test_missing_mirror_level_falls_back_to_nearest_parent(repo):
    assert gate.related_tests({"tools/sub/foo.py"}) == [
        "tests/ci/test_prompt_surface.py",
        "tests/tools/test_foo.py",
    ]


def test_integration_e2e_docker_matches_are_dropped(repo):
    assert gate.related_tests({"lonely.py"}) == []
    assert gate.related_tests({"pkg/lonely.py"}) == []


def test_changed_test_file_is_selected_wherever_it_lives(repo):
    assert gate.related_tests({"tests/e2e/x/test_a.py"}) == ["tests/e2e/x/test_a.py"]


def test_init_and_conftest_stems_select_nothing(repo):
    assert gate.related_tests({"pkg/__init__.py", "tests/conftest.py", "conftest.py"}) == []


def test_cross_dir_tests_selects_mapped_and_mirror(repo, monkeypatch):
    extra = [
        "hermes_cli/venv_sync.py",
        "tests/hermes_cli/test_venv_sync.py",
        "tests/pm/test_source_update_launch.py",
        "tests/cron/test_restart_safe_worker.py",
    ]
    for rel in extra:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    selected = gate.related_tests({"hermes_cli/venv_sync.py"})
    assert selected == [
        "tests/ci/test_prompt_surface.py",
        "tests/cron/test_restart_safe_worker.py",
        "tests/hermes_cli/test_venv_sync.py",
        "tests/pm/test_source_update_launch.py",
    ]


def test_cross_dir_tests_ignores_missing_targets_and_skip_parts(repo, monkeypatch):
    monkeypatch.setattr(gate, "CROSS_DIR_TESTS", {
        "bar.py": (
            "tests/missing/test_missing.py",
            "tests/e2e/test_ignored_e2e.py",
            "tests/tools/test_foo.py",
        )
    })
    (repo / "tests/e2e/test_ignored_e2e.py").write_text("")
    selected = gate.related_tests({"bar.py"})
    assert selected == ["tests/tools/test_foo.py"]


@pytest.mark.parametrize("changed", [
    "agent/prompt_builder.py",
    "tools/terminal_tool.py",
    "plugins/memory/hindsight/__init__.py",
    "toolsets.py",
    "tests/fixtures/prompt_surface/surfaces.json",
    "run_agent.py",
    "model_tools.py",
    "scripts/ci/prompt_surface.py",
])
def test_prompt_surface_selected_for_surface_inputs(repo, changed):
    assert "tests/ci/test_prompt_surface.py" in gate.related_tests({changed})


@pytest.mark.parametrize("changed,expected", [
    ("docs/x.md", []),
    ("website/docs/a.md", []),
    ("tests/tools/test_foo.py", ["tests/tools/test_foo.py"]),
    ("pm/install.py", ["tests/pm/test_install_x.py"]),
    ("agent/notes.md", []),
])
def test_prompt_surface_not_selected_for_unrelated(repo, changed, expected):
    assert gate.related_tests({changed}) == expected


def test_gateway_run_does_not_select_prompt_surface(repo):
    selected = gate.related_tests({"gateway/run.py"})
    assert "tests/ci/test_prompt_surface.py" not in selected
    assert "tests/agent/test_run_agent.py" in selected



def test_importer_fallback_when_no_stem_match(repo):
    _write(repo, {
        "tests/pkg/test_a.py": "from pkg.session_x import Y\n",
        "tests/pkg/test_b.py": "from pkg import session_x as m\n",
        "tests/pkg/test_p.py": "from pkg import (\n    other,\n    session_x,\n)\n",
        "tests/other/test_d.py": 'def t(monkeypatch):\n    monkeypatch.setattr("pkg.session_x.attr", 1)\n',
        "tests/pkg/test_c.py": "from pkg.session_x_log import Z\n",
        "tests/pkg/test_e.py": "import mypkg.session_x\n",
        "tests/pkg/test_f.py": "from pkg import session_xy\n",
        "tests/e2e/x/test_g.py": "from pkg.session_x import Y\n",
    })
    notes: list[str] = []
    assert gate.related_tests({"pkg/session_x.py"}, notes) == [
        "tests/other/test_d.py",
        "tests/pkg/test_a.py",
        "tests/pkg/test_b.py",
        "tests/pkg/test_p.py",
    ]
    assert notes == ["importer fallback: pkg/session_x.py -> 4 file(s)"]


def test_importer_fallback_skipped_when_stem_matches(repo):
    _write(repo, {"tests/tools/test_bar.py": "from tools.foo import x\n"})
    assert gate.related_tests({"tools/foo.py"}) == [
        "tests/ci/test_prompt_surface.py",
        "tests/tools/test_foo.py",
    ]


def test_root_module_importer_needs_import_form(repo):
    _write(repo, {
        "tests/test_h.py": "import lonely\n",
        "tests/test_i.py": 'from unittest.mock import patch\n\n@patch("lonely.f")\ndef t(f):\n    pass\n',
        "tests/test_j.py": 'x = "lonely"\n# lonely\n',
    })
    assert gate.related_tests({"lonely.py"}) == ["tests/test_h.py", "tests/test_i.py"]


def test_incident_t_edada6ea_replay():
    """Real repo: the module's tests are test_managed_worker_*.py, which no stem rule selects."""
    selected = gate.related_tests({
        "gateway/session_managed_worker.py",
        "tests/gateway/test_managed_worker_sandbox_safe_path.py",
    })
    assert "tests/gateway/test_managed_worker_sandbox.py" in selected
