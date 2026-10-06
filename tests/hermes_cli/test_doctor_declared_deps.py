"""``hermes doctor`` "Declared core dependencies": a pyproject base pin that never landed in the venv
(2026-10: ``snowballstemmer`` absent, tool_search silently off for weeks) is an issue naming the
package and the install command; a version outside its specifier is a warning only."""

from pathlib import Path

from hermes_cli import doctor_platform, main_install_repair


def _run(monkeypatch, capsys, *, missing, drifted, should_fix=False):
    venv_python = Path("/fake/venv/bin/python")
    monkeypatch.setattr(doctor_platform, "_doctor_venv_python", lambda: venv_python)
    monkeypatch.setattr(doctor_platform, "_is_termux", lambda: False)
    monkeypatch.setattr(main_install_repair, "missing_core_dependencies", lambda py, **kw: list(missing))
    monkeypatch.setattr(main_install_repair, "drifted_core_dependencies", lambda py, **kw: list(drifted))
    finding = doctor_platform._check_declared_dependencies(should_fix)
    return finding, capsys.readouterr().out, venv_python


def test_missing_declared_dep_is_an_issue_with_install_command(monkeypatch, capsys):
    finding, out, venv_python = _run(monkeypatch, capsys, missing=["snowballstemmer"], drifted=[])
    assert len(finding.issues) == 1
    issue = finding.issues[0]
    assert "snowballstemmer" in issue
    assert f"uv pip install --python {venv_python} -e ." in issue
    assert "--no-deps" not in issue
    assert "snowballstemmer" in out


def test_version_drift_is_a_warning_not_an_issue(monkeypatch, capsys):
    finding, out, _ = _run(monkeypatch, capsys, missing=[],
                           drifted=[("nemo-relay", "0.7.2", "nemo-relay>=0.8.3,<0.9")])
    assert finding.issues == [] and finding.manual_issues == []
    assert "nemo-relay 0.7.2" in out and ">=0.8.3" in out


def test_all_declared_deps_present_is_ok(monkeypatch, capsys):
    finding, out, _ = _run(monkeypatch, capsys, missing=[], drifted=[])
    assert finding.issues == [] and "all installed" in out


def test_fix_runs_the_base_install_and_counts_the_repair(monkeypatch, capsys):
    state = {"missing": ["snowballstemmer"]}
    installs = []
    monkeypatch.setattr(main_install_repair, "_default_venv_install_target", lambda: (["uv", "pip"], {"X": "1"}))

    def fake_verify(prefix, *, env=None, group="all"):
        installs.append((prefix, env))
        state["missing"] = []

    monkeypatch.setattr(main_install_repair, "_verify_core_dependencies_installed", fake_verify)
    monkeypatch.setattr(doctor_platform, "_doctor_venv_python", lambda: Path("/fake/py"))
    monkeypatch.setattr(main_install_repair, "missing_core_dependencies", lambda py, **kw: list(state["missing"]))
    monkeypatch.setattr(main_install_repair, "drifted_core_dependencies", lambda py, **kw: [])
    finding = doctor_platform._check_declared_dependencies(True)
    assert installs == [(["uv", "pip"], {"X": "1"})]
    assert finding.fixed == 1 and finding.issues == []
