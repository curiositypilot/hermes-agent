"""``hermes doctor`` "Declared core dependencies": a pyproject base pin that never landed in the venv
(2026-10: ``snowballstemmer`` absent, tool_search silently off for weeks) is an issue naming the
package and the install command; a version outside its specifier is a warning only."""

import sys
from pathlib import Path

from hermes_cli import declared_deps, doctor_platform


def _run(monkeypatch, capsys, *, missing, drifted, should_fix=False):
    venv_python = Path("/fake/venv/bin/python")
    monkeypatch.setattr(doctor_platform, "_doctor_venv_python", lambda: venv_python)
    monkeypatch.setattr(doctor_platform, "_is_termux", lambda: False)
    monkeypatch.setattr(declared_deps, "missing_core_dependencies", lambda py, **kw: list(missing))
    monkeypatch.setattr(declared_deps, "drifted_core_dependencies", lambda py, **kw: list(drifted))
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


def test_fix_never_hides_a_missing_dep(monkeypatch, capsys):
    finding, _out, _ = _run(monkeypatch, capsys, missing=["snowballstemmer"], drifted=[], should_fix=True)
    assert finding.fixed == 0 and any("snowballstemmer" in i for i in finding.issues)


def test_probe_reports_missing_and_drifted_against_a_real_interpreter(monkeypatch):
    monkeypatch.setattr(declared_deps, "_declared_base_dependencies",
                        lambda: (["pytest>=0.0.1", "packaging<0.1", "no-such-dist-xyz>=1"],
                                 ["pytest", "packaging", "no-such-dist-xyz"]))
    assert declared_deps.missing_core_dependencies(Path(sys.executable)) == ["no-such-dist-xyz"]
    assert [d[0] for d in declared_deps.drifted_core_dependencies(Path(sys.executable))] == ["packaging"]
