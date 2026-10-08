"""Read-only probe: pyproject base deps missing from, or drifted in, a venv.

Fork module (t_8ba9c4fc / intake t_4fa76c06): an editable reinstall with ``--no-deps`` left
``snowballstemmer`` out and tool_search silently off for weeks. ``hermes doctor`` uses these to
name the missing package. Probing never installs anything; PM / the install owner repairs.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# A probe that could not run: interpreter missing/unlaunchable, or the script exited non-zero.
DEP_PROBE_ERRORS = (OSError, subprocess.SubprocessError)

_MISSING_DEPS_SCRIPT = (
    "import importlib.metadata as md, sys\n"
    "missing=[]\n"
    "for name in sys.argv[1:]:\n"
    "    try: md.version(name)\n"
    "    except md.PackageNotFoundError: missing.append(name)\n"
    "print('\\n'.join(missing))\n")

_DEP_VERSIONS_SCRIPT = (
    "import importlib.metadata as md, sys\n"
    "for name in sys.argv[1:]:\n"
    "    try: print(name + '\\t' + md.version(name))\n"
    "    except md.PackageNotFoundError: pass\n")


def _naive_requirement(spec: str) -> tuple[str, str]:
    """``(name, head)`` of a ``name OP version ; marker`` spec without ``packaging``."""
    head = spec.split(";", 1)[0].strip()
    bare = head
    for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
        if op in bare:
            bare = bare.split(op, 1)[0]
            break
    return bare.strip().split("[", 1)[0].strip(), head


def _parse_requirements(raw_deps: list[str]) -> list[tuple[str, "object | None", str]]:
    """``(name, marker, head)`` per dep spec — ``packaging`` when importable, else a naive split."""
    parsed: list[tuple[str, "object | None", str]] = []
    try:
        from packaging.requirements import Requirement  # type: ignore
        for spec in raw_deps:
            try:
                req = Requirement(spec)
            except Exception:
                continue
            parsed.append((req.name, req.marker, spec.split(";", 1)[0].strip()))
    except Exception:
        for spec in raw_deps:
            name, head = _naive_requirement(spec)
            if name:
                parsed.append((name, None, head))
    return parsed


def _applicable_dependency_names(raw_deps: list[str]) -> list[str]:
    """Declared dep names whose ``;`` markers apply here (else ``ptyprocess ; sys_platform !=
    'win32'`` would false-positive on Windows). An unevaluable marker counts as applicable."""
    applicable: list[str] = []
    for name, marker, _ in _parse_requirements(raw_deps):
        try:
            if marker is None or marker.evaluate():  # type: ignore[union-attr]
                applicable.append(name)
        except Exception:
            applicable.append(name)
    return applicable


def _declared_base_dependencies() -> tuple[list[str], list[str]] | None:
    """``(raw specs, marker-applicable names)`` of pyproject's base deps; ``None`` when unreadable."""
    from hermes_cli.main_install_repair import _pyproject_project
    project = _pyproject_project("dep verification: failed to read pyproject.toml: %s")
    if project is None:
        return None
    raw_deps = project.get("dependencies", []) or []
    return raw_deps, _applicable_dependency_names(raw_deps)


def _run_dep_probe(venv_python: Path, script: str, names: list[str], env: dict[str, str] | None) -> list[str]:
    """Run *script* over *names* in *venv_python*; non-blank stdout lines. Raises a
    ``DEP_PROBE_ERRORS`` member (``CalledProcessError`` carries the probe's stderr) when it cannot answer."""
    result = subprocess.run(
        [str(venv_python), "-c", script, *names],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, [str(venv_python), "-c", "<dep probe>"], result.stdout, result.stderr)
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def missing_core_dependencies(venv_python: Path, *, env: dict[str, str] | None = None) -> list[str]:
    """Declared pyproject base deps (markers applied) with no installed distribution in
    *venv_python*'s environment; ``[]`` when there is no readable pyproject. A probe that cannot
    run raises a ``DEP_PROBE_ERRORS`` member: an unknown state is not reported as healthy."""
    declared = _declared_base_dependencies()
    if declared is None or not declared[1]:
        return []
    return _run_dep_probe(venv_python, _MISSING_DEPS_SCRIPT, declared[1], env)


def drifted_core_dependencies(
    venv_python: Path, *, env: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
    """``(name, installed version, declared spec)`` for installed base deps whose version falls
    outside the pyproject specifier. Missing deps are not listed (see :func:`missing_core_dependencies`).
    Raises like :func:`missing_core_dependencies` when the probe cannot run."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version
    declared = _declared_base_dependencies()
    if declared is None or not declared[1]:
        return []
    raw_deps, applicable = declared
    installed: dict[str, str] = {}
    for line in _run_dep_probe(venv_python, _DEP_VERSIONS_SCRIPT, applicable, env):
        name, _, version = line.partition("\t")
        if version:
            installed[name] = version
    drifted: list[tuple[str, str, str]] = []
    for spec in raw_deps:
        try:
            req = Requirement(spec)
        except InvalidRequirement as e:
            logger.warning("dep drift: unparseable pyproject requirement %r: %s", spec, e)
            continue
        version = installed.get(req.name)
        if version is None or not req.specifier:
            continue
        try:
            ok = req.specifier.contains(Version(version), prereleases=True)
        except InvalidVersion as e:
            logger.warning("dep drift: %s has unparseable installed version %r: %s", req.name, version, e)
            continue
        if not ok:
            drifted.append((req.name, version, spec.split(";", 1)[0].strip()))
    return drifted
