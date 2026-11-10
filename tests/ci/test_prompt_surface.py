"""The committed prompt-surface snapshot is exactly what the code renders.

``tests/fixtures/prompt_surface/`` holds the system prompt and ``tools[]`` array a stock install
sends on each surface (see ``scripts/ci/prompt_surface.py``). A code change that alters either
must carry the regenerated snapshot, so the change shows up in review as real prompt text and
real schemas, and the snapshot path needs a hermes-agent-core approval to merge.

Regenerate with ``scripts/run-in-hermes-env python scripts/ci/prompt_surface.py render``.
"""

from __future__ import annotations

import filecmp
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("prompt_surface", REPO / "scripts" / "ci" / "prompt_surface.py")
assert _spec is not None and _spec.loader is not None
prompt_surface = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prompt_surface)


def _tree_differences(left: Path, right: Path) -> list[str]:
    cmp = filecmp.dircmp(left, right)
    out = [f"only in committed: {n}" for n in cmp.left_only] + [f"not committed: {n}" for n in cmp.right_only]
    _, mismatch, errors = filecmp.cmpfiles(left, right, cmp.common_files, shallow=False)
    out += [f"stale: {n}" for n in mismatch + errors]
    for sub in cmp.common_dirs:
        out += [f"{sub}/{d}" for d in _tree_differences(left / sub, right / sub)]
    return out


def test_committed_prompt_surface_is_current(tmp_path):
    rendered = tmp_path / "prompt_surface"
    assert prompt_surface.render(rendered) == 0
    differences = _tree_differences(prompt_surface.SNAPSHOT, rendered)
    assert not differences, (
        "The system prompt or tool schemas changed. Regenerate the snapshot and commit it with the "
        "change (merging it needs a hermes-agent-core approval):\n"
        "    scripts/run-in-hermes-env python scripts/ci/prompt_surface.py render\n"
        + "\n".join(differences[:20])
    )


def test_render_ignores_enclosing_git_repo(tmp_path, monkeypatch):
    """An ambient temp dir inside a git repo must not leak a Workspace block.

    Worker scratch lives under ``~/.hermes``, which is a repo. ``render`` used to
    honor that ``TMPDIR``, so plain cli prompts gained a machine-specific
    ``- Root:`` line the committed snapshot (rendered outside a repo) rejects.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    # A bare ``git init`` is not a coding workspace (``_has_code_files``). The
    # worker leak is ``~/.hermes``, a repo that holds ``.py`` files.
    (outer / "probe.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", str(outer)], check=True, capture_output=True)
    monkeypatch.setattr(tempfile, "tempdir", str(outer))
    rendered = tmp_path / "rendered"
    assert prompt_surface.render(rendered) == 0
    text = (rendered / "prompts" / "cli.claude.txt").read_text(encoding="utf-8")
    assert "- Root:" not in text
    assert "Workspace (snapshot" not in text


def test_summary_names_the_surface_and_tool_that_changed(tmp_path):
    base = prompt_surface.SNAPSHOT
    head = tmp_path / "head"
    shutil.copytree(base, head)
    prompt = head / "prompts" / "cli.claude.txt"
    text = prompt.read_text(encoding="utf-8-sig")
    prompt.write_text(text.replace("# Finishing the job", "# Finish every job"), encoding="utf-8")
    tool = head / "tools" / "write_file.json"
    text = tool.read_text(encoding="utf-8-sig")
    tool.write_text(text.replace('"description": "', '"description": "Prefer patch. ', 1), encoding="utf-8")

    summary = prompt_surface.summarize(base, head)

    assert "| cli.claude |" in summary and "+# Finish every job" in summary
    assert "`write_file`: description +" in summary
    assert prompt_surface.summarize(base, base) == ""


def test_child_pythonpath_is_root_only_for_venv_interpreter(monkeypatch, tmp_path):
    """A venv interpreter keeps its own site-packages. Always appending the committed generation
    would shadow it (and can move snapshot bytes), so this must stay exactly ``ROOT``."""
    import pm.environments

    monkeypatch.setattr(prompt_surface.sys, "prefix", str(tmp_path / "venv-prefix"))
    monkeypatch.setattr(prompt_surface.sys, "base_prefix", str(tmp_path / "base-prefix"))
    version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    venv = tmp_path / "committed-venv"
    (venv / "lib" / f"python{version}" / "site-packages").mkdir(parents=True)
    monkeypatch.setattr(pm.environments, "committed_venv", lambda _root: venv)

    result = prompt_surface._child_pythonpath()

    assert result == str(prompt_surface.ROOT)
    assert os.pathsep not in result


def test_child_pythonpath_adds_committed_site_packages_for_store_interpreter(monkeypatch, tmp_path):
    """The store interpreter has no third-party packages. Restore the committed generation after
    ``ROOT``, and do not inherit the parent's PYTHONPATH."""
    import pm.environments

    store = str(tmp_path / "store")
    monkeypatch.setattr(prompt_surface.sys, "prefix", store)
    monkeypatch.setattr(prompt_surface.sys, "base_prefix", store)
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(pm.environments, "committed_venv", lambda _root: tmp_path / "venv")
    monkeypatch.setattr(pm.environments, "site_packages", lambda _venv: site)
    poison = str(tmp_path / "poison-pythonpath")
    monkeypatch.setenv("PYTHONPATH", poison)

    result = prompt_surface._child_pythonpath()

    assert result == os.pathsep.join((str(prompt_surface.ROOT), str(site)))
    assert poison not in result
