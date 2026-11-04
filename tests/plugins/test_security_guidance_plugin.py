"""Tests for the security-guidance plugin.

Covers ``plugins/security-guidance/``:

  * ``patterns.py`` data integrity — every rule has a name, reminder, and at
    least one trigger; names are unique.
  * ``_scan_content`` — true positives (pickle.load, yaml.load, eval,
    dangerouslySetInnerHTML, GitHub Actions workflow), true negatives
    (.md skips Python rules, ``model.eval()`` doesn't trip eval),
    path-only rules (``path_check``), content-only rules
    (``path_filter``).
  * Hooks — ``transform_tool_result`` appends a warning block in warn
    mode and stays out of error results; ``pre_tool_call`` blocks
    writes when ``SECURITY_GUIDANCE_BLOCK=1`` and stays silent
    otherwise. ``skill_manage`` is scanned per operation, including the
    legacy flat shape.
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SECURITY_GUIDANCE_BLOCK", raising=False)
    monkeypatch.delenv("SECURITY_GUIDANCE_DISABLE", raising=False)
    yield hermes_home

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _load_patterns():
    """Import patterns.py in isolation (no plugin glue)."""
    pat_path = _repo_root() / "plugins" / "security-guidance" / "patterns.py"
    spec = importlib.util.spec_from_file_location(
        "security_guidance_patterns_under_test", pat_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_plugin_init():
    """Import the plugin __init__.py with patterns.py as a sibling."""
    plugin_dir = _repo_root() / "plugins" / "security-guidance"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.security_guidance",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.security_guidance"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.security_guidance"] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# patterns.py data integrity
# ---------------------------------------------------------------------------

class TestPatternsData:

    def test_every_rule_has_required_fields(self):
        p = _load_patterns()
        for rule in p.SECURITY_PATTERNS:
            assert "ruleName" in rule
            assert rule.get("reminder")
            # At least one of substrings/regex/scan_literals/path_check must be present —
            # otherwise the rule could never fire.
            assert any(k in rule for k in ("substrings", "regex", "scan_literals", "path_check")), rule

    def test_rule_names_are_unique(self):
        p = _load_patterns()
        names = [r["ruleName"] for r in p.SECURITY_PATTERNS]
        assert len(names) == len(set(names))

# ---------------------------------------------------------------------------
# _scan_content
# ---------------------------------------------------------------------------

class TestScanContent:
    def test_pickle_load_in_py_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.py", "import pickle\nx = pickle.load(open('p.pkl', 'rb'))\n"
        )
        names = [n for n, _ in findings]
        assert "pickle_deserialization" in names

    def test_method_call_eval_does_not_trip(self):
        """model.eval() / redis.eval() / spec.eval() must not match eval_injection."""
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "model.eval()\nout = model(x)\n")
        assert "eval_injection" not in [n for n, _ in findings]

    def test_bare_eval_in_py_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "result = eval(user_input)\n")
        assert "eval_injection" in [n for n, _ in findings]

    def test_xml_unsafe_parse_spares_defusedxml(self):
        """The rule's own recommended fix (defusedxml) must not warn; stdlib calls still do."""
        mod = _load_plugin_init()
        safe = (
            "root = defusedxml.ElementTree.fromstring(body)\n"
            "doc = defusedxml.minidom.parseString(body)\n"
        )
        assert "xml_unsafe_parse" not in [n for n, _ in mod._scan_content("/x/app.py", safe)]
        for unsafe in (
            "root = xml.etree.ElementTree.fromstring(body)\n",
            "root = ET.parse(path)\n",
            "doc = xml.dom.minidom.parseString(body)\n",
        ):
            assert "xml_unsafe_parse" in [n for n, _ in mod._scan_content("/x/app.py", unsafe)]

    def test_subprocess_shell_true_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.py", "subprocess.run('ls ' + path, shell=True)\n"
        )
        assert "python_subprocess_shell" in [n for n, _ in findings]

    def test_dangerously_set_inner_html_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.tsx", "<div dangerouslySetInnerHTML={{__html: x}} />"
        )
        assert "react_dangerously_set_html" in [n for n, _ in findings]

    def test_github_workflow_path_check_fires_on_path_alone(self):
        """github_actions_workflow has no regex/substring — fires on path."""
        mod = _load_plugin_init()
        findings = mod._scan_content(
            ".github/workflows/test.yml", "name: CI\non: pull_request"
        )
        assert "github_actions_workflow" in [n for n, _ in findings]

    def test_non_workflow_path_doesnt_trip_workflow_rule(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "name: CI")
        assert "github_actions_workflow" not in [n for n, _ in findings]

    def test_empty_content_returns_no_findings(self):
        mod = _load_plugin_init()
        assert mod._scan_content("/tmp/foo.py", "") == []

    def test_huge_content_skipped(self):
        mod = _load_plugin_init()
        # 1 MB of content with a dangerous pattern at the end — scanner caps
        # out at _MAX_SCAN_BYTES (256 KB), so this should return [].
        big = "x" * (1024 * 1024) + "\npickle.load(open('p.pkl', 'rb'))\n"
        assert mod._scan_content("/tmp/foo.py", big) == []

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class TestTransformToolResultHook:
    def test_warns_on_write_file_with_dangerous_content(self):
        mod = _load_plugin_init()
        args = {
            "path": "/tmp/foo.py",
            "content": "import pickle\nx = pickle.loads(b)\n",
        }
        result = mod._on_transform_tool_result(
            tool_name="write_file",
            args=args,
            result='{"success": true, "bytes_written": 30}',
        )
        assert isinstance(result, str)
        assert "Security guidance" in result
        assert "pickle_deserialization" in result
        # The original JSON should still be there at the start of the string.
        assert result.startswith('{"success": true')

    def test_no_warn_on_clean_content(self):
        mod = _load_plugin_init()
        args = {"path": "/tmp/foo.py", "content": "import json\nx = json.loads(b)\n"}
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"success": true}'
            )
            is None
        )

    def test_patch_tool_new_string_scanned(self):
        mod = _load_plugin_init()
        args = {
            "path": "/tmp/foo.py",
            "old_string": "x = 1",
            "new_string": "x = eval(user_input)",
        }
        result = mod._on_transform_tool_result(
            tool_name="patch", args=args, result='{"success": true}'
        )
        assert isinstance(result, str)
        assert "eval_injection" in result

    def test_untargeted_tool_skipped(self):
        mod = _load_plugin_init()
        # The plugin only scans write_file/patch/skill_manage. terminal output
        # should pass through untouched.
        args = {"command": "echo pickle.load"}
        assert (
            mod._on_transform_tool_result(
                tool_name="terminal", args=args, result='{"output": "pickle.load"}'
            )
            is None
        )

    def test_disable_kill_switch(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_DISABLE", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"ok": true}'
            )
            is None
        )

    def test_block_mode_makes_transform_hook_quiet(self, monkeypatch):
        """In block mode, pre_tool_call handles the warning; the transform
        hook stays silent so we don't double-emit."""
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"ok": true}'
            )
            is None
        )

class TestPreToolCallHook:

    def test_blocks_in_block_mode_on_dangerous_pattern(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        out = mod._on_pre_tool_call(tool_name="write_file", args=args)
        assert isinstance(out, dict)
        assert out["action"] == "block"
        assert "pickle_deserialization" in out["message"]
        assert "SECURITY_GUIDANCE_BLOCK" in out["message"]  # tells user how to disable

# ---------------------------------------------------------------------------
# Bundled-plugin discovery
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    def test_manifest_declares_registered_hooks(self):
        """Manifest metadata must use the field consumed by plugin discovery."""
        import hermes_yaml as yaml

        plugin_dir = _repo_root() / "plugins" / "security-guidance"
        manifest = yaml.safe_load(
            (plugin_dir / "plugin.yaml").read_text(encoding="utf-8")
        )
        mod = _load_plugin_init()
        registered = []

        class HookContext:
            def register_hook(self, name, _callback):
                registered.append(name)

        mod.register(HookContext())
        assert set(manifest["provides_hooks"]) == set(registered)
        assert "hooks" not in manifest


# ---------------------------------------------------------------------------
# skill_manage — operations[] batch (and the legacy flat shape)
# ---------------------------------------------------------------------------

_PICKLE_PY = "x = pickle.loads(b)\n"
_EVAL_PY = "x = eval(user_input)"
_CLEAN_PY = "x = json.loads(b)\n"


def _write_file_op(content: str, file_path: str = "scripts/a.py") -> dict:
    return {
        "action": "write_file",
        "name": "x",
        "file_path": file_path,
        "file_content": content,
    }


class TestSkillManageScan:
    def test_operations_write_file_warns(self):
        mod = _load_plugin_init()
        result = mod._on_transform_tool_result(
            tool_name="skill_manage",
            args={"operations": [_write_file_op(_PICKLE_PY)]},
            result='{"success": true}',
        )
        assert isinstance(result, str)
        assert "pickle_deserialization" in result

    def test_operations_patch_new_string_warns(self):
        mod = _load_plugin_init()
        result = mod._on_transform_tool_result(
            tool_name="skill_manage",
            args={
                "operations": [
                    {
                        "action": "patch",
                        "name": "x",
                        "file_path": "scripts/a.py",
                        "new_string": _EVAL_PY,
                    }
                ]
            },
            result='{"success": true}',
        )
        assert isinstance(result, str)
        assert "eval_injection" in result

    def test_skill_md_targets_recorded(self, monkeypatch):
        """create/patch with no file_path target SKILL.md. Path filters must not matter."""
        mod = _load_plugin_init()
        recorded: list[tuple[str, str]] = []

        def recorder(path, content):
            recorded.append((path, content))
            return []

        monkeypatch.setattr(mod, "_scan_content", recorder)
        create_body = "---\nname: x\n---\nbody\n"
        rewrite = "full rewrite\n"
        patched = "one patched line\n"
        mod._scan_args(
            "skill_manage",
            {"operations": [{"action": "create", "name": "x", "content": create_body}]},
        )
        mod._scan_args(
            "skill_manage",
            {"operations": [{"action": "patch", "name": "x", "content": rewrite}]},
        )
        mod._scan_args(
            "skill_manage",
            {"operations": [{"action": "patch", "name": "x", "new_string": patched}]},
        )
        assert recorded == [
            ("SKILL.md", create_body),
            ("SKILL.md", rewrite),
            ("SKILL.md", patched),
        ]

    def test_legacy_flat_write_file_still_warns(self):
        mod = _load_plugin_init()
        result = mod._on_transform_tool_result(
            tool_name="skill_manage",
            args={
                "action": "write_file",
                "file_path": "scripts/a.py",
                "file_content": _PICKLE_PY,
            },
            result='{"success": true}',
        )
        assert isinstance(result, str)
        assert "pickle_deserialization" in result

    def test_legacy_flat_create_recorded(self, monkeypatch):
        mod = _load_plugin_init()
        recorded: list[tuple[str, str]] = []

        def recorder(path, content):
            recorded.append((path, content))
            return []

        monkeypatch.setattr(mod, "_scan_content", recorder)
        body = "---\nname: x\n---\nflat create\n"
        mod._scan_args(
            "skill_manage",
            {"action": "create", "name": "x", "content": body},
        )
        assert recorded == [("SKILL.md", body)]

    def test_stringified_operations_warns(self):
        mod = _load_plugin_init()
        result = mod._on_transform_tool_result(
            tool_name="skill_manage",
            args={"operations": json.dumps([_write_file_op(_PICKLE_PY)])},
            result='{"success": true}',
        )
        assert isinstance(result, str)
        assert "pickle_deserialization" in result

    def test_bad_operations_json_returns_empty(self):
        mod = _load_plugin_init()
        assert mod._scan_args("skill_manage", {"operations": "not json"}) == []

    def test_dedupes_rule_across_ops(self):
        mod = _load_plugin_init()
        findings = mod._scan_args(
            "skill_manage",
            {
                "operations": [
                    _write_file_op(_PICKLE_PY, "scripts/a.py"),
                    _write_file_op(_PICKLE_PY, "scripts/b.py"),
                ]
            },
        )
        names = [name for name, _ in findings]
        assert names.count("pickle_deserialization") == 1

    def test_non_content_and_clean_ops_return_empty(self):
        mod = _load_plugin_init()
        cases = [
            {"operations": [{"action": "delete", "name": "x"}]},
            {"operations": [{"action": "remove_file", "name": "x", "file_path": "scripts/a.py"}]},
            {"operations": [_write_file_op(_CLEAN_PY)]},
            {"operations": []},
            {"operations": [None, 3]},
        ]
        for args in cases:
            assert mod._scan_args("skill_manage", args) == [], args

    def test_block_mode_refuses_batch(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        out = mod._on_pre_tool_call(
            tool_name="skill_manage",
            args={"operations": [_write_file_op(_PICKLE_PY)]},
        )
        assert isinstance(out, dict)
        assert out["action"] == "block"
        assert "pickle_deserialization" in out["message"]



# ---------------------------------------------------------------------------
# Doc-default path filter and Python literal masking (intake t_d1714856)
# ---------------------------------------------------------------------------

_PROSE_RISKS = "yaml.load(f) and verify=False and subprocess.run(x, shell=True)\n"


def _names(mod, path: str, content: str) -> list[str]:
    return [name for name, _ in mod._scan_content(path, content)]


class TestDocDefaultFilter:
    def test_spec_md_quoting_xml_parse_is_quiet(self):
        mod = _load_plugin_init()
        assert "xml_unsafe_parse" not in _names(mod, "/x/spec.md", "quotes `root = ET.fromstring(body)` in prose")

    @pytest.mark.parametrize("path", ["/x/notes.md", "/x/cfg.yaml", "/x/x.json", "/x/readme.txt"])
    def test_doc_files_skip_language_rules(self, path):
        mod = _load_plugin_init()
        assert _names(mod, path, _PROSE_RISKS) == []

    @pytest.mark.parametrize(
        ("path", "content", "rule"),
        [
            (".github/workflows/ci.yml", "name: CI\n", "github_actions_workflow"),
            ("/x/app.py", "root = ET.fromstring(body)\n", "xml_unsafe_parse"),
            ("/x/app.go", 'cmd := exec.Command("sh", "-c", c)\n', "go_exec_shell_injection"),
            ("/x/x.ts", "el.innerHTML = v\n", "innerHTML_xss"),
        ],
    )
    def test_code_files_still_match(self, path, content, rule):
        mod = _load_plugin_init()
        assert rule in _names(mod, path, content)


class TestPythonLiteralMasking:
    @pytest.mark.parametrize(
        ("content", "rule"),
        [
            # t_8c3667d6's fixture: a patch payload quoted inside a test dict.
            ('p = {"new_string": "+root = ET.fromstring(body)"}\n', "xml_unsafe_parse"),
            ('def f():\n    """Never call yaml.load(f)."""\n', "unsafe_yaml_load"),
            ("x = 1  # pickle.loads is unsafe\n", "pickle_deserialization"),
            ('s = f"call ET.fromstring({x})"\n', "xml_unsafe_parse"),
        ],
    )
    def test_literals_and_comments_do_not_match(self, content, rule):
        mod = _load_plugin_init()
        assert rule not in _names(mod, "/x/tests/test_x.py", content)

    @pytest.mark.parametrize(
        ("content", "rule"),
        [
            ('x = f"{ET.fromstring(b)}"\n', "xml_unsafe_parse"),
            ('a = np.load("f.npy", allow_pickle=True)\n', "pickle_wrapper_load"),
            ('cfg = yaml.load(open("c"))\n', "unsafe_yaml_load"),
            ('subprocess.run("ls " + p, shell=True)\n', "python_subprocess_shell"),
            ('s = "a"; ET.parse(f)\n', "xml_unsafe_parse"),
        ],
    )
    def test_code_around_literals_still_matches(self, content, rule):
        mod = _load_plugin_init()
        assert rule in _names(mod, "/x/app.py", content)

    def test_masking_applies_to_pyi_not_ipynb(self):
        mod = _load_plugin_init()
        fixture = 'x = "ET.fromstring(b)"\n'
        assert "xml_unsafe_parse" not in _names(mod, "/x/stub.pyi", fixture)
        assert "xml_unsafe_parse" in _names(mod, "/x/nb.ipynb", fixture)

    def test_open_call_fragment_still_matches(self):
        mod = _load_plugin_init()
        assert "xml_unsafe_parse" in _names(mod, "/x/app.py", "foo(ET.fromstring(\n")

    def test_unterminated_triple_quote_is_masked_to_end(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("/x/app.py", 'a = """open\nET.parse(f)\n')
        assert isinstance(findings, list)
        assert "xml_unsafe_parse" not in [n for n, _ in findings]

    def test_unterminated_single_quote_masks_only_its_line(self):
        mod = _load_plugin_init()
        assert "xml_unsafe_parse" not in _names(mod, "/x/app.py", 'x = "ET.parse(f)\n')
        assert "xml_unsafe_parse" in _names(mod, "/x/app.py", 'x = "abc\nET.parse(f)\n')

    def test_bad_dedent_fragment_does_not_raise(self):
        mod = _load_plugin_init()
        assert "xml_unsafe_parse" in _names(mod, "/x/app.py", "def f():\n  x = 1\n y = 2\nET.parse(q)\n")

    @pytest.mark.parametrize(
        ("path", "content", "rule"),
        [
            ("/x/page.py", "HTML = '<script src=\"https://cdn.x/a.js\"></script>'\n", "script_src_without_sri"),
            ("/x/c.py", "c = AES.new(k, mode='aes-256-ecb')\n", "aes_ecb_mode"),
            ("/x/c.py", 'NODE_TLS_REJECT_UNAUTHORIZED = "0"\n', "tls_verification_disabled"),
        ],
    )
    def test_scan_literals_arms_read_raw_content(self, path, content, rule):
        mod = _load_plugin_init()
        assert rule in _names(mod, path, content)

    def test_masked_text_keeps_line_structure(self):
        mod = _load_plugin_init()
        src = 'a = """x\ny"""  # c\r\nb = 1\n'
        masked = mod._mask_python_literals(src)
        assert len(masked) == len(src)
        assert masked.count("\n") == src.count("\n") and masked.count("\r") == src.count("\r")
        assert masked.endswith("b = 1\n") and "x" not in masked and "#" not in masked

    # Patch fragments that start deeper than they end (t_23823b99): tokenize raises IndentationError
    # at the first unmatched dedent; literals after it must still be masked.
    def test_deeper_first_line_fragment_is_masked(self):
        mod = _load_plugin_init()
        src = '        return x\n\n    def test_y(self):\n        s = "ET.fromstring(body)"\n'
        assert "xml_unsafe_parse" not in _names(mod, "/x/test_y.py", src)
        masked = mod._mask_python_literals(src)
        assert "ET.fromstring" not in masked
        assert len(masked) == len(src) and masked.count("\n") == src.count("\n")

    def test_fragment_triple_quote_after_bad_dedent_is_masked(self):
        mod = _load_plugin_init()
        src = '        return x\n    def f():\n        s = """a\n   ET.parse(q)\n"""  # c\n        ET.parse(real)\n'
        masked = mod._mask_python_literals(src)
        assert "ET.parse(q)" not in masked
        assert masked.find("ET.parse(real)") == src.find("ET.parse(real)")
        assert "xml_unsafe_parse" in _names(mod, "/x/app.py", src)

    def test_code_after_bad_dedent_still_matches(self):
        mod = _load_plugin_init()
        assert "xml_unsafe_parse" in _names(mod, "/x/app.py", "        x = 1\n    ET.parse(f)\n")

    def test_mixed_tabs_spaces_fragment_is_masked(self):
        mod = _load_plugin_init()
        assert "xml_unsafe_parse" not in _names(mod, "/x/app.py", '\tif a:\n  b = "ET.parse(f)"\n')

    def test_unterminated_single_quote_after_bad_dedent(self):
        mod = _load_plugin_init()
        masked = mod._mask_python_literals('        x = 1\n    y = "ET.parse(f)\n    ET.parse(g)\n')
        assert "ET.parse(f)" not in masked and "ET.parse(g)" in masked


class TestV4APatchSections:
    def test_pathless_patch_uses_section_path(self):
        mod = _load_plugin_init()
        args = {"mode": "patch", "patch": "*** Begin Patch\n*** Update File: docs/spec.md\n+root = ET.fromstring(body)\n*** End Patch\n"}
        assert mod._scan_args("patch", args) == []

    def test_code_section_in_mixed_patch_still_warns(self):
        mod = _load_plugin_init()
        args = {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n*** Update File: docs/spec.md\n+ET.fromstring(prose)\n"
                "*** Add File: app/x.py\n+import x\n+root = ET.fromstring(body)\n"
                "*** Update File: app/y.py\n@@ def f @@\n+root = ET.parse(f)\n*** End Patch\n"
            ),
        }
        assert [name for name, _ in mod._scan_args("patch", args)] == ["xml_unsafe_parse"]

    def test_removed_lines_and_quoted_payloads_do_not_warn(self):
        mod = _load_plugin_init()
        args = {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n*** Update File: app/x.py\n@@ def f @@\n"
                "-root = ET.fromstring(body)\n+root = safe_fromstring(body)\n"
                '+FIXTURE = "+root = ET.fromstring(body)"\n*** End Patch\n'
            ),
        }
        assert mod._scan_args("patch", args) == []

    def test_headerless_patch_text_scanned_at_arg_path(self):
        mod = _load_plugin_init()
        args = {"path": "/x/app.py", "patch": "root = ET.fromstring(body)\n"}
        assert "xml_unsafe_parse" in [name for name, _ in mod._scan_args("patch", args)]
