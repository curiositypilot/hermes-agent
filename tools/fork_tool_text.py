"""Fork-owned shorter wording for core tool descriptions (intake t_13709b89).

The texts live in upstream-owned ``tools/*.py`` files; editing them there would conflict on every
upstream sync. Instead ``model_tools._compute_tool_definitions`` passes the built definitions through
:func:`apply_fork_tool_text` right after the dynamic-schema rewriters, so this file is the only
place the fork carries the shorter wording.

Each edit is a substring replacement inside one description field. It applies only when ``old``
occurs exactly once in that field; otherwise the field keeps upstream's text and the skip is logged
once per process (tests/tools/test_fork_tool_text.py fails on the next sync instead). Substring
edits rather than whole-text replacement because read_file, execute_code and delegate_task put
config values into their text at runtime. No ``old`` may therefore contain a configurable cap
number, an availability-dependent execute_code signature line, delegate_task's "Children cannot
call" line or its delivery paragraph (``oneshot_delegate_function`` string-matches that one).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

Path = tuple[str | int, ...]
_DESC: Path = ("description",)


def _param(*names: str) -> Path:
    path: list[str] = ["parameters"]
    for name in names:
        path += ["properties", name]
    return tuple(path) + ("description",)


EDITS: dict[str, list[tuple[Path, str, str]]] = {
    "clarify": [
        (_DESC, "Ask the user one or more questions when you need a decision or feedback. Pass every question "
                "in `questions` (1-5 entries) — a single question is a one-entry array, and several INDEPENDENT "
                "questions belong in ONE call (one form beats a chain of clarify calls; if one answer would change "
                "another question, ask separately).",
         "Ask the user for a decision or feedback. Put several independent questions in one call; if one "
         "answer would change another question, ask separately."),
        (_DESC, "choices — put your recommended option FIRST, the UI marks it '(Recommended)' and auto-appends",
         "choices; put your recommended option first, the UI marks it '(Recommended)' and appends"),
        (_DESC, "never enumerated inside the question text (choices render as pickable rows; options written "
                "into the question are dead prose the user can't click). Result: {responses: [...], outcome} in "
                "question order. Each response has status",
         "never in the question text, where the user cannot click them. Result: {responses: [...], outcome}. "
         "Responses come back in question order with the question text echoed; each has status"),
        (_DESC, "unless answered); outcome is", "unless answered). outcome is"),
        (_DESC, "low-stakes questions yourself; don't use this for dangerous-command confirmation",
         "low-stakes questions yourself. Not for dangerous-command confirmation"),
        (_param("questions"), "The question(s). Each: question text (options excluded), optional choices "
                              "(recommended first; omit for free-text), optional multi_select. Responses come back "
                              "in question order with the question text echoed.",
         "1-5 questions; a single question is a one-entry array."),
    ],
    "delegate_task": [
        (_DESC, "Each gets its own conversation, terminal session, and toolset, and only its final summary returns "
                "to you. Pass every task in `tasks` — one entry spawns one subagent, several run in parallel "
                "(limit in the tasks description).",
         "Each has its own conversation, terminal session and toolset; only its final summary returns to you."),
        (_DESC, "USE FOR: reasoning-heavy subtasks, work that would flood your context, or independent parallel "
                "workstreams.\nDO NOT USE FOR (use these instead):\n- Mechanical multi-step work with no reasoning "
                "needed -> execute_code",
         "Use for reasoning-heavy subtasks, work that would flood your context, or independent parallel "
         "workstreams.\nDo not use for:\n- Mechanical multi-step work needing no reasoning -> execute_code"),
        (_DESC, "- Tasks needing user interaction -> subagents cannot ask questions",
         "- Tasks needing user interaction: subagents cannot ask questions"),
        (_DESC, "notify=True); /stop, /new, or process exit", "notify=True). /stop, /new or process exit"),
        (_DESC, "RULES:\n", "Rules:\n"),
        (_DESC, "output language, tone, or style (e.g. \"respond in Chinese\").",
         "output language, tone or style."),
        (_DESC, "Child summaries are SELF-REPORTS, not verified facts: a child claiming \"uploaded successfully\" "
                "or \"file written\" may be wrong. For external side effects",
         "Child summaries are self-reports, not verified facts. For external side effects"),
        (_DESC, "before telling the user the operation succeeded.", "before telling the user it succeeded."),
        (_DESC, "returns findings instead; the parent applies", "returns findings and the parent applies"),
        (_param("action"), "Live control of running children: 'list' = ids/goals/status/transcripts; 'steer' = "
                           "queue course-correction text into one child (subagent_id + message) without stopping "
                           "it; 'stop' = end one child early (subagent_id; partial result still returns). Control "
                           "actions return immediately; goal/tasks are ignored unless spawning.",
         "The others control running children live, return immediately and ignore tasks: 'list' = "
         "ids/goals/status/transcripts; 'steer' = queue course-correction text into one child without "
         "stopping it (subagent_id + message); 'stop' = end one child early (subagent_id; partial result "
         "still returns)."),
        (_param("message"), "For action='steer':", "For 'steer':"),
        (_param("subagent_id"), "Target for action='steer'/'stop' (ids from the spawn response or action='list').",
         "Target of 'steer'/'stop'; ids come from the spawn response or 'list'."),
        (_param("tasks"), "The task(s), up to ", "Every task to spawn: up to "),
        (_param("tasks"), "(set via delegation.max_concurrent_children). Each entry spawns one subagent with "
                          "isolated context and terminal session; a single task",
         "(set via delegation.max_concurrent_children), one subagent per entry; a single task"),
        (_param("tasks")[:-1] + ("items", "properties", "context", "description"),
         "Background THIS child needs", "Background this child needs"),
        (_param("tasks")[:-1] + ("items", "properties", "context", "description"),
         "its own context — repeat", "its own context: repeat"),
        (_param("tasks")[:-1] + ("items", "properties", "goal", "description"),
         "accomplish. Be specific and self-contained — it knows nothing about your conversation history.",
         "accomplish; specific and self-contained (it knows nothing of your conversation)."),
        (_param("tasks")[:-1] + ("items", "properties", "images", "description"),
         "Optional images this child must SEE (max 8): local file paths or http(s) URLs — e.g. a screenshot the "
         "user sent, a design mock, a chart. Vision-capable children receive the pixels on their first turn; "
         "non-vision children get path hints for vision_analyze. Text files do NOT belong here — put paths in "
         "'context' instead.",
         "Images this child must see (max 8): local file paths or http(s) URLs. Vision-capable children receive "
         "the pixels on their first turn; others get path hints for vision_analyze. Text files do NOT belong "
         "here: put their paths in 'context'."),
        (_param("tasks")[:-1] + ("items", "properties", "output_schema", "description"),
         "Optional JSON Schema this child's final answer must validate against (told to the child up front; "
         "parent validates with one bounded correction retry; result gains schema_valid, plus schema_errors on "
         "failure — the child's raw text is still returned as summary, never discarded). Keep it forgiving — "
         "require only fields you will read.",
         "JSON Schema the child's final answer must validate against (the child is told up front; the parent "
         "validates, with one bounded correction retry). The result gains schema_valid, plus schema_errors on "
         "failure; the child's raw text is still returned as summary. Keep it forgiving: require only fields "
         "you will read."),
    ],
    "execute_code": [
        (_DESC, "Run Python that calls Hermes tools programmatically. Use when you need 3+ tool calls",
         "Run Python that calls Hermes tools. Use for 3+ tool calls"),
        (_DESC, "branching, or loops (N pages/files", "branching, loops (N pages/files"),
        (_DESC, "Available via `from hermes_tools import ...`:\n\n", "Available via `from hermes_tools import ...`:\n"),
        (_DESC, "Calls run in a persistent session kernel: variables, imports, and loaded data survive across "
                "execute_code calls,",
         "The session kernel persists: variables, imports and loaded data survive across calls,"),
        (_DESC, "the FULL text is auto-saved to a file whose path rides in the result.",
         "the full text is saved to a file whose path is in the result."),
        (_DESC, "Helpers require imports: `from hermes_tools import json_parse, shell_quote, retry`. "
                "json_parse(text) — tolerant json.loads for terminal() output; shell_quote(s) — shlex.quote for "
                "dynamic shell args; retry(fn, max_attempts=3, delay=2) — exponential backoff.",
         "Helpers must be imported from hermes_tools too: json_parse(text) = tolerant json.loads for terminal() "
         "output; shell_quote(s) = shlex.quote for dynamic shell args; retry(fn, max_attempts=3, delay=2) = "
         "exponential backoff."),
        (_param("code"), "Python code to execute. Import tools with", "Python code. Import tools with"),
        (_param("reset"), "Discard the kernel's persistent state and start fresh before running this code.",
         "Discard the kernel's state before running this code."),
    ],
    "read_file": [
        (_DESC, "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. "
                "Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit "
                "for large files. Reads exceeding ",
         "Read a text file with line numbers; paginate large files with offset and limit. Use instead of "
         "cat/head/tail in terminal. Output: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. "
         "Reads over "),
        (_DESC, "return a next_offset; continue with offset to read the rest. A single line longer",
         "return a next_offset; continue from it with offset. A line longer"),
        (_DESC, "auto-extract to readable text:", "auto-extract to text:"),
        (_DESC, "Cannot read images/binary — use vision_analyze", "Cannot read images/binary; use vision_analyze"),
        (_param("limit"), "Maximum number of lines to read (default: 2000, max: 2000). Reads are additionally "
                          "capped at a ",
         "Max lines to read (max 2000), also capped at a "),
        (_param("offset"), "Line number to start reading from (1-indexed, default: 1)", "First line to read (1-indexed)."),
        (_param("path"), "Path to the file to read (absolute, relative, or ~/path)", "Absolute, relative or ~/path."),
    ],
    "search_files": [
        (_DESC, "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. "
                "Ripgrep-backed, faster than shell equivalents.",
         "Search file contents by regex, or find files by name glob. Use instead of grep/rg/find/ls in "
         "terminal; ripgrep-backed and faster."),
        (_DESC, "\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with "
                "line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by "
                "glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls. Discovery order is the fast "
                "bounded default; exact global newest-first order is an explicit opt-in and may scan the full tree.",
         ""),
        (_param("context"), "Number of context lines before and after each match (grep mode only)",
         "Content search only: context lines before and after each match."),
        (_param("file_glob"), "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)",
         "Content search only: restrict to files matching a glob (e.g. '*.py')."),
        (_param("limit"), "Maximum number of results to return (default: 50)", "Max results."),
        (_param("offset"), "Skip first N results for pagination (default: 0)", "Skip the first N results (pagination)."),
        (_param("order"), "File-search order: 'discovery' is fast bounded traversal order; 'modified' is exact "
                          "global newest-first and may scan the full tree; ignored for content",
         "File search only. 'discovery' (default): fast bounded traversal order; 'modified': exact global "
         "newest-first, may scan the full tree."),
        (_param("output_mode"), "Output format for grep mode: 'content' shows matching lines with line numbers, "
                                "'files_only' lists file paths, 'count' shows match counts per file",
         "Content search only. 'content': matching lines with line numbers; 'files_only': file paths; "
         "'count': match counts per file."),
        (_param("path"), "Directory or file to search in (default: current working directory)",
         "Directory or file to search in."),
        (_param("pattern"), "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search",
         "Regex for content search; glob for file search (e.g. '*.py', '*config*')."),
        (_param("target"), "'content' searches inside file contents, 'files' searches for files by name",
         "'content': inside files; 'files': by file name."),
    ],
    "session_search": [
        (_DESC, "Search or read past Hermes conversations (FTS5), or scroll inside one.",
         "Search or read past Hermes conversations (FTS5)."),
        (_DESC, "(top-N matching sessions, top result fully hydrated)", "(top-N matching sessions)"),
        (_DESC, "= read a whole session — how you resolve an `@session:<profile>/<id>` link (split on '/' into "
                "profile + id);",
         "= read a whole session, which is how you resolve an `@session:<profile>/<id>` link;"),
        (_DESC, "history ONLY — when the user", "history ONLY: when the user"),
        (_DESC, "Use for questions about past conversations: 'what did we do about X', 'where did we leave Y'.",
         "Use for questions about past conversations."),
        (_param("after"), "Discovery shape only. Inclusive lower bound on session start time. ISO",
         "Discovery only. Inclusive lower bound on session start time: ISO"),
        (_param("after"), "Use only when the user names a time frame. sort is a ranking bias, not a bound.",
         "Use after/before only when the user names a time frame."),
        (_param("around_message_id"), "Scroll shape. Message id to center the window on — use match_message_id",
         "Scroll shape: message id to center the window on; match_message_id"),
        (_param("before"), "Discovery shape only. Exclusive upper bound on session start time. ISO date/datetime "
                           "(a date-only value is midnight UTC that day) or relative duration (7d = older than a "
                           "week). Use only when the user names a time frame.",
         "Discovery only. Exclusive upper bound on session start time: ISO date/datetime (date-only = midnight "
         "UTC that day) or relative duration (7d = older than a week)."),
        (_param("detail"), "Discovery shape only. 'adaptive' (default) fully hydrates the top-ranked result and "
                           "returns only the exact anchor message for lower-ranked results. 'full' returns",
         "Discovery only. 'adaptive': fully hydrates the top-ranked result, returns only the exact anchor "
         "message for the rest. 'full':"),
        (_param("exclude_session_ids"), "Discovery shape only. Session ids already inspected this task. Those "
                                        "sessions and their lineage are omitted so a later query explores instead "
                                        "of repeating the same hit. Cap 20.",
         "Discovery only. Session ids already inspected this task; they and their lineage are omitted so a "
         "later query does not repeat the same hit. Max 20."),
        (_param("limit"), "Discovery shape only. Max sessions to return (default 3, max 10). Bump to 5–10 when the "
                          "topic likely spans several sessions and you want to pick the right one to scroll into.",
         "Discovery only. Max sessions to return (max 10). Use 5–10 when the topic likely spans several "
         "sessions."),
        (_param("profile"), "Optional. Read sessions from another Hermes profile's database (read-only). Use when "
                            "resolving an `@session:<profile>/<id>` link: pass the profile segment here with "
                            "session_id as the id segment. Omit to use the current profile.",
         "Another Hermes profile whose database to read (read-only); omit for the current profile. For an "
         "`@session:<profile>/<id>` link, split on '/': profile here, id in session_id."),
        (_param("query"), "Search query (discovery shape). Keywords, phrases, or boolean expressions to find in "
                          "past sessions. Omit to browse recent sessions. Ignored when session_id + "
                          "around_message_id are set (scroll shape).",
         "Discovery: keywords, phrases or boolean expressions. Omit to browse recent sessions. Ignored when "
         "session_id + around_message_id are set."),
        (_param("role_filter"), "Optional. Comma-separated roles to include. Discovery defaults to "
                                "'user,assistant' (tool output is usually noise).",
         "Comma-separated roles to include. Discovery defaults to 'user,assistant'."),
        (_param("session_id"), "Scroll shape. Session to read inside. Use the session_id returned from a prior "
                               "discovery call. Must be paired with around_message_id.",
         "Session from a prior discovery call: with around_message_id it scrolls; alone it reads the whole "
         "session."),
        (_param("sort"), "Discovery shape only. Temporal bias on top of FTS5 ranking: omit for relevance-only",
         "Discovery only. Temporal bias on top of FTS5 ranking, not a bound: omit for relevance only"),
        (_param("window"), "Scroll shape only. Messages to return on each side of the anchor (anchor itself always "
                           "included). Clamped to [1, 20]. Default 5.",
         "Scroll only. Messages on each side of the anchor (anchor always included). Clamped to [1, 20]."),
    ],
    "skill_manage": [
        (_DESC, "Create, update or delete skills: your procedural memory for recurring tasks. The call is an "
                "operations array (a single edit is a list of one); it applies atomically — any failure",
         "Create, update or delete skills, your procedural memory for recurring tasks. The operations apply "
         "atomically: any failure"),
        (_DESC, "create (full SKILL.md; lands in", "create (lands in"),
        (_DESC, "patch (targeted old_string/new_string fix — preferred; content alone REPLACES the whole file, "
                "read it via skill_view() first),",
         "patch (old_string/new_string preferred, or a full content rewrite),"),
        (_DESC, "Description: max 60 chars", "Skill description: max 60 chars"),
        (_param("operations"), "Ordered ops; each names", "Ordered ops (a single edit is a list of one); each names"),
        (("parameters", "properties", "operations", "items", "anyOf", 0, "properties", "category", "description"),
         "Optional category subdir", "Category subdir"),
        (("parameters", "properties", "operations", "items", "anyOf", 1, "properties", "file_path", "description"),
         "Optional supporting file (write_file's shape)", "Supporting file to patch (write_file's shape)"),
        (("parameters", "properties", "operations", "items", "anyOf", 1, "properties", "old_string", "description"),
         "Text to find (same matching semantics as the patch tool).", "Text to find (matching as in the patch tool)."),
        (("parameters", "properties", "operations", "items", "anyOf", 1, "properties", "replace_all", "description"),
         "Replace all occurrences (default false).", "Default false."),
        (("parameters", "properties", "operations", "items", "anyOf", 2, "properties", "content", "description"),
         "Full SKILL.md rewrite (REPLACES the whole file; last resort).",
         "Full SKILL.md that REPLACES the whole file; last resort. Read the file via skill_view() first."),
        (("parameters", "properties", "operations", "items", "anyOf", 3, "properties", "file_path", "description"),
         "Path RELATIVE to the skill's own directory, e.g. 'references/api.md' — no leading slash, never "
         "absolute; first segment references/, templates/, scripts/, or assets/.",
         "RELATIVE to the skill's own directory, e.g. 'references/api.md': never absolute, no leading slash; "
         "first segment references/, templates/, scripts/ or assets/."),
    ],
    "terminal": [
        (_DESC, "Foreground (default) returns as soon as the command finishes. To wait", "To wait"),
        (_DESC, " Background is only for commands that must keep running after this call returns; health-check a "
                "server in a separate call and manage jobs with process_manage.",
         ""),
        (_DESC, "prefixing later commands with cd.\n", "prefixing later commands with cd."),
        (_param("background"), "Keep running after this call returns; returns a session_id.",
         "Only for commands that must keep running after this call returns; returns a session_id. Health-check "
         "a server in a separate call; manage jobs with process_manage."),
        (_param("command"), "The shell command to execute", "Shell command."),
        (_param("heartbeat"), "With background=true: also notify", "Background only: also notify"),
        (_param("notify"), "With background=true. true:", "Background only. true:"),
        (_param("persist_on_release"), "With background=true: survive", "Background only: survive"),
        (_param("pty"), "With background=true: pseudo-terminal", "Background only: pseudo-terminal"),
        (_param("timeout"), "Finishing early returns at once, so set it generously.",
         "A foreground command returns as soon as it finishes, so set it generously."),
    ],
    "todo_list": [
        # The first sentence stays: a changed one must fit the 80-char tool_search catalog summary.
        (_DESC, "Use for complex tasks with 3+ steps or when the user provides multiple tasks. For 'all N items' "
                "tasks, enumerate every instance as its own checklist item so none are silently dropped.",
         "Also when the user gives multiple tasks. For 'all N items' tasks, make every instance its own item."),
        (_param("merge"), "false (default): replace the entire list with a fresh plan.",
         "false: replace the entire list."),
        (_param("todos"), "Task items to write.", "Items to write."),
        (_param("todos")[:-1] + ("items", "properties", "parent", "description"),
         "Optional id of another item, making this a nested subtask.",
         "Id of another item, making this its subtask."),
    ],
}

_skipped: set[tuple[str, int]] = set()


def _get(node: Any, path: Path) -> Any:
    for key in path:
        if isinstance(key, int):
            node = node[key] if isinstance(node, list) and key < len(node) else None
        else:
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def _set(node: Any, path: Path, value: str) -> None:
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _log_skip(tool: str, index: int, path: Path, reason: str) -> None:
    if (tool, index) in _skipped:
        return
    _skipped.add((tool, index))
    logger.warning("fork tool text: edit %d for %s at %s skipped (%s); upstream text kept",
                   index, tool, "/".join(map(str, path)), reason)


def apply_edits(function: dict[str, Any], edits: list[tuple[Path, str, str]], tool: str = "") -> tuple[dict, list[int]]:
    """``function`` with *edits* applied, as a deep copy; also the indexes of skipped edits."""
    out = copy.deepcopy(function)
    skipped = []
    for index, (path, old, new) in enumerate(edits):
        text = _get(out, path)
        count = text.count(old) if isinstance(text, str) else 0
        if count != 1:
            skipped.append(index)
            _log_skip(tool, index, path, "absent" if count == 0 else f"{count} matches")
            continue
        _set(out, path, text.replace(old, new, 1))
    return out, skipped


def apply_fork_tool_text(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*tool_defs* with :data:`EDITS` applied; edited entries are copies, registry schemas never mutate."""
    out = []
    for td in tool_defs:
        function = td.get("function") or {}
        name = function.get("name") or ""
        edits = EDITS.get(name)
        if edits:
            function, _ = apply_edits(function, edits, name)
            td = {**td, "function": function}
        out.append(td)
    return out
