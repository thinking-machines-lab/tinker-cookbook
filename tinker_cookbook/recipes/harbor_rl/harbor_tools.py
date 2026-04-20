"""Harbor tools for RL training with sandbox environments.

Tool interfaces mirror kimi-cli (src/kimi_cli/tools/) so models pretrained
with kimi-cli see familiar tools.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 16384
MAX_GLOB_MATCHES = 1000
MAX_READ_LINES = 1000
MAX_READ_LINE_LENGTH = 2000
MAX_READ_BYTES = 100 * 1024  # 100KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_result(message: str, output: str = "") -> ToolResult:
    """Build a ToolResult with kimi-cli style <system> tags."""
    if output:
        content = f"<system>{message}</system>\n{output}" if message else output
    else:
        content = f"<system>{message}</system>" if message else ""
    return simple_tool_result(content)


def _error_result(error: str, output: str = "") -> ToolResult:
    """Build an error ToolResult with kimi-cli style <system> tags."""
    if output:
        content = f"<system>ERROR: {error}</system>\n{output}"
    else:
        content = f"<system>ERROR: {error}</system>"
    return simple_tool_result(content)


def _truncate_line(line: str, max_length: int) -> str:
    if len(line) <= max_length:
        return line
    return line[: max(max_length - 3, 0)] + "..."


# ---------------------------------------------------------------------------
# HarborBashTool (existing)
# ---------------------------------------------------------------------------


class HarborBashTool:
    """Bash tool that executes commands in a sandbox.

    Wraps a SandboxInterface as a tinker_cookbook Tool via the @tool decorator.
    """

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 120) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
    ) -> ToolResult:
        """Execute a bash command in the sandbox environment.

        Use this to run shell commands, install packages, edit files, etc.
        """
        result = await self._sandbox.run_command(
            command, workdir="/", timeout=self._command_timeout, max_output_bytes=MAX_OUTPUT_CHARS
        )
        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]
        output = json.dumps({"exit_code": result.exit_code, "stdout": stdout, "stderr": stderr})
        return simple_tool_result(output)


# ---------------------------------------------------------------------------
# HarborGlobTool (mirrors src/kimi_cli/tools/file/glob.py)
# ---------------------------------------------------------------------------


class HarborGlobTool:
    """Glob tool that searches for files in a sandbox."""

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 30) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def Glob(
        self,
        pattern: Annotated[str, "Glob pattern to match files/directories."],
        directory: Annotated[
            str | None,
            "Absolute path to the directory to search in (defaults to working directory).",
        ] = None,
        include_dirs: Annotated[bool, "Whether to include directories in results."] = True,
    ) -> ToolResult:
        """Find files and directories using glob patterns.

        This tool supports standard glob syntax like *, ?, and ** for recursive searches.

        When to use:
        - Find files matching specific patterns (e.g., all Python files: *.py)
        - Search for files recursively in subdirectories (e.g., src/**/*.js)
        - Locate configuration files (e.g., *.config.*, *.json)
        - Find test files (e.g., test_*.py, *_test.go)

        Example patterns:
        - *.py - All Python files in current directory
        - src/**/*.js - All JavaScript files in src directory recursively
        - test_*.py - Python test files starting with test_
        - *.config.{js,ts} - Config files with .js or .ts extension

        Bad example patterns:
        - **, **/*.py - Starting with ** gets rejected as it would recursively search all directories.
        - node_modules/**/*.js - Avoid recursively searching known large directories."""
        # Reject patterns starting with **
        if pattern.startswith("**"):
            ls_result = await self._sandbox.run_command(
                "ls /", workdir="/", timeout=self._command_timeout
            )
            listing = ls_result.stdout[:MAX_OUTPUT_CHARS] if ls_result.exit_code == 0 else ""
            return _error_result(
                f"Pattern `{pattern}` starts with '**' which is not allowed. "
                "This would recursively search all directories and may include large "
                "directories like `node_modules`. Use more specific patterns instead. "
                "For your convenience, a list of all files and directories in the "
                "top level of the working directory is provided below.",
                output=listing,
            )

        dir_path = directory or "/"
        script = (
            "import json, pathlib\n"
            f"d = pathlib.Path({dir_path!r})\n"
            "matches = []\n"
            f"for p in sorted(d.glob({pattern!r})):\n"
            f"    if not {include_dirs!r} and p.is_dir():\n"
            "        continue\n"
            "    matches.append(str(p.relative_to(d)))\n"
            "total = len(matches)\n"
            f"matches = matches[:{MAX_GLOB_MATCHES}]\n"
            'print(json.dumps({"matches": matches, "total": total}))\n'
        )

        result = await self._sandbox.run_command(
            f"python3 -c {shlex.quote(script)}",
            workdir="/",
            timeout=self._command_timeout,
        )

        if result.exit_code != 0:
            return _error_result(
                f"Failed to search for pattern {pattern}. Error: {result.stderr[:MAX_OUTPUT_CHARS]}"
            )

        try:
            data = json.loads(result.stdout.strip())
            matches: list[str] = data["matches"]
            total: int = data["total"]
        except (json.JSONDecodeError, KeyError) as e:
            return _error_result(f"Failed to parse glob results. Error: {e}")

        if not matches:
            return _tool_result(f"No matches found for pattern `{pattern}`.")

        message = f"Found {total} matches for pattern `{pattern}`."
        if total > MAX_GLOB_MATCHES:
            message += (
                f" Only the first {MAX_GLOB_MATCHES} matches are returned. "
                "You may want to use a more specific pattern."
            )

        output = "\n".join(matches)
        return _tool_result(message, output)


# ---------------------------------------------------------------------------
# HarborGrepTool (mirrors src/kimi_cli/tools/file/grep_local.py)
# ---------------------------------------------------------------------------


def _build_rg_args(
    rg_path: str,
    pattern: str,
    path: str,
    output_mode: str,
    before_context: int | None,
    after_context: int | None,
    context: int | None,
    line_number: bool,
    ignore_case: bool,
    file_type: str | None,
    glob_filter: str | None,
    multiline: bool,
    include_ignored: bool,
) -> list[str]:
    """Build ripgrep command-line arguments, mirroring kimi-cli's _build_rg_args."""
    args: list[str] = [rg_path]

    # Fixed args
    if output_mode != "content":
        args.extend(["--max-columns", "500"])
    args.append("--hidden")
    if include_ignored:
        args.append("--no-ignore")
    for vcs_dir in (".git", ".svn", ".hg", ".bzr", ".jj", ".sl"):
        args.extend(["--glob", f"!{vcs_dir}"])

    # Search options
    if ignore_case:
        args.append("--ignore-case")
    if multiline:
        args.extend(["--multiline", "--multiline-dotall"])

    # Content display options
    if output_mode == "content":
        if before_context is not None:
            args.extend(["--before-context", str(before_context)])
        if after_context is not None:
            args.extend(["--after-context", str(after_context)])
        if context is not None:
            args.extend(["--context", str(context)])
        if line_number:
            args.append("--line-number")

    # File filtering
    if glob_filter:
        args.extend(["--glob", glob_filter])
    if file_type:
        args.extend(["--type", file_type])

    # Output mode
    if output_mode == "files_with_matches":
        args.append("--files-with-matches")
    elif output_mode == "count_matches":
        args.extend(["--count-matches", "--with-filename"])

    args.append("--")
    args.append(pattern)
    args.append(path)

    return args


class HarborGrepTool:
    """Grep tool backed by ripgrep in a sandbox."""

    def __init__(
        self,
        sandbox: SandboxInterface,
        rg_path: str = "/usr/local/bin/rg",
        command_timeout: int = 60,
    ) -> None:
        self._sandbox = sandbox
        self._rg_path = rg_path
        self._command_timeout = command_timeout

    @tool
    async def Grep(
        self,
        pattern: Annotated[
            str,
            Field(description="The regular expression pattern to search for in file contents"),
        ],
        path: Annotated[
            str,
            Field(
                description=(
                    "File or directory to search in. Defaults to current working directory. "
                    "If specified, it must be an absolute path."
                ),
                default=".",
            ),
        ] = ".",
        glob: Annotated[
            str | None,
            Field(
                description=(
                    "Glob pattern to filter files (e.g. `*.js`, `*.{ts,tsx}`). "
                    "No filter by default."
                ),
                default=None,
            ),
        ] = None,
        output_mode: Annotated[
            str,
            Field(
                description=(
                    "`content`: Show matching lines (supports `-B`, `-A`, `-C`, `-n`, "
                    "`head_limit`); `files_with_matches`: Show file paths (supports "
                    "`head_limit`); `count_matches`: Show total number of matches. "
                    "Defaults to `files_with_matches`."
                ),
                default="files_with_matches",
            ),
        ] = "files_with_matches",
        before_context: Annotated[
            int | None,
            Field(
                alias="-B",
                description=(
                    "Number of lines to show before each match (the `-B` option). "
                    "Requires `output_mode` to be `content`."
                ),
                default=None,
            ),
        ] = None,
        after_context: Annotated[
            int | None,
            Field(
                alias="-A",
                description=(
                    "Number of lines to show after each match (the `-A` option). "
                    "Requires `output_mode` to be `content`."
                ),
                default=None,
            ),
        ] = None,
        context: Annotated[
            int | None,
            Field(
                alias="-C",
                description=(
                    "Number of lines to show before and after each match (the `-C` option). "
                    "Requires `output_mode` to be `content`."
                ),
                default=None,
            ),
        ] = None,
        line_number: Annotated[
            bool,
            Field(
                alias="-n",
                description=(
                    "Show line numbers in output (the `-n` option). "
                    "Requires `output_mode` to be `content`. Defaults to true."
                ),
                default=True,
            ),
        ] = True,
        ignore_case: Annotated[
            bool,
            Field(
                alias="-i", description="Case insensitive search (the `-i` option).", default=False
            ),
        ] = False,
        type: Annotated[
            str | None,
            Field(
                description=(
                    "File type to search. Examples: py, rust, js, ts, go, java, etc. "
                    "More efficient than `glob` for standard file types."
                ),
                default=None,
            ),
        ] = None,
        head_limit: Annotated[
            int | None,
            Field(
                description=(
                    "Limit output to first N lines/entries, equivalent to `| head -N`. "
                    "Works across all output modes: content (limits output lines), "
                    "files_with_matches (limits file paths), count_matches (limits count "
                    "entries). Defaults to 250. "
                    "Pass 0 for unlimited (use sparingly — large result sets waste context)."
                ),
                default=250,
                ge=0,
            ),
        ] = 250,
        offset: Annotated[
            int,
            Field(
                description=(
                    "Skip first N lines/entries before applying head_limit, "
                    "equivalent to `| tail -n +N | head -N`. "
                    "Works across all output modes. Defaults to 0."
                ),
                default=0,
                ge=0,
            ),
        ] = 0,
        multiline: Annotated[
            bool,
            Field(
                description=(
                    "Enable multiline mode where `.` matches newlines and patterns can span "
                    "lines (the `-U` and `--multiline-dotall` options). "
                    "By default, multiline mode is disabled."
                ),
                default=False,
            ),
        ] = False,
        include_ignored: Annotated[
            bool,
            Field(
                description=(
                    "Include files that are ignored by `.gitignore`, `.ignore`, and other "
                    "ignore rules. Useful for searching gitignored artifacts such as build "
                    "outputs (e.g. `dist/`, `build/`) or `node_modules`. "
                    "Defaults to false."
                ),
                default=False,
            ),
        ] = False,
    ) -> ToolResult:
        """A powerful search tool based on ripgrep.

        Tips:
        - ALWAYS use Grep tool instead of running `grep` or `rg` command.
        - Use ripgrep pattern syntax, not grep syntax. Escape braces like \\{ to search for {.
        - Hidden files (dotfiles like .gitlab-ci.yml, .eslintrc.json) are always searched.
        - To search files excluded by .gitignore, set include_ignored to true."""
        args = _build_rg_args(
            rg_path=self._rg_path,
            pattern=pattern,
            path=path,
            output_mode=output_mode,
            before_context=before_context,
            after_context=after_context,
            context=context,
            line_number=line_number,
            ignore_case=ignore_case,
            file_type=type,
            glob_filter=glob,
            multiline=multiline,
            include_ignored=include_ignored,
        )
        cmd = " ".join(shlex.quote(arg) for arg in args)

        result = await self._sandbox.run_command(
            cmd, workdir="/", timeout=self._command_timeout, max_output_bytes=MAX_OUTPUT_CHARS
        )

        # rg exit codes: 0=matches found, 1=no matches, 2+=error
        if result.exit_code == 1:
            return _tool_result("No matches found")
        if result.exit_code != 0:
            return _error_result(f"Failed to grep. Error: {result.stderr[:MAX_OUTPUT_CHARS]}")

        output = result.stdout
        if not output or not output.strip():
            return _tool_result("No matches found")

        message = ""

        # Strip path prefix for relative paths (per-line to avoid corrupting match content)
        if path not in (".", "/"):
            prefix = path.rstrip("/") + "/"
            output = "\n".join(line.removeprefix(prefix) for line in output.split("\n"))

        # Split into lines for post-processing
        lines = output.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]

        # Count matches summary (for count_matches mode)
        if output_mode == "count_matches":
            total_matches = 0
            total_files = 0
            for line in lines:
                idx = line.rfind(":")
                if idx > 0:
                    try:
                        total_matches += int(line[idx + 1 :])
                        total_files += 1
                    except ValueError:
                        pass
            count_summary = f"Found {total_matches} total occurrences across {total_files} files."
            message = count_summary

        # Offset + head_limit pagination
        if offset > 0:
            lines = lines[offset:]

        if head_limit and len(lines) > head_limit:
            total = len(lines) + offset
            lines = lines[:head_limit]
            trunc_msg = (
                f"Results truncated to {head_limit} lines (total: {total}). "
                f"Use offset={offset + head_limit} to see more."
            )
            message = f"{message} {trunc_msg}" if message else trunc_msg

        output = "\n".join(lines)

        if not output and not message:
            return _tool_result("No matches found")

        return _tool_result(message, output)


# ---------------------------------------------------------------------------
# HarborReadFileTool (mirrors src/kimi_cli/tools/file/read.py)
# ---------------------------------------------------------------------------


class HarborReadFileTool:
    """ReadFile tool that reads files from a sandbox."""

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 60) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def ReadFile(
        self,
        path: Annotated[
            str,
            "The path to the file to read. "
            "Absolute paths are required when reading files outside the working directory.",
        ],
        line_offset: Annotated[
            int,
            Field(
                description=(
                    "The line number to start reading from. "
                    "By default read from the beginning of the file. "
                    "Set this when the file is too large to read at once. "
                    "Negative values read from the end of the file "
                    f"(e.g. -100 reads the last 100 lines). "
                    f"The absolute value of negative offset cannot exceed {MAX_READ_LINES}."
                ),
                default=1,
            ),
        ] = 1,
        n_lines: Annotated[
            int,
            Field(
                description=(
                    "The number of lines to read. "
                    f"By default read up to {MAX_READ_LINES} lines, which is the max "
                    "allowed value. Set this value when the file is too large to read at once."
                ),
                default=MAX_READ_LINES,
                ge=1,
            ),
        ] = MAX_READ_LINES,
    ) -> ToolResult:
        """Read text content from a file.

        Tips:
        - Content will be returned with a line number before each line like cat -n format.
        - Use line_offset and n_lines parameters when you only need to read a part of the file.
        - Always read multiple files in one response when possible.
        - The maximum number of lines that can be read at once is 1000.
        - Any lines longer than 2000 characters will be truncated, ending with "..."."""
        if not path or not path.strip():
            return _error_result("path is required")

        # Validate line_offset
        if line_offset == 0:
            return _error_result(
                "line_offset cannot be 0; use 1 for the first line or -1 for the last line"
            )
        if line_offset < -MAX_READ_LINES:
            return _error_result(
                f"line_offset cannot be less than -{MAX_READ_LINES}. "
                "Use a positive line_offset with the total line count "
                "to read from a specific position."
            )

        # Read 2x the byte budget so we can count total_lines even when content is capped
        result = await self._sandbox.read_file(
            path, max_bytes=MAX_READ_BYTES * 2, timeout=self._command_timeout
        )

        if result.exit_code != 0:
            stderr = result.stderr.strip()
            if "No such file" in stderr or "not found" in stderr.lower():
                return _error_result(f"`{path}` does not exist.")
            if "Is a directory" in stderr:
                return _error_result(f"`{path}` is not a file.")
            return _error_result(f"Failed to read {path}. Error: {stderr}")

        content = result.stdout
        all_lines = content.splitlines()
        total_lines = len(all_lines)

        n_lines = min(n_lines, MAX_READ_LINES)

        if line_offset < 0:
            return self._format_tail(all_lines, total_lines, line_offset, n_lines, path)
        else:
            return self._format_forward(all_lines, total_lines, line_offset, n_lines, path)

    def _format_forward(
        self,
        all_lines: list[str],
        total_lines: int,
        line_offset: int,
        n_lines: int,
        path: str,
    ) -> ToolResult:
        lines_read: list[str] = []
        truncated_line_numbers: list[int] = []
        n_bytes = 0
        max_lines_reached = False
        max_bytes_reached = False

        for i in range(line_offset - 1, len(all_lines)):
            line = all_lines[i]
            truncated = _truncate_line(line, MAX_READ_LINE_LENGTH)
            if truncated != line:
                truncated_line_numbers.append(i + 1)
            lines_read.append(truncated)
            n_bytes += len(truncated.encode("utf-8"))

            if len(lines_read) >= n_lines:
                break
            if len(lines_read) >= MAX_READ_LINES:
                max_lines_reached = True
                break
            if n_bytes >= MAX_READ_BYTES:
                max_bytes_reached = True
                break

        formatted = []
        for idx, line in enumerate(lines_read):
            line_num = line_offset + idx
            formatted.append(f"{line_num:6d}\t{line}")

        message = (
            f"{len(lines_read)} lines read from file starting from line {line_offset}."
            if lines_read
            else "No lines read from file."
        )
        message += f" Total lines in file: {total_lines}."
        if max_lines_reached:
            message += f" Max {MAX_READ_LINES} lines reached."
        elif max_bytes_reached:
            message += f" Max {MAX_READ_BYTES} bytes reached."
        elif len(lines_read) < n_lines:
            message += " End of file reached."
        if truncated_line_numbers:
            message += f" Lines {truncated_line_numbers} were truncated."

        output = "\n".join(formatted)
        return _tool_result(message, output)

    def _format_tail(
        self,
        all_lines: list[str],
        total_lines: int,
        line_offset: int,
        n_lines: int,
        path: str,
    ) -> ToolResult:
        tail_count = abs(line_offset)

        # Build tail buffer: (line_no, truncated_line, was_truncated)
        tail_buf: deque[tuple[int, str, bool]] = deque(maxlen=tail_count)
        for i, line in enumerate(all_lines):
            truncated = _truncate_line(line, MAX_READ_LINE_LENGTH)
            tail_buf.append((i + 1, truncated, truncated != line))

        all_entries = list(tail_buf)
        line_limit = min(n_lines, MAX_READ_LINES)
        candidates = all_entries[:line_limit]
        max_lines_reached = len(all_entries) > MAX_READ_LINES and len(candidates) == MAX_READ_LINES

        # Apply MAX_BYTES — keep newest lines that fit
        total_candidate_bytes = sum(len(e[1].encode("utf-8")) for e in candidates)
        max_bytes_reached = False
        if total_candidate_bytes > MAX_READ_BYTES:
            max_bytes_reached = True
            kept = 0
            nb = 0
            for entry in reversed(candidates):
                nb += len(entry[1].encode("utf-8"))
                if nb > MAX_READ_BYTES:
                    break
                kept += 1
            candidates = candidates[len(candidates) - kept :]

        lines_out: list[str] = []
        truncated_line_numbers: list[int] = []
        for line_no, truncated, was_truncated in candidates:
            if was_truncated:
                truncated_line_numbers.append(line_no)
            lines_out.append(f"{line_no:6d}\t{truncated}")

        start_line = candidates[0][0] if candidates else total_lines + 1
        message = (
            f"{len(candidates)} lines read from file starting from line {start_line}."
            if candidates
            else "No lines read from file."
        )
        message += f" Total lines in file: {total_lines}."
        if max_lines_reached:
            message += f" Max {MAX_READ_LINES} lines reached."
        elif max_bytes_reached:
            message += f" Max {MAX_READ_BYTES} bytes reached."
        elif len(candidates) < n_lines:
            message += " End of file reached."
        if truncated_line_numbers:
            message += f" Lines {truncated_line_numbers} were truncated."

        output = "\n".join(lines_out)
        return _tool_result(message, output)


# ---------------------------------------------------------------------------
# HarborWriteFileTool (mirrors src/kimi_cli/tools/file/write.py)
# ---------------------------------------------------------------------------


class HarborWriteFileTool:
    """WriteFile tool that writes files in a sandbox."""

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 60) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def WriteFile(
        self,
        path: Annotated[
            str,
            "The path to the file to write. "
            "Absolute paths are required when writing files outside the working directory.",
        ],
        content: Annotated[str, "The content to write to the file."],
        mode: Annotated[
            Literal["overwrite", "append"],
            "The mode to use to write to the file. "
            "Two modes are supported: `overwrite` for overwriting the whole file and "
            "`append` for appending to the end of an existing file.",
        ] = "overwrite",
    ) -> ToolResult:
        """Write content to a file.

        Tips:
        - When mode is not specified, defaults to overwrite. Always write with caution.
        - When content is too long (e.g., > 100 lines), use this tool multiple times.
        - Use overwrite mode first time, then append mode after."""
        if not path or not path.strip():
            return _error_result("path is required")

        try:
            if mode == "overwrite":
                result = await self._sandbox.write_file(
                    path, content, timeout=self._command_timeout
                )
            else:
                # Append directly via shell to avoid truncating large files
                result = await self._sandbox.run_command(
                    f"cat >> {shlex.quote(path)} << 'HARBOR_EOF'\n{content}\nHARBOR_EOF",
                    workdir="/",
                    timeout=self._command_timeout,
                )

            if result.exit_code != 0:
                return _error_result(
                    f"Failed to write to {path}. Error: {result.stderr[:MAX_OUTPUT_CHARS]}"
                )

            # Get file size
            stat_result = await self._sandbox.run_command(
                f"stat -c %s {shlex.quote(path)}",
                workdir="/",
                timeout=self._command_timeout,
            )
            file_size = stat_result.stdout.strip() if stat_result.exit_code == 0 else "unknown"
            action = "overwritten" if mode == "overwrite" else "appended to"
            return _tool_result(f"File successfully {action}. Current size: {file_size} bytes.")

        except Exception as e:
            return _error_result(f"Failed to write to {path}. Error: {e}")


# ---------------------------------------------------------------------------
# HarborStrReplaceFileTool (mirrors src/kimi_cli/tools/file/replace.py)
# ---------------------------------------------------------------------------


class Edit(BaseModel):
    """A single string replacement edit."""

    old: str = Field(description="The old string to replace. Can be multi-line.")
    new: str = Field(description="The new string to replace with. Can be multi-line.")
    replace_all: bool = Field(description="Whether to replace all occurrences.", default=False)


class HarborStrReplaceFileTool:
    """StrReplaceFile tool that edits files in a sandbox."""

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 60) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def StrReplaceFile(
        self,
        path: Annotated[
            str,
            "The path to the file to edit. "
            "Absolute paths are required when editing files outside the working directory.",
        ],
        edit: Annotated[
            Edit | list[Edit],
            "The edit(s) to apply to the file. "
            "You can provide a single edit or a list of edits here.",
        ],
    ) -> ToolResult:
        """Replace specific strings within a specified file.

        Tips:
        - Only use on text files.
        - Multi-line strings are supported.
        - Can specify single or multiple edits in one call.
        - Should prefer this tool over WriteFile tool and Shell sed command."""
        if not path or not path.strip():
            return _error_result("path is required")

        # Read the full file (large cap to avoid truncating content before editing)
        result = await self._sandbox.read_file(
            path, max_bytes=10 * 1024 * 1024, timeout=self._command_timeout
        )
        if result.exit_code != 0:
            stderr = result.stderr.strip()
            if "No such file" in stderr or "not found" in stderr.lower():
                return _error_result(f"`{path}` does not exist.")
            return _error_result(f"Failed to read {path}. Error: {stderr}")

        content = result.stdout
        original_content = content

        edits = [edit] if isinstance(edit, Edit) else edit

        # Apply edits sequentially, counting replacements as we go
        total_replacements = 0
        for e in edits:
            if e.replace_all:
                total_replacements += content.count(e.old)
                content = content.replace(e.old, e.new)
            else:
                if e.old in content:
                    total_replacements += 1
                content = content.replace(e.old, e.new, 1)

        if content == original_content:
            return _error_result(
                "No replacements were made. The old string was not found in the file."
            )

        # Write back
        write_result = await self._sandbox.write_file(path, content, timeout=self._command_timeout)
        if write_result.exit_code != 0:
            return _error_result(
                f"Failed to write {path}. Error: {write_result.stderr[:MAX_OUTPUT_CHARS]}"
            )

        return _tool_result(
            f"File successfully edited. "
            f"Applied {len(edits)} edit(s) with {total_replacements} total replacement(s)."
        )


# ---------------------------------------------------------------------------
# HarborReward (existing)
# ---------------------------------------------------------------------------


@dataclass
class HarborReward:
    """Reward function for Harbor tasks.

    Grades by uploading test files to the sandbox, running test.sh,
    and parsing reward from /logs/verifier/reward.txt or reward.json.

    Called once at episode end with the full message history.
    """

    tests_dir: Path
    sandbox: SandboxInterface
    grader_timeout: int = 60

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the completed episode by running test.sh in the sandbox."""
        try:
            # 1. Upload test files to /tests/ in sandbox
            await self._upload_tests()

            # 2. Create log directory and run test.sh
            # Run from /root (not /) because test.sh checks if PWD=/ and exits early
            await self.sandbox.run_command("mkdir -p /logs/verifier", workdir="/root")
            result = await self.sandbox.run_command(
                "bash /tests/test.sh",
                workdir="/root",
                timeout=self.grader_timeout,
            )
            logger.info("test.sh completed with exit_code=%d", result.exit_code)
            if result.stdout:
                logger.debug("test.sh stdout: %s", result.stdout[:500])
            if result.stderr:
                logger.debug("test.sh stderr: %s", result.stderr[:500])

            # 3. Parse reward
            reward = await self._parse_reward()
            return reward, {"reward": reward, "test_passed": float(reward > 0)}

        except Exception as e:
            logger.error("Harbor grading failed: %s", e)
            return 0.0, {"reward": 0.0, "test_passed": 0.0, "grading_error": 1.0}

    async def _upload_tests(self) -> None:
        """Upload test files from local tests_dir to /tests/ in sandbox."""
        await self.sandbox.run_command("mkdir -p /tests", workdir="/")
        for file_path in self.tests_dir.iterdir():
            if not file_path.is_file():
                continue
            content = file_path.read_text()
            target = f"/tests/{file_path.name}"
            await self.sandbox.write_file(target, content, executable=(file_path.suffix == ".sh"))

    async def _parse_reward(self) -> float:
        """Parse reward from /logs/verifier/reward.txt or reward.json."""
        # Try reward.txt first
        result = await self.sandbox.read_file("/logs/verifier/reward.txt")
        if result.exit_code == 0 and result.stdout.strip():
            reward = float(result.stdout.strip())
            logger.info("Parsed reward from reward.txt: %s", reward)
            return reward

        # Try reward.json
        result = await self.sandbox.read_file("/logs/verifier/reward.json")
        if result.exit_code == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            reward = float(data.get("reward", 0.0))
            logger.info("Parsed reward from reward.json: %s", reward)
            return reward

        logger.warning("No reward file found at /logs/verifier/")
        return 0.0
