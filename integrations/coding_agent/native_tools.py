"""Checkout-confined file and shell tools for OpenSRE's built-in coding agent.

Every path resolves inside one checkout: ``..``, an absolute path elsewhere, or
a symlink that leads out is refused, and nothing under ``.git`` is written.
``bash`` runs in the checkout with a filtered environment (no OpenSRE or
provider secrets) and refuses git verbs that commit, push, or rewrite history,
because the caller owns commit and push. A tool never raises for a bad request:
it returns ``{"ok": False, "error": ...}`` so the model can correct itself.
Once the run's deadline passes every tool refuses, so the model stops and
summarizes instead of starting more work.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from core.tool import RegisteredTool
from core.tool_framework.utils import object_schema, string_property
from integrations.coding_agent.models import Progress
from integrations.llm_cli.agent_exec import poll_agent_process
from integrations.llm_cli.subprocess_env import build_cli_subprocess_env

_SOURCE = "coding_agent"
_READ_LINE_LIMIT = 2000
_MAX_LINE_CHARS = 2000
_MAX_OUTPUT_CHARS = 30000
_MAX_LISTED_PATHS = 200
_MAX_MATCH_LINES = 200
_DEFAULT_COMMAND_TIMEOUT_SEC = 120
_MAX_COMMAND_TIMEOUT_SEC = 600
_GIT_TIMEOUT_SEC = 30
_BINARY_SNIFF_BYTES = 8192

#: Git verbs that commit, push, move branches, or discard work. The caller
#: commits and pushes the result, so the agent leaves its edits in the tree.
_REFUSED_GIT_VERBS = frozenset(
    {
        "am",
        "checkout",
        "cherry-pick",
        "clean",
        "commit",
        "config",
        "filter-branch",
        "merge",
        "pull",
        "push",
        "rebase",
        "remote",
        "reset",
        "restore",
        "revert",
        "stash",
        "switch",
        "tag",
        "update-ref",
        "worktree",
    }
)
_GIT_GLOBAL_TAKES_VALUE = frozenset(
    {"-C", "-c", "--config-env", "--exec-path", "--git-dir", "--namespace", "--work-tree"}
)
_SHELL_OPERATORS = frozenset({"&&", "||", "|", ";", "&"})
# ``git`` as its own word (also ``/usr/bin/git``), not ``.git``, ``gitk`` or ``git-lfs``.
_GIT_WORD = re.compile(r"(?<![-\w.])git(?![-\w])")
_TOKEN_TRIM = "\"'`;&|()"

#: Keeps package managers and test runners non-interactive, as in CI, and keeps
#: Python bytecode out of the diff the caller commits.
_COMMAND_ENV = {
    "CI": "true",
    "NO_COLOR": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

_PATH_DESCRIPTION = "File path relative to the checkout root."
_DEADLINE_ERROR = (
    "The time budget for this run is used up. Stop calling tools and write your final "
    "summary now: what you changed, what you verified, and what is left."
)


class _RefusedInput(ValueError):
    """A tool request the model can correct (bad path, missing read, ambiguous edit)."""


def refused_git_verb(command: str) -> str | None:
    """The first git verb in *command* that the agent may not run, else ``None``."""
    for match in _GIT_WORD.finditer(command):
        verb = _git_verb(command[match.end() :])
        if verb in _REFUSED_GIT_VERBS:
            return verb
    return None


def _git_verb(rest: str) -> str:
    """The subcommand after ``git`` and its global options, or ``""`` when there is none."""
    takes_value = False
    for raw in rest.split():
        if raw in _SHELL_OPERATORS:
            return ""
        token = raw.strip(_TOKEN_TRIM)
        if takes_value:
            takes_value = False
            continue
        if token in _GIT_GLOBAL_TAKES_VALUE:
            takes_value = True
            continue
        if not token or token.startswith("-"):
            continue
        return token
    return ""


def _clip(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Keep the head and tail of long output; failures usually print at the end."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n… [{omitted} characters omitted] …\n{text[-half:]}"


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(number, high))


class WorkspaceTools:
    """The tools one coding run uses, bound to its checkout, deadline, and progress line."""

    def __init__(
        self,
        workspace: str,
        *,
        deadline: float,
        on_progress: Progress | None = None,
    ) -> None:
        self._root = Path(workspace).expanduser().resolve()
        self._deadline = deadline
        self._on_progress = on_progress
        self._read: set[Path] = set()
        self._shell = shutil.which("bash") or "/bin/sh"
        self._env = build_cli_subprocess_env(_COMMAND_ENV)
        self.deadline_reached = False

    def registered_tools(self) -> list[RegisteredTool]:
        """The six tools, in the order the model sees them."""
        return [
            _tool(
                "read_file",
                (
                    "Read a text file from the checkout. Returns numbered lines "
                    "(line number, tab, text). Read a file before editing or overwriting it. "
                    "Use offset/limit for long files."
                ),
                {
                    "path": string_property(description=_PATH_DESCRIPTION, min_length=1),
                    "offset": _integer("First line to return (1-based). Defaults to 1."),
                    "limit": _integer(f"Lines to return. Defaults to {_READ_LINE_LIMIT}."),
                },
                ("path",),
                self.read_file,
            ),
            _tool(
                "edit_file",
                (
                    "Replace an exact string in a file you have read. old_string must match "
                    "the file exactly, including indentation (without the line-number prefix "
                    "read_file adds), and be unique unless replace_all is true."
                ),
                {
                    "path": string_property(description=_PATH_DESCRIPTION, min_length=1),
                    "old_string": string_property(
                        description="Exact text to replace.", min_length=1
                    ),
                    "new_string": string_property(description="Replacement text."),
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of exactly one.",
                    },
                },
                ("path", "old_string", "new_string"),
                self.edit_file,
            ),
            _tool(
                "write_file",
                (
                    "Create a file, or overwrite one you have read, with the full content "
                    "given. Prefer edit_file for changes to an existing file."
                ),
                {
                    "path": string_property(description=_PATH_DESCRIPTION, min_length=1),
                    "content": string_property(description="Complete new file content."),
                },
                ("path", "content"),
                self.write_file,
            ),
            _tool(
                "glob",
                (
                    "List files in the checkout matching a glob, newest first. '*' stays "
                    "within one directory; use '**/' to search subdirectories "
                    "(e.g. '**/*.py', 'src/**/test_*.ts'). Ignored files are skipped."
                ),
                {"pattern": string_property(description="Glob pattern.", min_length=1)},
                ("pattern",),
                self.glob,
            ),
            _tool(
                "grep",
                (
                    "Search file contents with an extended regular expression. Returns "
                    "path:line:text matches. Ignored and binary files are skipped."
                ),
                {
                    "pattern": string_property(description="Extended regex.", min_length=1),
                    "path": string_property(
                        description="File or directory to search. Defaults to the checkout."
                    ),
                    "glob": string_property(
                        description="Only search files matching this glob, e.g. '**/*.py'."
                    ),
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive."},
                },
                ("pattern",),
                self.grep,
            ),
            _tool(
                "bash",
                (
                    "Run a shell command in the checkout root, e.g. a failing test, linter, "
                    "type check, build, or dependency install. Returns exit code, stdout and "
                    "stderr. Commands run non-interactively with CI=true. git commit, push, "
                    "reset, checkout, and other history-changing git commands are refused."
                ),
                {
                    "command": string_property(description="Shell command.", min_length=1),
                    "timeout_seconds": _integer(
                        f"Seconds before the command is stopped. Defaults to "
                        f"{_DEFAULT_COMMAND_TIMEOUT_SEC}, at most {_MAX_COMMAND_TIMEOUT_SEC}."
                    ),
                },
                ("command",),
                self.bash,
            ),
        ]

    def read_file(self, path: str, offset: Any = 1, limit: Any = _READ_LINE_LIMIT) -> dict:
        try:
            target = self._checked(path)
            if not target.is_file():
                raise _RefusedInput(f"{self._rel(target)} is not a file.")
            data = target.read_bytes()
        except (_RefusedInput, OSError) as exc:
            return _error(exc)
        if b"\0" in data[:_BINARY_SNIFF_BYTES]:
            return _error(f"{self._rel(target)} is a binary file.")
        self._progress(f"Reading {self._rel(target)}")
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = _bounded_int(offset, default=1, low=1, high=max(len(lines), 1))
        count = _bounded_int(limit, default=_READ_LINE_LIMIT, low=1, high=_READ_LINE_LIMIT)
        chunk = lines[start - 1 : start - 1 + count]
        self._read.add(target)
        return {
            "ok": True,
            "path": self._rel(target),
            "content": "\n".join(
                f"{number:>6}\t{line[:_MAX_LINE_CHARS]}" for number, line in enumerate(chunk, start)
            ),
            "total_lines": len(lines),
            "truncated": start - 1 + count < len(lines),
        }

    def edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: Any = False
    ) -> dict:
        try:
            target = self._checked(path, write=True)
            self._require_read(target)
            text = _read_text(target)
            updated, replaced = _replaced(
                text, old_string, new_string, replace_all=bool(replace_all), label=self._rel(target)
            )
            _write_text(target, updated)
        except (_RefusedInput, OSError, UnicodeDecodeError) as exc:
            return _error(exc)
        self._progress(f"Editing {self._rel(target)}")
        return {"ok": True, "path": self._rel(target), "replacements": replaced}

    def write_file(self, path: str, content: str) -> dict:
        try:
            target = self._checked(path, write=True)
            if target.is_dir():
                raise _RefusedInput(f"{self._rel(target)} is a directory.")
            if target.exists():
                self._require_read(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text(target, content)
        except (_RefusedInput, OSError) as exc:
            return _error(exc)
        self._read.add(target)
        self._progress(f"Writing {self._rel(target)}")
        return {"ok": True, "path": self._rel(target), "bytes": len(content.encode("utf-8"))}

    def glob(self, pattern: str) -> dict:
        if self._past_deadline():
            return _error(_DEADLINE_ERROR)
        listed = self._git("ls-files", "-z", "-co", "--exclude-standard", "--", f":(glob){pattern}")
        if isinstance(listed, str):
            return _error(listed)
        found = [self._root / name for name in listed.stdout.split("\0") if name]
        existing = [path for path in found if path.is_file()]
        existing.sort(key=_mtime, reverse=True)
        return {
            "ok": True,
            "paths": [self._rel(path) for path in existing[:_MAX_LISTED_PATHS]],
            "total": len(existing),
            "truncated": len(existing) > _MAX_LISTED_PATHS,
        }

    def grep(
        self,
        pattern: str,
        path: str = "",
        glob: str = "",
        ignore_case: Any = False,
    ) -> dict:
        if self._past_deadline():
            return _error(_DEADLINE_ERROR)
        try:
            scope = self._rel(self._checked(path)) if path.strip() else ""
        except _RefusedInput as exc:
            return _error(exc)
        if glob.strip():
            prefix = f"{scope}/" if scope and scope != "." else ""
            pathspec = f":(glob){prefix}{glob.strip()}"
        else:
            pathspec = scope or "."
        flags = ["-i"] if ignore_case else []
        searched = self._git(
            "grep", "-n", "-I", "--untracked", "-E", *flags, "-e", pattern, "--", pathspec
        )
        if isinstance(searched, str):
            return _error(searched)
        if searched.returncode == 1:
            return {"ok": True, "matches": [], "total": 0, "truncated": False}
        if searched.returncode != 0:
            return _error(searched.stderr.strip() or "git grep failed.")
        lines = [line[:_MAX_LINE_CHARS] for line in searched.stdout.splitlines()]
        return {
            "ok": True,
            "matches": lines[:_MAX_MATCH_LINES],
            "total": len(lines),
            "truncated": len(lines) > _MAX_MATCH_LINES,
        }

    def bash(self, command: str, timeout_seconds: Any = _DEFAULT_COMMAND_TIMEOUT_SEC) -> dict:
        if self._past_deadline():
            return _error(_DEADLINE_ERROR)
        if not command.strip():
            return _error("Give a command to run.")
        verb = refused_git_verb(command)
        if verb is not None:
            return _error(
                f"git {verb} is not allowed here: OpenSRE commits and pushes the result "
                "itself. Leave your changes in the working tree."
            )
        remaining = max(int(self._deadline - time.monotonic()), 1)
        timeout = _bounded_int(
            timeout_seconds,
            default=_DEFAULT_COMMAND_TIMEOUT_SEC,
            low=1,
            high=min(_MAX_COMMAND_TIMEOUT_SEC, remaining),
        )
        self._progress(f"Running {' '.join(command.split())[:80]}")
        outcome = poll_agent_process(
            [self._shell, "-c", command],
            cwd=str(self._root),
            env=self._env,
            timeout_sec=timeout,
        )
        if outcome.spawn_error:
            return _error(outcome.spawn_error)
        return {
            "ok": outcome.returncode == 0 and not outcome.timed_out,
            "exit_code": outcome.returncode,
            "timed_out": outcome.timed_out,
            "stdout": _clip(outcome.stdout),
            "stderr": _clip(outcome.stderr),
        }

    def _checked(self, path: str, *, write: bool = False) -> Path:
        """Resolve *path* inside the checkout, or refuse it; file tools also enforce the deadline."""
        if self._past_deadline():
            raise _RefusedInput(_DEADLINE_ERROR)
        raw = (path or "").strip()
        if not raw:
            raise _RefusedInput("A path is required.")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise _RefusedInput(f"{raw} is outside the checkout; only its files can be used.")
        parts = resolved.relative_to(self._root).parts
        if write and parts and parts[0] == ".git":
            raise _RefusedInput("Files under .git cannot be written.")
        return resolved

    def _require_read(self, target: Path) -> None:
        if target not in self._read:
            raise _RefusedInput(f"Read {self._rel(target)} with read_file before changing it.")

    def _rel(self, path: Path) -> str:
        relative = path.relative_to(self._root).as_posix()
        return relative or "."

    def _past_deadline(self) -> bool:
        if time.monotonic() >= self._deadline:
            self.deadline_reached = True
        return self.deadline_reached

    def _progress(self, step: str) -> None:
        if self._on_progress is not None:
            self._on_progress(step)

    def _git(self, *args: str) -> subprocess.CompletedProcess[str] | str:
        """Run a read-only git query in the checkout; a string is the failure to report."""
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self._root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_GIT_TIMEOUT_SEC,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"git {args[0]} failed: {exc}"


def _replaced(text: str, old: str, new: str, *, replace_all: bool, label: str) -> tuple[str, int]:
    """Apply one exact replacement (or all of them); CRLF files match LF-typed strings."""
    if old == new:
        raise _RefusedInput("old_string and new_string are identical; nothing would change.")
    if old not in text and "\r\n" in text:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    count = text.count(old)
    if count == 0:
        raise _RefusedInput(
            f"old_string was not found in {label}. Read the file again and copy the exact "
            "text, including indentation."
        )
    if count > 1 and not replace_all:
        raise _RefusedInput(
            f"old_string occurs {count} times in {label}. Include more surrounding lines "
            "to make it unique, or set replace_all."
        )
    return text.replace(old, new, -1 if replace_all else 1), count if replace_all else 1


def _read_text(target: Path) -> str:
    # newline="" keeps CRLF line endings exactly as they are on disk.
    with target.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_text(target: Path, content: str) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _error(reason: object) -> dict:
    return {"ok": False, "error": str(reason)}


def _integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: tuple[str, ...],
    run: Any,
) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        description=description,
        input_schema=object_schema(properties=properties, required=required),
        source=_SOURCE,
        run=run,
    )


__all__ = ["WorkspaceTools", "refused_git_verb"]
