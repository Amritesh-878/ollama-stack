"""The tools the model may call. Each refuses rather than guessing when a match is ambiguous."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ollama_stack.binaries import on_path

# Never staged whatever the target repo ignores: __pycache__ broke a final `git add` once even
# with exclude pathspecs, so the exclusion lives here rather than in a .gitignore we do not own.
JUNK: tuple[str, ...] = (
    "__pycache__",
    ".pyc",
    ".pyo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".egg-info",
    ".venv/",
    ".git/",
)
READ_LIMIT = 200_000
GIT_TIMEOUT = 30
GATE_TIMEOUT = 900


class ToolError(RuntimeError):
    """The tool refused. The model reads this text and gets another turn to fix it."""


def numbered(body: str) -> str:
    """Measured: unnumbered input drifts 1 line at 42 and up to 15 at 109, because it counts."""
    rows = body.splitlines()
    width = len(str(len(rows)))
    return "\n".join(f"{n:>{width}}| {line}" for n, line in enumerate(rows, 1))


def as_list(value: Any) -> list[str]:
    """files_changed came back once as the string "['a.py']"; recover it, never iterate it."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list | tuple):
        return [str(item) for item in parsed]
    return [text]


# Files the gate or git executes before any test runs, so writing one is arbitrary code
# execution rather than an edit. pytest imports conftest.py; git reads .git/config every command.
EXECUTED_BY_TOOLING: tuple[str, ...] = (
    "conftest.py",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "noxfile.py",
    "tox.ini",
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    "pytest.ini",
    "sitecustomize.py",
    "usercustomize.py",
)


def executes(relative: str) -> bool:
    """True for a path whose contents run during the gate, or that git acts on."""
    # lstrip takes a character set, not a prefix: it would turn ".git/config" into "git/config".
    posix = relative.replace("\\", "/").removeprefix("./")
    parts = posix.split("/")
    if ".git" in parts:
        return True
    return parts[-1] in EXECUTED_BY_TOOLING


def is_junk(path: str) -> bool:
    """True for anything a build produced, so it never reaches the index or the handoff."""
    posix = path.replace("\\", "/")
    return any(marker.strip("/") in posix.split("/") or posix.endswith(marker)
               for marker in JUNK)


def git(repo: Path, *args: str) -> str:
    """Reads and writes the working tree; nothing here commits, merges or pushes."""
    binary = on_path("git")
    if binary is None:
        raise ToolError("git is not on PATH, so the repository cannot be inspected")
    try:
        done = subprocess.run(
            [binary, *args], cwd=repo, capture_output=True, text=True,
            timeout=GIT_TIMEOUT, check=True,
        )
    except FileNotFoundError as exc:
        raise ToolError("git is not on PATH, so the repository cannot be inspected") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git {' '.join(args)} timed out after {GIT_TIMEOUT}s") from exc
    except subprocess.CalledProcessError as exc:
        raise ToolError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return done.stdout


@dataclass(frozen=True)
class Gate:
    """One command in the target repo's quality pipeline, run in order and never reordered."""

    label: str
    command: tuple[str, ...]


# uv, never a bare tool name: on this machine bare ruff, mypy and pytest resolve outside the venv.
DEFAULT_GATE: tuple[Gate, ...] = (
    Gate("lint", ("uv", "run", "ruff", "check", "--fix", ".")),
    Gate("typecheck", ("uv", "run", "mypy", ".")),
    Gate("tests", ("uv", "run", "pytest")),
)


@dataclass(frozen=True)
class GateRun:
    """One command's real exit code and output, kept verbatim for the handoff."""

    label: str
    command: str
    code: int
    output: str


@dataclass(frozen=True)
class GateResult:
    """What the pipeline actually did, which is not what the model says it did."""

    runs: tuple[GateRun, ...]

    @property
    def passed(self) -> bool:
        """The exit code decides. A model summary claiming success cannot overrule this."""
        return bool(self.runs) and all(run.code == 0 for run in self.runs)

    @property
    def failures(self) -> list[str]:
        return [run.label for run in self.runs if run.code != 0]

    def transcript(self) -> str:
        return "\n\n".join(
            f"$ {run.command}\n{run.output.strip()}\n[exit {run.code}]" for run in self.runs
        )


def run_gate(repo: Path, gate: tuple[Gate, ...] = DEFAULT_GATE) -> GateResult:
    """Stops at the first failure: a test run over a red typecheck reports on the wrong code."""
    runs: list[GateRun] = []
    for step in gate:
        label = " ".join(step.command)
        # Resolved off PATH, never taken bare: the repo under test is the one directory that
        # must not get to supply the tool that judges it.
        binary = on_path(step.command[0])
        if binary is None:
            runs.append(GateRun(step.label, label, 127, f"{step.command[0]} is not on PATH"))
            break
        try:
            done = subprocess.run(
                [binary, *step.command[1:]], cwd=repo, capture_output=True, text=True,
                timeout=GATE_TIMEOUT, check=False,
            )
            output, code = (done.stdout + done.stderr), done.returncode
        except FileNotFoundError:
            output, code = f"{step.command[0]} is not on PATH", 127
        except subprocess.TimeoutExpired:
            output, code = f"timed out after {GATE_TIMEOUT}s", 124
        runs.append(GateRun(step.label, label, code, output))
        if code != 0:
            break
    return GateResult(tuple(runs))


class Files:
    """Bounded access: edits stay inside the repo, reads may also reach a read-only workspace."""

    def __init__(
        self, repo: Path, workspace: Path | None = None, read_limit: int = READ_LIMIT
    ) -> None:
        self.repo = repo.resolve()
        # Never inferred from a sibling directory: that assumption has broken three times.
        self.workspace = workspace.resolve() if workspace is not None else None
        self.read_limit = read_limit

    def _read_roots(self) -> tuple[Path, ...]:
        return (self.repo,) if self.workspace is None else (self.repo, self.workspace)

    def _resolve(self, raw: str, roots: tuple[Path, ...]) -> Path:
        given = Path(raw)
        inbounds: list[Path] = []
        for root in roots:
            candidate = (given if given.is_absolute() else root / given).resolve()
            if candidate == root or root in candidate.parents:
                # An existing file wins: the same relative path is in bounds under both roots.
                if candidate.exists():
                    return candidate
                inbounds.append(candidate)
        if inbounds:
            return inbounds[0]
        named = " or ".join(str(root) for root in roots)
        raise ToolError(f"{raw} resolves outside {named}, so it cannot be reached from here")

    def read(self, raw: str) -> str:
        path = self._resolve(raw, self._read_roots())
        if not path.is_file():
            extra = "" if self.workspace else " No workspace is mounted, so only the repo."
            raise ToolError(f"{raw} is not a readable file.{extra}")
        body = path.read_text(encoding="utf-8", errors="replace")
        if len(body) > self.read_limit:
            raise ToolError(
                f"{raw} is {len(body)} characters, over this run's {self.read_limit} limit, so "
                "nothing was read. A file this size would not fit the context window alongside "
                "the conversation. Ask for a smaller file, or raise --num-ctx."
            )
        return numbered(body)

    def _refuse_if_executed(self, raw: str, path: Path) -> None:
        """One guard for both writers: edit had none, and .git/config is always an existing file."""
        relative = str(path.relative_to(self.repo))
        if executes(relative):
            raise ToolError(
                f"{raw} is run by the toolchain rather than tested by it, so writing it would "
                "execute code outside this task. Refused. If the task genuinely needs this file "
                "changed, say so in your summary and stop."
            )
        if is_junk(relative):
            raise ToolError(f"{raw} looks like build output and will not be written")

    def edit(self, raw: str, old: str, new: str) -> str:
        """A non-unique match is an error, never a first-match guess: that edits the wrong line."""
        path = self._resolve(raw, (self.repo,))
        self._refuse_if_executed(raw, path)
        if not path.is_file():
            raise ToolError(f"{raw} does not exist. Use write_file to create it.")
        if not old:
            raise ToolError("old_string is empty. Give the exact text to replace.")
        body = path.read_text(encoding="utf-8")
        hits = body.count(old)
        if hits == 0:
            raise ToolError(
                f"old_string was not found in {raw}. The line-number prefixes in read_file "
                "output are not part of the file; strip them before copying."
            )
        if hits > 1:
            raise ToolError(
                f"old_string appears {hits} times in {raw}, so which one to edit is ambiguous "
                "and nothing was changed. Include more surrounding lines to make it unique."
            )
        path.write_text(body.replace(old, new, 1), encoding="utf-8")
        return f"edited {raw}: {len(old.splitlines())} lines became {len(new.splitlines())}"

    def create(self, raw: str, content: str) -> str:
        """Refuses a path that exists: the alternative is a whole-file rewrite by accident."""
        path = self._resolve(raw, (self.repo,))
        self._refuse_if_executed(raw, path)
        if path.exists():
            raise ToolError(f"{raw} already exists. Use edit_file; write_file is for new files.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"created {raw}, {len(content.splitlines())} lines"

    def listing(self, pattern: str = "") -> str:
        """git ls-files, so build output never reads as source the task forgot to mention."""
        args = ["ls-files"] + ([pattern] if pattern else [])
        found = [line for line in git(self.repo, *args).splitlines() if line and not is_junk(line)]
        if not found:
            return f"nothing tracked matches {pattern!r}" if pattern else "nothing is tracked here"
        return "\n".join(found)


SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read one file. Every line comes back prefixed with its number and a pipe, for "
                "citing lines. Those prefixes are NOT part of the file: strip them before "
                "copying any text into edit_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List tracked files, optionally filtered by a path glob.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Optional glob like `src/*.py`."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file. old_string must appear EXACTLY ONCE or the "
                "edit is refused and nothing changes. Include surrounding lines to make it "
                "unique. This is the only way to change an existing file: never regenerate one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text from the file, without line-number prefixes.",
                    },
                    "new_string": {"type": "string", "description": "What it becomes."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a file that does not exist yet. Refused if the path already exists; use "
                "edit_file for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_gate",
            "description": (
                "Run the repository's lint, typecheck and test pipeline and return the real "
                "output and exit codes. Run this before finishing. A failing gate is not done."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "End the run. Call this exactly once, after the gate passes. files_changed must "
                "list every file you actually edited and nothing else; it is checked against the "
                "real diff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What you changed and why."},
                    "files_changed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths you edited, relative to the repository.",
                    },
                },
                "required": ["summary", "files_changed"],
            },
        },
    },
]

TOOL_NAMES = tuple(str(entry["function"]["name"]) for entry in SCHEMA)
