"""Drives a local model against one finished task file, and never decides that it is done."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ollama_stack.client import OllamaClient, OllamaError, estimate_tokens
from ollama_stack.implement.tools import (
    DEFAULT_GATE,
    SCHEMA,
    Files,
    Gate,
    GateResult,
    ToolError,
    as_list,
    git,
    run_gate,
)

MAX_TURNS = 35
# Measured 2026-08-26: real source runs 2.84 chars per token, so a chars/4 estimate under-counts
# code by 41% and a conversation sized at the client's own budget would overflow num_ctx.
CODE_ESTIMATE_FACTOR = 1.41
# How many trailing messages the first compaction pass leaves alone, so context survives.
KEEP_RECENT = 6
# The share of the turn budget one file read may use, so a read cannot alone exceed it.
READ_SHARE = 0.6
DROPPED = "[dropped to stay inside the context window. Read the file again if you still need it.]"
NO_CALL_NUDGE = (
    "You made no tool call. Use the tools: read_file, list_files, edit_file, write_file, "
    "run_gate, finish. If the work is done, run_gate and then call finish."
)

SYSTEM = """You are implementing one task in a repository, as a Task Implementer.

PLATFORM: {system} {release}. The shell is {shell}. Unix commands such as ls, head, find and cat \
do not exist here, and you have no shell anyway: the tools below are the only way to touch \
anything.

Rules that matter more than finishing fast:

- Change only what the task file asks for. A change nobody asked for survives the whole test \
suite and is visible only in the diff.
- Use edit_file with an exact old_string. NEVER regenerate a whole file. Retyping a 900-line file \
to change five lines has silently lost content before, and the tests did not catch it.
- The line-number prefixes in read_file output are not part of the file. Strip them.
- write_file is only for a file that does not exist yet.
- Call run_gate before you finish. A gate that fails is not done, whatever your summary says.
- Call finish exactly once, and list in files_changed every file you actually edited. That list \
is compared against the real diff.
- You cannot commit, push, or merge. Do not try.
"""


@dataclass
class Outcome:
    """What the run did, kept separate from what the model said it did."""

    turns: int = 0
    finished: bool = False
    summary: str = ""
    reported: list[str] = field(default_factory=list)
    gate: GateResult | None = None
    notes: list[str] = field(default_factory=list)
    compactions: int = 0
    peak_prompt_tokens: int = 0
    gate_runs: int = 0
    num_ctx: int = 0
    budget: int = 0

    @property
    def passed(self) -> bool:
        """A run passes only when the gate did. The model's summary has no vote here."""
        return self.finished and self.gate is not None and self.gate.passed


def turn_budget(client: OllamaClient) -> int:
    """Stricter than the client's on two measured grounds, and the tighter one wins."""
    derated = int(client.prompt_budget / CODE_ESTIMATE_FACTOR)
    # Under num_ctx//2 the client's overflow check short-circuits, so a legitimate count in the
    # clamp band cannot be mistaken for a truncation across 35 turns.
    return max(1, min(derated, client.num_ctx // 2 - 1))


def read_limit(client: OllamaClient) -> int:
    """One file may use part of the turn budget, never all of it: the conversation needs room."""
    return int(turn_budget(client) * 4 * READ_SHARE)


def dirty(repo: Path) -> str:
    """Any output at all means stop: reported-versus-actual is unanswerable on a dirty tree."""
    return git(repo, "status", "--porcelain").strip()


def branch(repo: Path, name: str) -> None:
    git(repo, "checkout", "-b", name)


def _arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Arguments have arrived as a JSON string rather than an object before now, so both parse."""
    function = call.get("function")
    if not isinstance(function, dict):
        return "", {}
    name = str(function.get("name", ""))
    raw = function.get("arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return name, {}
    return name, raw if isinstance(raw, dict) else {}


class Harness:
    """One run: a bounded tool loop that stops at a handoff and never at an approval."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        files: Files,
        task: str,
        gate: tuple[Gate, ...] = DEFAULT_GATE,
        max_turns: int = MAX_TURNS,
    ) -> None:
        self._client = client
        self._model = model
        self._files = files
        self._task = task
        self._gate = gate
        self._max_turns = max_turns
        self.outcome = Outcome()

    @property
    def budget(self) -> int:
        return turn_budget(self._client)

    def system_prompt(self) -> str:
        return SYSTEM.format(
            system=platform.system() or "unknown",
            release=platform.release(),
            shell="PowerShell" if platform.system() == "Windows" else "sh",
        )

    def _estimate(self, messages: list[dict[str, Any]]) -> int:
        return estimate_tokens("\n".join(str(m.get("content", "")) for m in messages))

    def _compact(self, messages: list[dict[str, Any]]) -> int:
        """Drops old tool output first: it is re-readable, and the task file is the boundary."""
        dropped = 0
        # Older results first, then recent ones: sparing the newest read would let one call
        # exceed the budget on its own and make compaction a no-op.
        for limit in (max(2, len(messages) - KEEP_RECENT), max(2, len(messages) - 1)):
            for message in messages[2:limit]:
                if self._estimate(messages) <= self.budget:
                    return dropped
                if message.get("role") == "tool" and message.get("content") != DROPPED:
                    message["content"] = DROPPED
                    dropped += 1
        return dropped

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "read_file":
            return self._files.read(str(args.get("path", "")))
        if name == "list_files":
            return self._files.listing(str(args.get("pattern", "")))
        if name == "edit_file":
            return self._files.edit(
                str(args.get("path", "")),
                str(args.get("old_string", "")),
                str(args.get("new_string", "")),
            )
        if name == "write_file":
            return self._files.create(str(args.get("path", "")), str(args.get("content", "")))
        if name == "run_gate":
            result = run_gate(self._files.repo, self._gate)
            self.outcome.gate = result
            self.outcome.gate_runs += 1
            verdict = "PASSED" if result.passed else f"FAILED: {', '.join(result.failures)}"
            return f"{result.transcript()}\n\ngate {verdict}"
        raise ToolError(f"there is no tool named {name!r}")

    def _finish(self, args: dict[str, Any]) -> None:
        self.outcome.finished = True
        self.outcome.summary = str(args.get("summary", "")).strip()
        self.outcome.reported = as_list(args.get("files_changed"))
        if self.outcome.gate is None:
            self.outcome.notes.append(
                "the model finished without ever running the gate, so its summary rests on nothing"
            )

    def run(self) -> Outcome:
        self.outcome.num_ctx = self._client.num_ctx
        self.outcome.budget = self.budget
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self._task},
        ]
        nudged = False
        while self.outcome.turns < self._max_turns:
            self.outcome.turns += 1
            self.outcome.compactions += self._compact(messages)
            try:
                reply = self._client.chat(messages, self._model, tools=SCHEMA)
            except OllamaError as exc:
                self.outcome.notes.append(f"the run stopped on turn {self.outcome.turns}: {exc}")
                return self.outcome
            self.outcome.peak_prompt_tokens = max(
                self.outcome.peak_prompt_tokens, reply.prompt_eval_count
            )
            if not reply.tool_calls:
                if nudged:
                    self.outcome.notes.append(
                        "the model stopped making tool calls without calling finish"
                    )
                    return self.outcome
                nudged = True
                messages.append({"role": "assistant", "content": reply.text})
                messages.append({"role": "user", "content": NO_CALL_NUDGE})
                continue
            nudged = False
            messages.append(
                {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls}
            )
            for call in reply.tool_calls:
                name, args = _arguments(call)
                if name == "finish":
                    self._finish(args)
                    return self.outcome
                try:
                    result = self._dispatch(name, args)
                except ToolError as exc:
                    result = f"REFUSED: {exc}"
                messages.append({"role": "tool", "content": result})
        self.outcome.notes.append(
            f"the run hit its {self._max_turns}-turn limit without calling finish"
        )
        return self.outcome
