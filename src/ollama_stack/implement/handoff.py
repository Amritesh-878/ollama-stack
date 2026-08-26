"""The handoff a run writes: unaudited, with the real gate output and the real diff beside it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ollama_stack.implement.harness import Outcome
from ollama_stack.implement.tools import git, is_junk

BANNER = (
    "> **IMPLEMENTED BY A LOCAL MODEL. NOT AUDITED. DO NOT MERGE.**\n"
    "> Nobody has re-run the gate independently or read this diff. Both are required before\n"
    "> this goes anywhere. A handoff is not an approval."
)
# Deliberately plain: a 900-line diff for a five-line task is the failure the suite cannot see.
BIG_DIFF_SHARE = 0.5


@dataclass(frozen=True)
class FileChange:
    """One file's real churn, which is the only place a whole-file rewrite is visible."""

    path: str
    added: int
    removed: int
    total: int

    @property
    def rewritten(self) -> bool:
        return self.total > 0 and (self.added + self.removed) >= self.total * BIG_DIFF_SHARE


def _created(repo: Path) -> list[str]:
    """numstat is tracked-only, and nothing here ever stages, so a new file is invisible to it."""
    fresh: list[str] = []
    for line in git(repo, "status", "--porcelain").splitlines():
        if line[:2].strip() == "??" or line.startswith("A "):
            path = line[3:].strip().strip(chr(34))
            if path and not is_junk(path):
                fresh.append(path)
    return fresh


def changes(repo: Path) -> list[FileChange]:
    """Tracked edits plus untracked creations, because a reviewer needs to see both."""
    found: list[FileChange] = []
    for path in _created(repo):
        target = repo / path
        if not target.is_file():
            continue
        lines = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        found.append(FileChange(path, lines, 0, lines))
    for line in git(repo, "diff", "--numstat", "HEAD").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, path = parts
        target = repo / path
        total = (
            len(target.read_text(encoding="utf-8", errors="replace").splitlines())
            if target.is_file()
            else 0
        )
        found.append(
            FileChange(path, _count(added), _count(removed), total)
        )
    return found


def _count(raw: str) -> int:
    """A binary file's numstat column is a dash, not a number."""
    return int(raw) if raw.isdigit() else 0


def staged_junk(repo: Path) -> list[str]:
    return [
        line[3:].strip()
        for line in git(repo, "status", "--porcelain").splitlines()
        if line and is_junk(line[3:].strip())
    ]


def discrepancy(reported: list[str], actual: list[str]) -> list[str]:
    """Both directions: a fabricated entry and a silent extra edit are different failures."""
    said, did = {p.replace("\\", "/") for p in reported}, {p.replace("\\", "/") for p in actual}
    lines = [f"reported but not changed: `{path}`" for path in sorted(said - did)]
    lines += [f"changed but not reported: `{path}`" for path in sorted(did - said)]
    return lines


def render(outcome: Outcome, repo: Path, task: str, model: str, branch: str) -> str:
    """Every section a reader cannot reconstruct from the diff, and nothing they can."""
    found = changes(repo)
    actual = [change.path for change in found]
    gaps = discrepancy(outcome.reported, actual)
    junk = staged_junk(repo)
    verdict = "PASSED" if outcome.passed else "FAILED"

    lines = [
        f"# {task}: Handoff",
        "",
        BANNER,
        "",
        f"**Task:** `{task}`",
        f"**Branch:** `{branch}`",
        f"**Model:** `{model}`, {outcome.turns} turns",
        f"**Status:** {verdict}",
        "",
        "---",
        "",
        "## Quality gate",
        "",
    ]

    if outcome.gate is None:
        lines += [
            "**The gate never ran.** Nothing here has been checked, by anyone or anything.",
            "",
        ]
    else:
        lines += ["```", outcome.gate.transcript(), "```", ""]
        if outcome.gate.passed:
            lines.append("**Result:** every command exited 0.")
        else:
            lines.append(
                f"**Result: FAILED** on {', '.join(outcome.gate.failures)}. This verdict comes "
                "from the exit codes, not from the model's summary."
            )
        lines.append("")

    lines += ["---", "", "## What changed", ""]
    if not found:
        lines.append("Nothing. The working tree is identical to `HEAD`.")
    else:
        lines += ["| file | added | removed | of | rewritten? |", "| --- | --- | --- | --- | --- |"]
        for change in found:
            flag = "**YES**" if change.rewritten else "no"
            lines.append(
                f"| `{change.path}` | {change.added} | {change.removed} | {change.total} | {flag} |"
            )
        if any(change.rewritten for change in found):
            lines += [
                "",
                "**A file marked rewritten had at least half its lines touched.** Read that diff "
                "before anything else: regenerating a file to change a little of it loses content "
                "silently and the test suite does not notice.",
            ]
    lines.append("")

    lines += ["## What the model said it changed", ""]
    if outcome.reported:
        lines += [f"- `{path}`" for path in outcome.reported]
    else:
        lines.append("It reported no files at all.")
    lines.append("")
    if gaps:
        lines += ["**The report and the diff disagree:**", ""] + [f"- {gap}" for gap in gaps]
    else:
        lines.append("The report and `git diff --name-only` agree exactly.")
    lines.append("")

    lines += ["## Its summary, unedited", "", outcome.summary or "It gave none.", ""]

    lines += ["---", "", "## Warnings for whoever picks this up next", ""]
    warnings = list(outcome.notes)
    if junk:
        warnings.append(f"build output is sitting in the tree: {', '.join(junk)}")
    if not outcome.finished:
        warnings.append("the run never called finish, so it stopped rather than completed")
    if outcome.compactions:
        warnings.append(
            f"{outcome.compactions} old tool results were dropped to stay inside the context "
            "window, so the model was working from less than it had read"
        )
    warnings.append(
        "this was implemented by a local model and has not been audited. Re-run the gate "
        "yourself and read the diff before it goes anywhere."
    )
    lines += [f"- {warning}" for warning in warnings]
    lines += [
        "",
        f"Peak `prompt_eval_count` was {outcome.peak_prompt_tokens} at `num_ctx` "
        f"{outcome.num_ctx}, against a harness turn budget of {outcome.budget} estimated tokens. "
        f"{outcome.compactions} tool results were dropped to hold that line.",
        "",
    ]
    return "\n".join(lines)
