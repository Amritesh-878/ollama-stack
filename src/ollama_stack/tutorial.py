"""The guided first run: every step runs for real, so it reports their numbers and never ours."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

SAMPLE = '''def total(items):
    """Sum the prices, skipping anything that has been refunded."""
    running = 0
    for item in items:
        if item.refunded:
            continue
        running += item.price
    return running
'''


@dataclass(frozen=True)
class Lesson:
    """One step: what the user would type, and what running it is supposed to teach."""

    key: str
    shown: str
    argv: list[str]
    teaches: str
    stdin: str | None = None


LESSONS: tuple[Lesson, ...] = (
    Lesson(
        "C1",
        "o status",
        ["status"],
        "Nothing is loaded yet, and that is the normal resting state. It also reads your card.",
    ),
    Lesson(
        "C2",
        "o start",
        ["start"],
        "That wait is the cold load, and it happens once. Every question after it is warm.",
    ),
    Lesson(
        "C3",
        "o what is 10+10",
        ["what", "is", "10+10"],
        "No quotes and no subcommand. Any words that are not flags are the question.",
    ),
    Lesson(
        "C4",
        "o --stats what is 10+10",
        ["--stats", "what", "is", "10+10"],
        "Same answer, plus where the time went. On a reasoning model most of it is thinking you "
        "never see.",
    ),
    Lesson(
        "C5",
        "o what is the current price of bitcoin",
        ["what", "is", "the", "current", "price", "of", "bitcoin"],
        "It searched on its own and cited sources. A sourced answer can still be wrong: measured "
        "here, the 4B model read five results and picked a content farm over python.org. Sources "
        "are something to check, not a guarantee.",
    ),
    Lesson(
        "C6",
        "o --think why is quicksort usually faster than mergesort",
        ["--think", "why", "is", "quicksort", "usually", "faster", "than", "mergesort"],
        "Thinking costs time before the first word and buys reasoning. It is off by default, and "
        "`o audit` is the one command that turns it on.",
    ),
    Lesson(
        "C7",
        "type sample.py | o explain this",
        ["explain", "this"],
        "Anything you pipe in becomes context for the question. This demo file is deliberately "
        "tiny - large files are still refused, and raising --num-ctx is the workaround.",
        SAMPLE,
    ),
    Lesson(
        "C8",
        "o config",
        ["config"],
        "Where your settings live, and which layer won. A flag beats an environment variable, "
        "which beats this file, which beats the built-in default.",
    ),
    Lesson(
        "C9",
        "o stop",
        ["stop"],
        "The card is yours again. This is the last thing to learn because it is the thing people "
        "forget.",
    ),
)


def launcher() -> list[str]:
    """Prefer the `o` on PATH, so the tutorial proves the install rather than bypassing it."""
    found = shutil.which("o")
    if found is not None:
        return [found]
    return [sys.executable, "-m", "ollama_stack"]


def say(message: str = "") -> None:
    print(message, flush=True)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def listing() -> int:
    """No terminal means print the nine steps and leave, rather than blocking on input."""
    say("o tutorial - the nine steps, printed because there is no terminal to run them in.")
    say()
    for lesson in LESSONS:
        say(f"  {lesson.key}  {lesson.shown}")
        say(f"      {lesson.teaches}")
        say()
    say("Run `o tutorial` in a terminal to step through these against your own machine.")
    return 0


def execute(lesson: Lesson) -> bool:
    """A step that fails is still a lesson, so this reports and the caller keeps going."""
    command = launcher() + lesson.argv
    try:
        done = subprocess.run(
            command,
            input=lesson.stdin,
            text=True,
            check=False,
        )
    except OSError as exc:
        say(f"  could not run it: {exc}")
        return False
    return done.returncode == 0


def prompt(lesson: Lesson, index: int, total: int) -> str:
    say()
    say(f"[{index}/{total}]  {lesson.shown}")
    try:
        answer = input("       Enter to run, s to skip, q to quit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        say()
        return "q"
    return answer or ""


def run() -> int:
    """Reads config and changes none of it; the only state it touches is C2's load and C9's stop."""
    if not interactive():
        return listing()
    say("o tutorial - nine steps, run for real against your machine.")
    say("Nothing here changes your settings. Quit at any point with q.")
    total = len(LESSONS)
    for index, lesson in enumerate(LESSONS, 1):
        answer = prompt(lesson, index, total)
        if answer.startswith("q"):
            say()
            say("Stopped. `o tutorial` picks up from the top whenever you want it.")
            return 0
        if answer.startswith("s"):
            continue
        say()
        if not execute(lesson) and lesson.key == "C2":
            say("  That model would not load. `o models` lists what else is available, and")
            say("  `o config set fast_model <tag>` points the fast role somewhere smaller.")
        say()
        say(f"  -> {lesson.teaches}")
    say()
    say("That is the whole tool. `o --help` lists everything else.")
    return 0
