"""The tutorial is the first thing a stranger runs, so it must not hang, crash, or change state."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from ollama_stack import config, tutorial
from ollama_stack.tutorial import LESSONS, Lesson


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Stub the subprocess rather than `execute`, so the real one is still under test."""
    monkeypatch.setenv(config.PATH_ENV, str(tmp_path / "config.toml"))
    monkeypatch.setattr(
        "ollama_stack.tutorial.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0),
    )
    yield


def _answers(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> None:
    queue = list(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": queue.pop(0) if queue else "q")


def test_no_terminal_prints_the_steps_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Piping `o tutorial` must not block on input nobody can supply."""
    assert tutorial.run() == 0
    out = capsys.readouterr().out
    for lesson in LESSONS:
        assert lesson.shown in out


def test_there_are_nine_steps_and_they_start_at_status_and_end_at_stop() -> None:
    assert len(LESSONS) == 9
    assert LESSONS[0].argv == ["status"]
    assert LESSONS[-1].argv == ["stop"]


def test_it_ends_on_stop_so_the_last_thing_taught_is_giving_the_card_back() -> None:
    assert LESSONS[-1].key == "C9"
    assert "card is yours again" in LESSONS[-1].teaches


def test_the_search_step_says_plainly_that_a_sourced_answer_can_still_be_wrong() -> None:
    lesson = next(item for item in LESSONS if item.key == "C5")
    assert "can still be wrong" in lesson.teaches


def test_the_piping_demo_file_is_small_enough_to_survive_the_context_guard() -> None:
    """The refusal at the prompt budget is real, so this demo must stay well under it."""
    from ollama_stack.client import estimate_tokens, prompt_budget

    lesson = next(item for item in LESSONS if item.key == "C7")
    assert lesson.stdin is not None
    assert estimate_tokens(lesson.stdin) < prompt_budget(config.DEFAULTS["num_ctx"]) // 4


def test_the_launcher_prefers_the_o_on_path_so_the_run_proves_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ollama_stack.tutorial.on_path", lambda name, path=None: "/usr/local/bin/o"
    )
    assert tutorial.launcher() == ["/usr/local/bin/o"]


def test_the_launcher_falls_back_to_the_module_when_o_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ollama_stack.tutorial.on_path", lambda name, path=None: None)
    assert tutorial.launcher()[-2:] == ["-m", "ollama_stack"]


def test_quitting_part_way_exits_zero_and_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tutorial, "interactive", lambda: True)
    _answers(monkeypatch, ["q"])
    assert tutorial.run() == 0
    assert "picks up from the top" in capsys.readouterr().out


def test_every_step_can_be_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tutorial, "interactive", lambda: True)
    ran: list[str] = []

    def record(lesson: Lesson) -> bool:
        ran.append(lesson.key)
        return True

    monkeypatch.setattr(tutorial, "execute", record)
    _answers(monkeypatch, ["s"] * len(LESSONS))
    assert tutorial.run() == 0
    assert ran == []


def test_a_failing_step_explains_and_continues_rather_than_aborting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tutorial that only works on a perfect machine teaches the wrong thing."""
    monkeypatch.setattr(tutorial, "interactive", lambda: True)
    monkeypatch.setattr(tutorial, "execute", lambda lesson: False)
    _answers(monkeypatch, [""] * len(LESSONS))
    assert tutorial.run() == 0
    out = capsys.readouterr().out
    assert "o config set fast_model" in out
    assert LESSONS[-1].teaches in out


def test_it_changes_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tutorial, "interactive", lambda: True)
    _answers(monkeypatch, [""] * len(LESSONS))
    tutorial.run()
    assert not config.config_path().exists()


def test_an_unrunnable_command_is_caught_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("no such binary")

    monkeypatch.setattr("ollama_stack.tutorial.subprocess.run", boom)
    lesson = Lesson("CX", "o nope", ["nope"], "nothing")
    assert tutorial.execute(lesson) is False
    assert "could not run it" in capsys.readouterr().out


def test_the_piping_lesson_names_a_command_that_exists_on_this_platform() -> None:
    """It taught `type sample.py`, which is Windows-only and named a file nobody created."""
    lesson = next(step for step in tutorial.LESSONS if step.key == "C7")
    expected = "type" if sys.platform == "win32" else "cat"
    assert lesson.shown.startswith(f"{expected} ")
    assert lesson.creates is not None


def test_every_lesson_that_shows_a_file_also_creates_it() -> None:
    for lesson in tutorial.LESSONS:
        if "sample.py" in lesson.shown:
            assert lesson.creates is not None, lesson.key
            assert lesson.creates[0] == str(tutorial.SAMPLE_PATH)


def _swallow_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lesson pipes the sample in, so the launcher only has to read and exit."""
    reader = [sys.executable, "-c", "import sys; sys.stdin.read()"]
    monkeypatch.setattr(tutorial, "launcher", lambda: reader)


def test_running_the_lesson_writes_the_file_the_command_refers_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lesson = next(step for step in tutorial.LESSONS if step.key == "C7")
    target = tmp_path / "nested" / "sample.py"
    monkeypatch.setattr(
        tutorial, "LESSONS", (replace(lesson, creates=(str(target), tutorial.SAMPLE)),)
    )
    _swallow_stdin(monkeypatch)
    tutorial.execute(tutorial.LESSONS[0])
    written = target
    assert written.is_file()
    assert "def total" in written.read_text(encoding="utf-8")


def test_a_file_the_user_already_has_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Their sample.py is theirs, and a tutorial is not a reason to lose it."""
    lesson = next(step for step in tutorial.LESSONS if step.key == "C7")
    theirs = tmp_path / "sample.py"
    lesson = replace(lesson, creates=(str(theirs), tutorial.SAMPLE))
    mine = "# mine" + chr(10)
    theirs.write_text(mine, encoding="utf-8")
    _swallow_stdin(monkeypatch)
    tutorial.execute(lesson)
    assert theirs.read_text(encoding="utf-8") == mine


def test_the_pipe_command_is_right_on_both_platforms_not_just_this_one() -> None:
    """Asserting only this machine's answer made the test tautological on Windows."""
    assert tutorial.cat_for("win32") == "type"
    for platform in ("linux", "darwin", "freebsd"):
        assert tutorial.cat_for(platform) == "cat", platform
