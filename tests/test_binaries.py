"""A planted program is the one input a lookup cannot be allowed to trust."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ollama_stack.binaries import on_path

WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""


def _plant(directory: Path, name: str) -> Path:
    """A file the operating system would be willing to execute, so the test is not vacuous."""
    target = directory / f"{name}{EXE}"
    target.write_text("#!/bin/sh\necho planted\n", encoding="utf-8")
    target.chmod(0o755)
    return target


def test_a_program_in_the_working_directory_is_never_what_gets_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git clone x && cd x && o status` must not run x's own nvidia-smi."""
    real = tmp_path / "bin"
    real.mkdir()
    _plant(real, "toolname")
    work = tmp_path / "work"
    work.mkdir()
    _plant(work, "toolname")
    monkeypatch.chdir(work)
    monkeypatch.setenv("PATH", str(real))
    found = on_path("toolname")
    assert found is not None
    assert Path(found).parent == real


def test_the_planted_copy_hides_nothing_and_the_real_one_is_still_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing to run the planted one is only half of it: refusing to run at all is a break."""
    real = tmp_path / "bin"
    real.mkdir()
    _plant(real, "git")
    work = tmp_path / "work"
    work.mkdir()
    _plant(work, "git")
    monkeypatch.chdir(work)
    monkeypatch.setenv("PATH", str(real))
    assert on_path("git") is not None


def test_the_working_directory_is_dropped_even_when_path_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    _plant(work, "toolname")
    monkeypatch.chdir(work)
    monkeypatch.setenv("PATH", os.pathsep.join([str(work), os.curdir, ""]))
    assert on_path("toolname") is None


def test_a_relative_path_entry_is_not_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bin` on PATH means "wherever this happens to be run", which is the thing being stopped."""
    work = tmp_path / "work"
    (work / "bin").mkdir(parents=True)
    _plant(work / "bin", "toolname")
    monkeypatch.chdir(work)
    monkeypatch.setenv("PATH", "bin")
    assert on_path("toolname") is None


def test_what_comes_back_is_a_full_path_so_nothing_searches_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    found = on_path("git")
    if found is None:
        pytest.skip("git is not installed on this machine")
    assert Path(found).is_absolute()
    assert Path(found).is_file()


def test_it_agrees_with_the_standard_library_when_nothing_is_planted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hardening must not quietly change which program a normal machine resolves."""
    monkeypatch.chdir(tmp_path)
    for name in ("git", "python"):
        expected = shutil.which(name)
        found = on_path(name)
        if expected is None:
            continue
        assert found is not None, name
        assert Path(found).resolve() == Path(expected).resolve(), name


def test_a_name_carrying_a_separator_is_refused_unless_it_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sub/tool` is a program under the working directory wearing a different hat."""
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    _plant(work / "sub", "toolname")
    monkeypatch.chdir(work)
    assert on_path(f"sub/toolname{EXE}") is None


def test_an_absolute_name_is_taken_as_given_and_returned_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller naming a full path chose it deliberately, so PATH is not consulted at all."""
    real = tmp_path / "bin"
    real.mkdir()
    planted = _plant(real, "toolname")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    found = on_path(str(planted))
    assert found is not None
    assert Path(found).is_absolute()
    assert Path(found) == planted


def test_an_absolute_name_that_is_not_there_is_none_rather_than_a_guess(tmp_path: Path) -> None:
    assert on_path(str(tmp_path / "missing" / "toolname")) is None
