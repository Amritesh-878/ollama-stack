"""Piping into the front door must not agree to fetching and running someone else's code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> Any:
    """Loaded by path: it lives at the repo root and is deliberately not part of the package."""
    spec = importlib.util.spec_from_file_location("_bootstrap_under_test", ROOT / "bootstrap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_terminal(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(module.sys, "stdin", type("S", (), {"isatty": staticmethod(lambda: False)}))


def test_no_terminal_never_agrees_to_an_install_on_the_users_behalf(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It answered yes and then ran a remote installer that nobody had consented to."""
    module = _bootstrap()
    _no_terminal(monkeypatch, module)
    assert module.ask("Install uv now?", installs=True) is False
    assert "--yes" in capsys.readouterr().out


def test_yes_is_how_that_consent_gets_given(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _bootstrap()
    _no_terminal(monkeypatch, module)
    monkeypatch.setattr(module, "ASSUME_YES", True)
    assert module.ask("Install uv now?", installs=True) is True


def test_a_harmless_question_still_takes_its_default_with_no_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the questions that install something are gated; the rest must stay scriptable."""
    module = _bootstrap()
    _no_terminal(monkeypatch, module)
    assert module.ask("Carry on?") is True
    assert module.ask("Carry on?", False) is False


def test_the_install_questions_are_the_ones_marked_as_installing() -> None:
    """A new install prompt added without the flag would silently reopen the hole."""
    source = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    for question in ('ask("Install uv now?"', 'ask(f"Let uv install Python {wanted}?"'):
        index = source.index(question)
        tail = source[index : source.index(")", index + len(question)) + 1]
        assert "installs=True" in tail, question


def test_bootstrap_uses_only_the_standard_library() -> None:
    """It runs before uv and before the package, so a third-party import is a broken front door."""
    import ast

    tree = ast.parse((ROOT / "bootstrap.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported - sys.stdlib_module_names - {"__future__"}
