"""The gate needs something real to run from the first commit, or a green CI proves nothing."""

from __future__ import annotations

import ollama_stack


def test_the_package_imports() -> None:
    assert ollama_stack.__version__ == "0.1.0"
