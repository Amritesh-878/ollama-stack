"""The registry is the one place model names live, because they have already drifted twice."""

from __future__ import annotations

from ollama_stack.models import DEFAULT_ALIAS, REGISTRY, resolve


def test_the_default_alias_exists_in_the_registry() -> None:
    assert DEFAULT_ALIAS in REGISTRY


def test_the_primary_is_qwen38() -> None:
    assert REGISTRY[DEFAULT_ALIAS].tag == "qwen3.8:27b"


def test_an_alias_resolves_to_its_tag() -> None:
    assert resolve("coder").tag == "qwen3-coder:30b"


def test_a_raw_tag_resolves_to_its_registry_entry() -> None:
    assert resolve("qwen3-coder:30b").measured


def test_an_unknown_tag_is_passed_through_as_unmeasured() -> None:
    spec = resolve("llama9:70b")
    assert spec.tag == "llama9:70b"
    assert not spec.measured


def test_the_primary_is_still_flagged_unmeasured() -> None:
    assert not REGISTRY["qwen"].measured, "flip this only when BRAINSTORM 2c/2d re-run on qwen3.8"
