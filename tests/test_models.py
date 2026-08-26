"""The registry is the one place model names live, because they have already drifted twice."""

from __future__ import annotations

from ollama_stack import models
from ollama_stack.models import (
    DEFAULT_ALIAS,
    FAST_ALIAS,
    HEAVY_ALIAS,
    REGISTRY,
    resolve,
)


def test_the_default_alias_exists_in_the_registry() -> None:
    assert DEFAULT_ALIAS in REGISTRY


def test_both_roles_exist_in_the_registry() -> None:
    assert {FAST_ALIAS, HEAVY_ALIAS} <= set(REGISTRY)


def test_a_bare_question_goes_to_the_small_model() -> None:
    """The whole point of the fast role: the always-warm model is 4 GiB, not 16."""
    assert DEFAULT_ALIAS == FAST_ALIAS
    assert REGISTRY[DEFAULT_ALIAS].tag == "qwen3.5:4b"


def test_the_heavy_role_is_the_27b() -> None:
    assert REGISTRY[HEAVY_ALIAS].tag == "qwen3.8:27b"


def test_the_two_roles_are_different_models() -> None:
    assert REGISTRY[FAST_ALIAS].tag != REGISTRY[HEAVY_ALIAS].tag


def test_the_earlier_qwen_alias_still_reaches_the_heavy_model() -> None:
    assert resolve("qwen").tag == REGISTRY[HEAVY_ALIAS].tag


def test_an_alias_resolves_to_its_tag() -> None:
    assert resolve("coder").tag == "qwen3-coder:30b"


def test_a_raw_tag_resolves_to_its_registry_entry() -> None:
    assert resolve("qwen3-coder:30b").measured


def test_an_unknown_tag_is_passed_through_as_unmeasured() -> None:
    spec = resolve("llama9:70b")
    assert spec.tag == "llama9:70b"
    assert not spec.measured


def test_only_the_heavy_role_is_flagged_measured() -> None:
    """Only heavy has a quality benchmark behind it; the fast role has latency figures."""
    assert REGISTRY[HEAVY_ALIAS].measured
    assert not REGISTRY[FAST_ALIAS].measured


def test_a_repointed_role_describes_the_role_not_another_alias() -> None:
    """Pointing heavy at the fast model made `o models` advertise heavy as the hot path."""
    models.set_role_tag(HEAVY_ALIAS, "qwen3.5:4b")
    try:
        spec = models.resolve(HEAVY_ALIAS)
        assert spec.tag == "qwen3.5:4b"
        assert "hot path" not in spec.summary
        assert "audits" in spec.summary
    finally:
        models.clear_role_tags()


def test_a_tier_recommendation_resolves_with_a_real_description() -> None:
    spec = models.resolve("qwen3.5:2b")
    assert spec.tag == "qwen3.5:2b"
    assert "not in the registry" not in spec.summary
