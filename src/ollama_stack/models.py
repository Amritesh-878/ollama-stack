"""The routing table, kept in one place because model names have already drifted twice."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NUM_CTX = 32768


@dataclass(frozen=True)
class ModelSpec:
    """One routable model and what it is for."""

    tag: str
    summary: str
    measured: bool


REGISTRY: dict[str, ModelSpec] = {
    "fast": ModelSpec("qwen3.5:4b", "hot path: bare questions and `o start`", False),
    "heavy": ModelSpec("qwen3.8:27b", "audits and long work; vision, 256K advertised", False),
    "qwen": ModelSpec("qwen3.8:27b", "the heavy model under its earlier alias", False),
    "coder": ModelSpec("qwen3-coder:30b", "agentic coding; the only measured implementer", True),
    "dev": ModelSpec("devstral:24b", "multi-file agentic work", False),
    "think": ModelSpec("deepseek-r1:32b", "open-ended reasoning, NOT defect hunting", True),
    "gem": ModelSpec("gemma4:26b", "vision, general chat", False),
    "deepseek": ModelSpec(
        "deepseek-coder-v2:16b-lite-instruct-q4_0", "quick code snippets", False
    ),
    "qwen36": ModelSpec("qwen3.6:27b", "previous daily driver, kept for A/B", False),
}

FAST_ALIAS = "fast"
HEAVY_ALIAS = "heavy"
DEFAULT_ALIAS = FAST_ALIAS


# Config repoints a role here rather than editing REGISTRY, so the built-in stays visible.
_ROLE_TAGS: dict[str, str] = {}


def set_role_tag(alias: str, tag: str) -> None:
    """Point an alias at a different tag than the registry ships with."""
    if alias in REGISTRY and tag:
        _ROLE_TAGS[alias] = tag


def clear_role_tags() -> None:
    _ROLE_TAGS.clear()


def _by_tag(tag: str) -> ModelSpec | None:
    for spec in REGISTRY.values():
        if spec.tag == tag:
            return spec
    return None


def resolve(name: str) -> ModelSpec:
    """Turn an alias or a raw Ollama tag into a spec, preferring aliases."""
    if name in REGISTRY:
        spec = REGISTRY[name]
        tag = _ROLE_TAGS.get(name, spec.tag)
        if tag == spec.tag:
            return spec
        # A repointed role is unmeasured until someone measures the tag it now points at.
        return _by_tag(tag) or ModelSpec(tag, f"{name}, repointed by config", False)
    return _by_tag(name) or ModelSpec(name, "not in the registry", False)
