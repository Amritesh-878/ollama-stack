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
    "heavy": ModelSpec("qwen3.8:27b", "audits and long work; vision, 256K advertised", True),
    "qwen": ModelSpec("qwen3.8:27b", "the heavy model under its earlier alias", True),
    "coder": ModelSpec("qwen3-coder:30b", "agentic coding; the only measured implementer", True),
    "dev": ModelSpec("devstral:24b", "multi-file agentic work", False),
    "think": ModelSpec("deepseek-r1:32b", "open-ended reasoning, NOT defect hunting", True),
    "gem": ModelSpec("gemma4:26b", "vision, general chat", False),
    "deepseek": ModelSpec(
        "deepseek-coder-v2:16b-lite-instruct-q4_0", "quick code snippets", False
    ),
    "qwen36": ModelSpec("qwen3.6:27b", "previous daily driver, benchmarked against heavy", True),
}

# Everything `o setup` can recommend, so a machine smaller than this one never gets warned about
# the model the wizard itself picked for it. Keyed by tag; these are sizes, not roles.
TIER_MODELS: dict[str, str] = {
    "qwen3.5:0.8b": "1 GB - CPU-only and very small cards",
    "qwen3.5:2b": "2.7 GB - fast path on cards under 10 GB",
    "qwen3.5:9b": "6.6 GB - heavy on 10-16 GB cards",
    "qwen3.5:27b": "17 GB - heavy on 16-21 GB cards",
}


def known_tag(tag: str) -> bool:
    """True for anything routable without a warning: a registry entry or a tier recommendation."""
    return tag in TIER_MODELS or _by_tag(tag) is not None

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


def _described(tag: str) -> str:
    known = _by_tag(tag)
    if known is not None:
        return known.summary
    return TIER_MODELS.get(tag, "not in the registry")


def resolve(name: str) -> ModelSpec:
    """Turn an alias or a raw Ollama tag into a spec, preferring aliases."""
    if name in REGISTRY:
        spec = REGISTRY[name]
        tag = _ROLE_TAGS.get(name, spec.tag)
        if tag == spec.tag:
            return spec
        # Describe the role, not whatever other alias happens to share the tag: pointing `heavy`
        # at the fast model used to make it advertise itself as the hot path.
        target = _by_tag(tag)
        # Measurement belongs to the tag, never to the role that happens to point at it.
        measured = target.measured if target is not None else False
        return ModelSpec(tag, f"{spec.summary.split(';')[0]} (set by config)", measured)
    return _by_tag(name) or ModelSpec(name, _described(name), False)
