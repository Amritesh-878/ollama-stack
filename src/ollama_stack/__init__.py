"""Local-model toolchain built on one Ollama client."""

from ollama_stack.client import ContextTruncationError, OllamaClient, OllamaError, Reply
from ollama_stack.models import REGISTRY, ModelSpec, resolve

__version__ = "0.1.0"

__all__ = [
    "REGISTRY",
    "ContextTruncationError",
    "ModelSpec",
    "OllamaClient",
    "OllamaError",
    "Reply",
    "resolve",
]
