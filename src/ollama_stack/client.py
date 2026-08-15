"""The single path to Ollama, so num_ctx and the prompt_eval_count check cannot be skipped."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from ollama_stack.models import DEFAULT_NUM_CTX, resolve

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 600


class OllamaError(RuntimeError):
    """Ollama returned something the caller cannot act on."""


class ContextTruncationError(OllamaError):
    """The prompt reached the usable window, so the front of it was silently dropped."""


@dataclass(frozen=True)
class Reply:
    """One response, carrying the counts that are the only honest signal about truncation."""

    text: str
    model: str
    num_ctx: int
    prompt_eval_count: int
    eval_count: int
    tool_calls: list[dict[str, Any]]

    @property
    def usable_window(self) -> int:
        """Roughly half of num_ctx, because generation needs room in the same budget."""
        return self.num_ctx // 2

    @property
    def counts_missing(self) -> bool:
        """No prompt_eval_count came back, so truncation can be neither shown nor ruled out."""
        return self.prompt_eval_count < 0

    @property
    def suspect_truncation(self) -> bool:
        """True once the prompt reached the window, and true when the count is missing entirely."""
        return self.counts_missing or self.prompt_eval_count >= self.usable_window


class OllamaClient:
    """Talks to a local Ollama, always with an explicit context window."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.2,
        timeout: int = DEFAULT_TIMEOUT,
        strict: bool = True,
    ) -> None:
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.strict = strict

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Ollama defaults to num_ctx 4096 whatever the model advertises, so it is always sent.
        payload["options"] = {"num_ctx": self.num_ctx, "temperature": self.temperature}
        payload["stream"] = False
        try:
            response = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"{path} failed against {self.host}: {exc}") from exc
        body: dict[str, Any] = response.json()
        return body

    def _reply(
        self,
        body: dict[str, Any],
        model: str,
        text: str,
        calls: list[dict[str, Any]],
    ) -> Reply:
        # Absent is -1, never 0: a count of zero would read as a small prompt that fit fine.
        raw = body.get("prompt_eval_count")
        reply = Reply(
            text=text,
            model=model,
            num_ctx=self.num_ctx,
            prompt_eval_count=int(raw) if raw is not None else -1,
            eval_count=int(body.get("eval_count", 0)),
            tool_calls=calls,
        )
        if self.strict and reply.counts_missing:
            raise OllamaError(
                "the response carried no prompt_eval_count, which is the only signal that says "
                "whether the prompt was truncated. Nothing can be concluded from this reply."
            )
        if self.strict and reply.suspect_truncation:
            raise ContextTruncationError(
                f"prompt_eval_count={reply.prompt_eval_count} reached the usable window "
                f"({reply.usable_window} of num_ctx={reply.num_ctx}). Ollama drops the FRONT of "
                "the prompt and the model answers confidently from what survived. Raise num_ctx "
                "or send less; do not trust this response."
            )
        return reply

    def generate(self, prompt: str, model: str, context: str = "") -> Reply:
        """One-shot completion."""
        spec = resolve(model)
        full = f"{context}\n\n{prompt}".strip() if context else prompt
        body = self._post("/api/generate", {"model": spec.tag, "prompt": full})
        return self._reply(body, spec.tag, str(body.get("response", "")), [])

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Reply:
        """Multi-turn exchange, with optional function calling."""
        spec = resolve(model)
        payload: dict[str, Any] = {"model": spec.tag, "messages": messages}
        if tools:
            payload["tools"] = tools
        body = self._post("/api/chat", payload)
        message: dict[str, Any] = body.get("message", {})
        calls: list[dict[str, Any]] = message.get("tool_calls") or []
        return self._reply(body, spec.tag, str(message.get("content", "")), calls)
