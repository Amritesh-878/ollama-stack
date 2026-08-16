"""The single path to Ollama, so num_ctx and the prompt_eval_count check cannot be skipped."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ollama_stack.models import DEFAULT_NUM_CTX, resolve

if TYPE_CHECKING:
    import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 600
PROBE_TIMEOUT = 10


class OllamaError(RuntimeError):
    """Ollama returned something the caller cannot act on."""


class ContextTruncationError(OllamaError):
    """The prompt reached the usable window, so the front of it was silently dropped."""


def usable_window(num_ctx: int) -> int:
    """Roughly half of num_ctx, because generation needs room in the same budget."""
    return num_ctx // 2


def estimate_tokens(text: str) -> int:
    """Characters over four: under-counts code, over-counts prose, cheap enough to run first."""
    return -(-len(text) // 4)


def _joined(prompt: str, context: str) -> str:
    return f"{context}\n\n{prompt}".strip() if context else prompt


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
        return usable_window(self.num_ctx)

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

    @property
    def usable_window(self) -> int:
        return usable_window(self.num_ctx)

    def _request(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        keep_alive: int | None = None,
        stream: bool = False,
    ) -> requests.Response:
        # Imported here, not at module scope, so parsing argv never pays for it.
        import requests

        # Ollama defaults to num_ctx 4096 whatever the model advertises, so it is always sent.
        payload["options"] = {"num_ctx": self.num_ctx, "temperature": self.temperature}
        payload["stream"] = stream
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            response = requests.post(
                f"{self.host}{path}", json=payload, timeout=self.timeout, stream=stream
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"{path} failed against {self.host}: {exc}") from exc
        return response

    def _post(
        self, path: str, payload: dict[str, Any], *, keep_alive: int | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = self._request(path, payload, keep_alive=keep_alive).json()
        return body

    def _get(self, path: str) -> dict[str, Any]:
        import requests

        try:
            response = requests.get(f"{self.host}{path}", timeout=PROBE_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"{path} failed against {self.host}: {exc}") from exc
        body: dict[str, Any] = response.json()
        return body

    def preflight(self, text: str) -> None:
        """Refuses before sending, because a post-hoc check arrives after the answer is read."""
        if not self.strict:
            return
        estimate = estimate_tokens(text)
        if estimate >= self.usable_window:
            raise ContextTruncationError(
                f"the prompt is estimated at {estimate} tokens, which reaches the usable window "
                f"({self.usable_window} of num_ctx={self.num_ctx}). Nothing was sent. Raise "
                "--num-ctx or send less. The estimate is characters over four and is not exact."
            )

    def _build(
        self,
        body: dict[str, Any],
        model: str,
        text: str,
        calls: list[dict[str, Any]],
    ) -> Reply:
        # Absent is -1, never 0: a count of zero would read as a small prompt that fit fine.
        raw = body.get("prompt_eval_count")
        return Reply(
            text=text,
            model=model,
            num_ctx=self.num_ctx,
            prompt_eval_count=int(raw) if raw is not None else -1,
            eval_count=int(body.get("eval_count", 0)),
            tool_calls=calls,
        )

    def _guard(self, reply: Reply) -> None:
        if not self.strict:
            return
        if reply.counts_missing:
            raise OllamaError(
                "the response carried no prompt_eval_count, which is the only signal that says "
                "whether the prompt was truncated. Nothing can be concluded from this reply."
            )
        if reply.suspect_truncation:
            raise ContextTruncationError(
                f"prompt_eval_count={reply.prompt_eval_count} reached the usable window "
                f"({reply.usable_window} of num_ctx={reply.num_ctx}). Ollama drops the FRONT of "
                "the prompt and the model answers confidently from what survived. Raise num_ctx "
                "or send less; do not trust this response."
            )

    def _checked_reply(
        self,
        body: dict[str, Any],
        model: str,
        text: str,
        calls: list[dict[str, Any]],
    ) -> Reply:
        reply = self._build(body, model, text, calls)
        self._guard(reply)
        return reply

    def generate(self, prompt: str, model: str, context: str = "") -> Reply:
        """One-shot completion."""
        spec = resolve(model)
        full = _joined(prompt, context)
        self.preflight(full)
        body = self._post("/api/generate", {"model": spec.tag, "prompt": full})
        return self._checked_reply(body, spec.tag, str(body.get("response", "")), [])

    def stream(self, prompt: str, model: str, context: str = "") -> StreamRun:
        """Same completion, delivered as it is produced."""
        spec = resolve(model)
        full = _joined(prompt, context)
        self.preflight(full)
        response = self._request(
            "/api/generate", {"model": spec.tag, "prompt": full}, stream=True
        )
        return StreamRun(self, response, spec.tag)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Reply:
        """Multi-turn exchange, with optional function calling."""
        spec = resolve(model)
        self.preflight("\n".join(str(m.get("content", "")) for m in messages))
        payload: dict[str, Any] = {"model": spec.tag, "messages": messages}
        if tools:
            payload["tools"] = tools
        body = self._post("/api/chat", payload)
        message: dict[str, Any] = body.get("message", {})
        calls: list[dict[str, Any]] = message.get("tool_calls") or []
        return self._checked_reply(body, spec.tag, str(message.get("content", "")), calls)


class StreamRun:
    """One streamed reply: text arrives first and the counts last, so the guard fires last too."""

    def __init__(self, client: OllamaClient, response: requests.Response, model: str) -> None:
        self._client = client
        self._response = response
        self._model = model
        self._final: Reply | None = None

    @property
    def reply(self) -> Reply:
        if self._final is None:
            raise OllamaError("the stream has not finished, so no counts exist yet")
        return self._final

    def __iter__(self) -> Iterator[str]:
        import json

        import requests

        text: list[str] = []
        try:
            for line in self._response.iter_lines():
                if not line:
                    continue
                chunk: dict[str, Any] = json.loads(line)
                if chunk.get("error"):
                    raise OllamaError(f"ollama failed mid-stream: {chunk['error']}")
                piece = str(chunk.get("response", ""))
                if piece:
                    text.append(piece)
                    yield piece
                if chunk.get("done"):
                    self._final = self._client._build(chunk, self._model, "".join(text), [])
        except requests.RequestException as exc:
            raise OllamaError(f"the stream broke against {self._client.host}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"ollama sent a chunk that is not JSON: {exc}") from exc
        if self._final is None:
            raise OllamaError("the stream ended with no final chunk, so no counts came back")
        self._client._guard(self._final)
