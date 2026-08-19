"""The single path to Ollama, so num_ctx and the prompt_eval_count check cannot be skipped."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ollama_stack.models import DEFAULT_NUM_CTX, resolve

if TYPE_CHECKING:
    import requests

# Not localhost: resolving it costs ~2s per connection where ollama binds IPv4 only.
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 600
PROBE_TIMEOUT = 10
PIN = -1
RELEASE = 0
MISSING_MODEL_RE = re.compile(r"model \"?'?([^'\"]+?)'?\"? not found")


class OllamaError(RuntimeError):
    """Ollama returned something the caller cannot act on."""


class ContextTruncationError(OllamaError):
    """The prompt would leave no room to answer into, or overflowed and lost its front."""


# Room left for the answer: the largest generation measured on this project is 12827 tokens, from
# an audit with thinking on, and 8192 covers an ordinary long reply without refusing valid prompts.
GENERATION_RESERVE = 8192
# Below this share of what we sent, the prompt was cut rather than merely mis-estimated.
TRUNCATION_RATIO = 0.6
# How far past num_ctx//2 the clamped count may land; measured at exactly +2 on both models.
CLAMP_SLACK = 8


def prompt_budget(num_ctx: int) -> int:
    """What the prompt may use: num_ctx minus room to generate into, never below half."""
    return max(num_ctx - GENERATION_RESERVE, num_ctx // 2)


def estimate_tokens(text: str) -> int:
    """Characters over four: under-counts code, over-counts prose, cheap enough to run first."""
    return -(-len(text) // 4)


def _joined(prompt: str, context: str) -> str:
    return f"{context}\n\n{prompt}".strip() if context else prompt


def _models(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Ollama owns this shape, so anything unexpected reads as an empty list, never a crash."""
    models = body.get("models")
    if not isinstance(models, list):
        return []
    return [entry for entry in models if isinstance(entry, dict)]


@dataclass(frozen=True)
class Reply:
    """One response, carrying the counts that are the only honest signal about truncation."""

    text: str
    model: str
    num_ctx: int
    prompt_eval_count: int
    eval_count: int
    tool_calls: list[dict[str, Any]]
    sent_estimate: int = 0
    done_reason: str = ""

    @property
    def ran_out_of_window(self) -> bool:
        """Generation hit num_ctx and stopped. Measured: gemma4:26b filled it with thinking."""
        return self.done_reason == "length"

    @property
    def prompt_budget(self) -> int:
        return prompt_budget(self.num_ctx)

    @property
    def counts_missing(self) -> bool:
        """No prompt_eval_count came back, so truncation can be neither shown nor ruled out."""
        return self.prompt_eval_count < 0

    @property
    def read_far_less_than_sent(self) -> bool:
        """Estimates run 0.72x to 1.32x of the real count, so only a gap outside that band tells."""
        return self.prompt_eval_count < self.sent_estimate * TRUNCATION_RATIO

    @property
    def clamped_to_half(self) -> bool:
        """Measured on both 27Bs: an overflowing prompt comes back at exactly num_ctx//2 + 2."""
        target = self.num_ctx // 2
        return target <= self.prompt_eval_count <= target + CLAMP_SLACK

    @property
    def suspect_truncation(self) -> bool:
        """Only real evidence of a cut: a send-time budget says nothing about what came back."""
        if self.counts_missing:
            return True
        # Ollama clamps to num_ctx//2, so a smaller send cannot have been cut at all.
        if self.sent_estimate < self.num_ctx // 2:
            return False
        return self.read_far_less_than_sent or self.clamped_to_half


class OllamaClient:
    """Talks to a local Ollama, always with an explicit context window."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.2,
        timeout: int = DEFAULT_TIMEOUT,
        strict: bool = True,
        think: bool = False,
    ) -> None:
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.strict = strict
        self.think = think
        self._sent_estimate = 0

    @property
    def prompt_budget(self) -> int:
        return prompt_budget(self.num_ctx)

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
        # Sent from here, never per call site: left off, a reasoning model triples the wait.
        payload["think"] = self.think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            response = requests.post(
                f"{self.host}{path}", json=payload, timeout=self.timeout, stream=stream
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise self._http_error(exc, path) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"{path} failed against {self.host}: {exc}") from exc
        return response

    def _http_error(self, exc: requests.HTTPError, path: str) -> OllamaError:
        """raise_for_status discards the body, and the body is where Ollama says what is wrong."""
        detail = ""
        if exc.response is not None:
            try:
                detail = str(exc.response.json().get("error", "")).strip()
            except ValueError:
                detail = ""
        if not detail:
            return OllamaError(f"{path} failed against {self.host}: {exc}")
        missing = MISSING_MODEL_RE.search(detail)
        if missing:
            return OllamaError(
                f"{detail}. Pull it first: `ollama pull {missing.group(1)}`, "
                "or run `o models` to see the aliases."
            )
        return OllamaError(f"{path} failed against {self.host}: {detail}")

    def _post(
        self, path: str, payload: dict[str, Any], *, keep_alive: int | None = None
    ) -> dict[str, Any]:
        response = self._request(path, payload, keep_alive=keep_alive)
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise OllamaError(f"{path} answered with something that is not JSON: {exc}") from exc
        # Ollama reports "does not support thinking" and friends as 200 plus this key, not a 4xx.
        if body.get("error"):
            raise OllamaError(f"ollama refused the request: {body['error']}")
        return body

    def _get(self, path: str) -> dict[str, Any]:
        import requests

        try:
            response = requests.get(f"{self.host}{path}", timeout=PROBE_TIMEOUT)
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"{path} failed against {self.host}: {exc}") from exc
        return body

    def preflight(self, text: str) -> None:
        """Refuses before sending, because a post-hoc check arrives after the answer is read."""
        estimate = estimate_tokens(text)
        # Kept even when not strict: it is what the reply's truncation check compares against.
        self._sent_estimate = estimate
        if not self.strict:
            return
        if estimate >= self.prompt_budget:
            raise ContextTruncationError(
                f"the prompt is estimated at {estimate} tokens, which reaches the prompt budget "
                f"({self.prompt_budget} of num_ctx={self.num_ctx}, leaving {GENERATION_RESERVE} "
                f"to answer into). Nothing was sent. Raise "
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
            done_reason=str(body.get("done_reason", "")),
            tool_calls=calls,
            sent_estimate=self._sent_estimate,
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
                f"prompt_eval_count={reply.prompt_eval_count} against a budget of "
                f"{reply.prompt_budget} and about {reply.sent_estimate} tokens sent, at "
                f"num_ctx={reply.num_ctx}. Ollama drops the FRONT of an oversized prompt and the "
                "model answers confidently from what survived. Raise num_ctx or send less; do "
                "not trust this response."
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
        started = time.perf_counter()
        response = self._request(
            "/api/generate", {"model": spec.tag, "prompt": full}, stream=True
        )
        return StreamRun(self, response, spec.tag, started)

    def load(self, model: str, keep_alive: int = PIN) -> None:
        """The one path exempt from the count check: a load evaluates nothing, so none exists."""
        spec = resolve(model)
        body = self._post(
            "/api/generate", {"model": spec.tag, "prompt": ""}, keep_alive=keep_alive
        )
        reason = body.get("done_reason")
        if reason is not None and reason != "load":
            raise OllamaError(f"asked {spec.tag} to load and it reported {reason!r} instead")

    def unload(self, model: str) -> None:
        """Asks Ollama to drop the model; whether it went is a question for /api/ps."""
        spec = resolve(model)
        self._post("/api/generate", {"model": spec.tag, "prompt": ""}, keep_alive=RELEASE)

    def ps(self) -> list[dict[str, Any]]:
        """What Ollama says is resident right now."""
        return _models(self._get("/api/ps"))

    def tags(self) -> list[dict[str, Any]]:
        """Every model pulled locally, resident or not."""
        return _models(self._get("/api/tags"))

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

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> StreamRun:
        """Streamed multi-turn, so a tool loop keeps the token-by-token output of the bare path."""
        spec = resolve(model)
        self.preflight("\n".join(str(m.get("content", "")) for m in messages))
        payload: dict[str, Any] = {"model": spec.tag, "messages": messages}
        if tools:
            payload["tools"] = tools
        started = time.perf_counter()
        response = self._request("/api/chat", payload, stream=True)
        return StreamRun(self, response, spec.tag, started)


class StreamRun:
    """One streamed reply: text arrives first and the counts last, so the guard fires last too."""

    def __init__(
        self,
        client: OllamaClient,
        response: requests.Response,
        model: str,
        started: float | None = None,
    ) -> None:
        self._client = client
        self._response = response
        self._model = model
        self._final: Reply | None = None
        self._started = time.perf_counter() if started is None else started
        self.first_chunk_seconds: float | None = None
        # A reasoning model emits these before a single visible word, which looks like a stall.
        self.thinking_tokens = 0
        self.tool_calls: list[dict[str, Any]] = []

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
                if self.first_chunk_seconds is None:
                    self.first_chunk_seconds = time.perf_counter() - self._started
                # /api/generate puts the token at the top level, /api/chat nests it in a message.
                raw = chunk.get("message")
                message: dict[str, Any] = raw if isinstance(raw, dict) else {}
                if chunk.get("thinking") or message.get("thinking"):
                    self.thinking_tokens += 1
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    self.tool_calls.extend(call for call in calls if isinstance(call, dict))
                piece = str(chunk.get("response", "") or message.get("content", ""))
                if piece:
                    text.append(piece)
                    yield piece
                if chunk.get("done"):
                    self._final = self._client._build(
                        chunk, self._model, "".join(text), self.tool_calls
                    )
        except requests.RequestException as exc:
            raise OllamaError(f"the stream broke against {self._client.host}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"ollama sent a chunk that is not JSON: {exc}") from exc
        if self._final is None:
            raise OllamaError("the stream ended with no final chunk, so no counts came back")
        self._client._guard(self._final)
