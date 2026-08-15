"""The client's whole job is that num_ctx and the truncation check cannot be skipped."""

from __future__ import annotations

from typing import Any

import pytest
import responses

from ollama_stack.client import ContextTruncationError, OllamaClient, OllamaError

GENERATE = "http://localhost:11434/api/generate"
CHAT = "http://localhost:11434/api/chat"


def _body(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"response": "hi", "prompt_eval_count": 10, "eval_count": 5}
    base.update(over)
    return base


@responses.activate
def test_generate_always_sends_num_ctx() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    OllamaClient(num_ctx=32768).generate("p", "qwen")
    assert '"num_ctx": 32768' in str(responses.calls[0].request.body)


@responses.activate
def test_generate_disables_streaming_so_counts_come_back() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    OllamaClient().generate("p", "qwen")
    sent = str(responses.calls[0].request.body)
    assert '"stream": false' in sent


@responses.activate
def test_alias_resolves_to_the_real_tag() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    reply = OllamaClient().generate("p", "coder")
    assert reply.model == "qwen3-coder:30b"


@responses.activate
def test_a_prompt_reaching_the_usable_window_is_refused() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=2050), status=200)
    with pytest.raises(ContextTruncationError, match="drops the FRONT"):
        OllamaClient(num_ctx=4096).generate("p", "qwen")


@responses.activate
def test_a_prompt_under_the_window_passes() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=1500), status=200)
    reply = OllamaClient(num_ctx=4096).generate("p", "qwen")
    assert not reply.suspect_truncation


@responses.activate
def test_landing_exactly_on_the_window_is_refused_not_allowed() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=2048), status=200)
    with pytest.raises(ContextTruncationError):
        OllamaClient(num_ctx=4096).generate("p", "qwen")


@responses.activate
def test_non_strict_reports_truncation_instead_of_raising() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=9000), status=200)
    reply = OllamaClient(num_ctx=4096, strict=False).generate("p", "qwen")
    assert reply.suspect_truncation


@responses.activate
def test_a_missing_prompt_eval_count_is_refused_not_treated_as_a_small_prompt() -> None:
    responses.add(responses.POST, GENERATE, json={"response": "hi"}, status=200)
    with pytest.raises(OllamaError, match="no prompt_eval_count"):
        OllamaClient().generate("p", "qwen")


@responses.activate
def test_a_missing_count_is_suspect_even_when_strict_is_off() -> None:
    responses.add(responses.POST, GENERATE, json={"response": "hi"}, status=200)
    reply = OllamaClient(strict=False).generate("p", "qwen")
    assert reply.counts_missing
    assert reply.suspect_truncation, "unknown must never read as clean"


@responses.activate
def test_a_genuine_zero_count_is_distinguishable_from_a_missing_one() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=0), status=200)
    reply = OllamaClient(strict=False).generate("p", "qwen")
    assert not reply.counts_missing


@responses.activate
def test_chat_forwards_tools_and_returns_tool_calls() -> None:
    calls = [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
    payload = {"message": {"content": "", "tool_calls": calls}, "prompt_eval_count": 10}
    responses.add(responses.POST, CHAT, json=payload, status=200)
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    reply = OllamaClient().chat([{"role": "user", "content": "go"}], "coder", tools=tools)
    assert reply.tool_calls == calls
    assert '"tools"' in str(responses.calls[0].request.body)


@responses.activate
def test_a_transport_failure_is_wrapped_not_leaked() -> None:
    responses.add(responses.POST, GENERATE, status=500)
    with pytest.raises(OllamaError, match="failed against"):
        OllamaClient().generate("p", "qwen")
