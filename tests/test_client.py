"""The client's whole job is that num_ctx and the truncation check cannot be skipped."""

from __future__ import annotations

import json
from typing import Any

import pytest
import responses

from ollama_stack.client import (
    ContextTruncationError,
    OllamaClient,
    OllamaError,
    estimate_tokens,
    usable_window,
)

GENERATE = "http://127.0.0.1:11434/api/generate"
CHAT = "http://127.0.0.1:11434/api/chat"


def _body(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"response": "hi", "prompt_eval_count": 10, "eval_count": 5}
    base.update(over)
    return base


def _ndjson(*chunks: dict[str, Any]) -> str:
    return "\n".join(json.dumps(chunk) for chunk in chunks)


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


def test_estimate_rounds_up_so_a_short_prompt_is_never_counted_as_nothing() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_the_window_rule_has_one_definition_shared_by_the_client_and_the_reply() -> None:
    assert usable_window(32768) == OllamaClient(num_ctx=32768).usable_window == 16384


def test_the_default_host_is_an_address_not_a_name() -> None:
    """Measured: resolving localhost costs ~2s per connection, which is most of a warm reply."""
    assert "localhost" not in OllamaClient().host


@responses.activate
def test_a_prompt_estimated_over_the_window_is_refused_before_anything_is_sent() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    with pytest.raises(ContextTruncationError, match="Nothing was sent"):
        OllamaClient(num_ctx=4096).generate("x" * 16384, "qwen")
    assert len(responses.calls) == 0


@responses.activate
def test_preflight_refuses_the_streaming_path_too() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    with pytest.raises(ContextTruncationError, match="Nothing was sent"):
        OllamaClient(num_ctx=4096).stream("x" * 16384, "qwen")
    assert len(responses.calls) == 0


@responses.activate
def test_preflight_counts_the_context_not_just_the_prompt() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    with pytest.raises(ContextTruncationError):
        OllamaClient(num_ctx=4096).generate("p", "qwen", context="x" * 16384)
    assert len(responses.calls) == 0


@responses.activate
def test_chat_preflights_on_the_whole_conversation() -> None:
    responses.add(responses.POST, CHAT, json=_body(), status=200)
    messages = [{"role": "user", "content": "x" * 16384}]
    with pytest.raises(ContextTruncationError):
        OllamaClient(num_ctx=4096).chat(messages, "qwen")
    assert len(responses.calls) == 0


@responses.activate
def test_a_non_strict_client_skips_the_preflight_as_well_as_the_post_hoc_check() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    OllamaClient(num_ctx=4096, strict=False).generate("x" * 16384, "qwen")
    assert len(responses.calls) == 1


@responses.activate
def test_stream_yields_chunks_in_order_and_captures_the_final_counts() -> None:
    body = _ndjson(
        {"response": "a", "done": False},
        {"response": "b", "done": False},
        {"response": "", "done": True, "prompt_eval_count": 12, "eval_count": 2},
    )
    responses.add(responses.POST, GENERATE, body=body, status=200)
    run = OllamaClient().stream("p", "qwen")
    assert list(run) == ["a", "b"]
    assert (run.reply.text, run.reply.prompt_eval_count, run.reply.eval_count) == ("ab", 12, 2)


@responses.activate
def test_stream_asks_for_streaming_where_generate_does_not() -> None:
    body = _ndjson({"response": "a", "done": True, "prompt_eval_count": 1})
    responses.add(responses.POST, GENERATE, body=body, status=200)
    list(OllamaClient().stream("p", "qwen"))
    assert '"stream": true' in str(responses.calls[0].request.body)


@responses.activate
def test_stream_yields_every_token_before_it_raises_on_the_final_counts() -> None:
    """The guard cannot fire earlier - the counts arrive last - so it must not lose the text."""
    body = _ndjson(
        {"response": "a", "done": False},
        {"response": "b", "done": True, "prompt_eval_count": 9000, "eval_count": 2},
    )
    responses.add(responses.POST, GENERATE, body=body, status=200)
    run = OllamaClient(num_ctx=4096).stream("p", "qwen")
    seen: list[str] = []
    with pytest.raises(ContextTruncationError, match="drops the FRONT"):
        for chunk in run:
            seen.append(chunk)
    assert seen == ["a", "b"]


@responses.activate
def test_a_stream_missing_its_counts_is_refused_like_a_whole_reply_would_be() -> None:
    responses.add(responses.POST, GENERATE, body=_ndjson({"response": "a", "done": True}))
    with pytest.raises(OllamaError, match="no prompt_eval_count"):
        list(OllamaClient().stream("p", "qwen"))


@responses.activate
def test_an_error_chunk_stops_the_stream_instead_of_reading_as_a_finished_reply() -> None:
    body = _ndjson({"response": "half", "done": False}, {"error": "llama runner exited"})
    responses.add(responses.POST, GENERATE, body=body, status=200)
    with pytest.raises(OllamaError, match="llama runner exited"):
        list(OllamaClient().stream("p", "qwen"))


@responses.activate
def test_a_stream_that_ends_without_a_final_chunk_is_an_error_not_a_short_answer() -> None:
    responses.add(responses.POST, GENERATE, body=_ndjson({"response": "a", "done": False}))
    with pytest.raises(OllamaError, match="no final chunk"):
        list(OllamaClient().stream("p", "qwen"))


@responses.activate
def test_a_chunk_that_is_not_json_is_reported_rather_than_crashing() -> None:
    responses.add(responses.POST, GENERATE, body="not json at all", status=200)
    with pytest.raises(OllamaError, match="not JSON"):
        list(OllamaClient().stream("p", "qwen"))


@responses.activate
def test_load_is_the_one_path_exempt_from_the_missing_count_check() -> None:
    """A load evaluates nothing, so demanding a count would make `o start` impossible."""
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "load"}, status=200)
    OllamaClient().load("qwen")
    assert '"keep_alive": -1' in str(responses.calls[0].request.body)


@responses.activate
def test_the_exemption_does_not_leak_into_generate() -> None:
    responses.add(responses.POST, GENERATE, json={"response": "hi"}, status=200)
    with pytest.raises(OllamaError, match="no prompt_eval_count"):
        OllamaClient().generate("p", "qwen")


@responses.activate
def test_the_exemption_does_not_leak_into_chat() -> None:
    responses.add(responses.POST, CHAT, json={"message": {"content": "hi"}}, status=200)
    with pytest.raises(OllamaError, match="no prompt_eval_count"):
        OllamaClient().chat([{"role": "user", "content": "go"}], "qwen")


@responses.activate
def test_a_pin_still_declares_num_ctx_so_it_is_not_left_to_ollamas_default() -> None:
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "load"}, status=200)
    OllamaClient(num_ctx=8192).load("qwen")
    assert '"num_ctx": 8192' in str(responses.calls[0].request.body)


@responses.activate
def test_a_load_that_reports_anything_but_a_load_is_refused() -> None:
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "stop"}, status=200)
    with pytest.raises(OllamaError, match="reported 'stop'"):
        OllamaClient().load("qwen")


@responses.activate
def test_unload_asks_ollama_to_release_the_model() -> None:
    responses.add(responses.POST, GENERATE, json={"done_reason": "unload"}, status=200)
    OllamaClient().unload("qwen3.8:27b")
    assert '"keep_alive": 0' in str(responses.calls[0].request.body)


@responses.activate
def test_a_two_hundred_that_is_not_json_is_reported_rather_than_crashing() -> None:
    responses.add(responses.POST, GENERATE, body="<html>proxy error</html>", status=200)
    with pytest.raises(OllamaError, match="not JSON"):
        OllamaClient().generate("p", "qwen")


@responses.activate
def test_ps_and_tags_return_an_empty_list_for_any_body_that_is_not_the_expected_shape() -> None:
    responses.add(responses.GET, "http://127.0.0.1:11434/api/ps", json={}, status=200)
    responses.add(responses.GET, "http://127.0.0.1:11434/api/tags", json={"models": 7}, status=200)
    client = OllamaClient()
    assert client.ps() == []
    assert client.tags() == []


@responses.activate
def test_asking_for_the_reply_before_the_stream_finishes_is_an_error() -> None:
    responses.add(responses.POST, GENERATE, body=_ndjson({"response": "a", "done": False}))
    run = OllamaClient().stream("p", "qwen")
    with pytest.raises(OllamaError, match="has not finished"):
        _ = run.reply
