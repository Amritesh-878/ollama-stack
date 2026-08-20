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
    Reply,
    estimate_tokens,
    prompt_budget,
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
def test_a_prompt_clamped_to_half_the_window_is_refused() -> None:
    """Measured on both 27Bs: an overflowing prompt comes back at exactly num_ctx//2 + 2."""
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=16386), status=200)
    with pytest.raises(ContextTruncationError, match="drops the FRONT"):
        OllamaClient(num_ctx=32768).generate("x" * 80_000, "qwen")


@responses.activate
def test_a_prompt_under_the_window_passes() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=1500), status=200)
    reply = OllamaClient(num_ctx=4096).generate("p", "qwen")
    assert not reply.suspect_truncation


@responses.activate
def test_landing_exactly_on_the_clamp_point_is_refused_not_allowed() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=16384), status=200)
    with pytest.raises(ContextTruncationError):
        OllamaClient(num_ctx=32768).generate("x" * 80_000, "qwen")


@responses.activate
def test_non_strict_reports_truncation_instead_of_raising() -> None:
    responses.add(responses.POST, GENERATE, json=_body(prompt_eval_count=16386), status=200)
    reply = OllamaClient(num_ctx=32768, strict=False).generate("x" * 80_000, "qwen")
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
    assert prompt_budget(32768) == OllamaClient(num_ctx=32768).prompt_budget == 24576


@responses.activate
def test_every_call_declares_whether_to_think_rather_than_leaving_it_to_the_model() -> None:
    """Left unsent, qwen3.5:4b spends ~150 tokens reasoning about `what is 10+10`."""
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    OllamaClient().generate("p", "qwen")
    assert '"think": false' in str(responses.calls[0].request.body)


@responses.activate
def test_a_thinking_client_asks_for_it() -> None:
    responses.add(responses.POST, GENERATE, json=_body(), status=200)
    OllamaClient(think=True).generate("p", "qwen")
    assert '"think": true' in str(responses.calls[0].request.body)


@responses.activate
def test_an_error_key_on_a_two_hundred_is_raised_not_read_as_a_missing_count() -> None:
    body = {"error": '"deepseek-coder-v2" does not support thinking'}
    responses.add(responses.POST, GENERATE, json=body, status=200)
    with pytest.raises(OllamaError, match="does not support thinking"):
        OllamaClient(think=True).generate("p", "deepseek")


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
        {"response": "b", "done": True, "prompt_eval_count": 16386, "eval_count": 2},
    )
    responses.add(responses.POST, GENERATE, body=body, status=200)
    run = OllamaClient(num_ctx=32768).stream("x" * 80_000, "qwen")
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


@responses.activate
def test_a_model_that_is_not_pulled_says_how_to_pull_it() -> None:
    """The first thing a new user hits, and raise_for_status throws the explanation away."""
    responses.add(
        responses.POST, GENERATE, json={"error": "model 'qwen3.5:4b' not found"}, status=404
    )
    with pytest.raises(OllamaError, match=r"ollama pull qwen3\.5:4b"):
        OllamaClient().generate("hi", "fast")


@responses.activate
def test_another_http_error_still_surfaces_ollamas_own_words() -> None:
    responses.add(responses.POST, GENERATE, json={"error": "out of memory"}, status=500)
    with pytest.raises(OllamaError, match="out of memory"):
        OllamaClient().generate("hi", "fast")


@responses.activate
def test_an_http_error_with_no_usable_body_falls_back_to_the_status() -> None:
    responses.add(responses.POST, GENERATE, body="<html>gateway</html>", status=502)
    with pytest.raises(OllamaError, match="failed against"):
        OllamaClient().generate("hi", "fast")


def test_a_prompt_that_used_to_be_refused_is_now_accepted() -> None:
    """X6: measured, 29514 tokens process in full at 32768; refusing at 16384 cost capability."""
    client = OllamaClient(num_ctx=32768)
    client.preflight("x" * 80_000)


def test_the_budget_leaves_room_to_generate_into() -> None:
    from ollama_stack.client import GENERATION_RESERVE

    assert prompt_budget(32768) == 32768 - GENERATION_RESERVE


def test_a_small_window_never_yields_a_zero_or_negative_budget() -> None:
    """A fixed reserve subtracted from a small num_ctx would otherwise refuse everything."""
    for num_ctx in (1024, 2048, 4096, 8192, 16384):
        assert 0 < prompt_budget(num_ctx) <= num_ctx


def test_the_preflight_and_the_post_hoc_check_share_one_threshold() -> None:
    """Two thresholds that drift apart accept a prompt and then reject its own reply."""
    from ollama_stack.client import Reply

    client = OllamaClient(num_ctx=32768)
    reply = Reply("", "m", 32768, 1000, 0, [], sent_estimate=45000)
    assert reply.suspect_truncation
    assert client.prompt_budget == reply.prompt_budget


def test_overflow_is_detectable_from_the_counts_whatever_the_threshold_is() -> None:
    """Measured: a payload over num_ctx comes back at num_ctx//2+2, front gone."""
    from ollama_stack.client import Reply

    sent_estimate = 45000
    reply = Reply("", "m", 32768, 16386, 0, [], sent_estimate=45000)
    assert reply.prompt_eval_count < sent_estimate // 2
    assert reply.suspect_truncation


def test_reading_more_than_estimated_is_not_truncation() -> None:
    """Measured live: 84 KB of code estimated 21516 and read 29367, and nothing was cut."""
    reply = Reply("", "m", 32768, 29367, 40, [], sent_estimate=21516)
    assert not reply.suspect_truncation


def test_the_send_budget_protects_against_overflow_at_the_worst_measured_ratio() -> None:
    """chars/4 under-counts code by 24% (3.03 chars/token measured), and the budget absorbs it."""
    worst_case_actual = prompt_budget(32768) * (4 / 3.03)
    assert worst_case_actual < 32768


def test_a_reply_cut_off_at_the_window_is_visible_rather_than_silent() -> None:
    """gemma4:26b filled 32768 tokens with thinking twice and delivered one byte each time."""
    cut = Reply("x", "m", 32768, 10, 30584, [], done_reason="length")
    assert cut.ran_out_of_window
    assert not Reply("x", "m", 32768, 10, 40, [], done_reason="stop").ran_out_of_window


@responses.activate
def test_a_dead_runner_names_the_flash_attention_workaround() -> None:
    """Measured on Blackwell: 5 of 9 models die in the FA kernel after loading successfully."""
    body = {"error": "llama runner process has terminated: exit status 0xc0000409"}
    responses.add(responses.POST, GENERATE, json=body, status=500)
    with pytest.raises(OllamaError, match="OLLAMA_FLASH_ATTENTION"):
        OllamaClient().generate("hi", "fast")
