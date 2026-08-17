"""Bare invocation is the whole product, so dispatch and streaming are tested at the CLI edge."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import responses

from ollama_stack.__main__ import RESERVED, _parse, _utf8, main
from ollama_stack.cli import app
from ollama_stack.models import FAST_ALIAS, HEAVY_ALIAS, REGISTRY

GENERATE = "http://127.0.0.1:11434/api/generate"
CHAT = "http://127.0.0.1:11434/api/chat"


def _ndjson(*chunks: dict[str, Any]) -> str:
    return "\n".join(json.dumps(chunk) for chunk in chunks)


def _chat_body(*words: str) -> str:
    """The search path talks to /api/chat, which nests the token inside a message."""
    parts: list[dict[str, Any]] = [{"message": {"content": w}, "done": False} for w in words]
    parts.append(
        {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": len(words),
        }
    )
    return _ndjson(*parts)


def _sent(index: int = 0) -> dict[str, Any]:
    raw = responses.calls[index].request.body
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    body: dict[str, Any] = json.loads(text)
    return body


def _stream_body(*words: str, prompt_eval_count: int = 10) -> str:
    parts: list[dict[str, Any]] = [{"response": word, "done": False} for word in words]
    parts.append(
        {
            "response": "",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": prompt_eval_count,
            "eval_count": len(words),
        }
    )
    return _ndjson(*parts)


def test_a_bare_question_becomes_the_prompt() -> None:
    opts, words = _parse(["what", "is", "10+10"])
    assert words == ["what", "is", "10+10"]
    assert opts.model == FAST_ALIAS


def test_a_bare_question_does_not_think_unless_asked() -> None:
    """Measured: thinking costs qwen3.5:4b ~1.4s before it shows a single word."""
    assert _parse(["what", "is", "10+10"])[0].think is False
    assert _parse(["--think", "why"])[0].think is True
    assert _parse(["--think", "--no-think", "why"])[0].think is False


def test_flags_before_the_prompt_are_not_prompt_text() -> None:
    opts, words = _parse(["-m", "coder", "what", "is", "10+10"])
    assert opts.model == "coder"
    assert words == ["what", "is", "10+10"]


def test_flags_after_the_prompt_parse_the_same_way() -> None:
    opts, words = _parse(["what", "is", "10+10", "--num-ctx", "8192", "--stats"])
    assert (opts.num_ctx, opts.stats) == (8192, True)
    assert words == ["what", "is", "10+10"]


def test_an_unrecognised_leading_token_is_prompt_text_not_an_error() -> None:
    _, words = _parse(["why", "-not-", "this"])
    assert words == ["why", "-not-", "this"]


def test_a_flag_missing_its_value_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-m"]) == 2
    assert "wants a value" in capsys.readouterr().err


def test_num_ctx_must_be_a_positive_number(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--num-ctx", "lots", "hello"]) == 2
    assert "positive number" in capsys.readouterr().err


def test_no_arguments_prints_usage_and_exits_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    assert "usage: o <question>" in capsys.readouterr().err


def test_version_is_printed_without_dispatching(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_a_reserved_first_word_runs_the_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["models"])
    assert "qwen3.8:27b" in capsys.readouterr().out


def test_every_subcommand_is_reserved_so_none_can_be_eaten_as_a_prompt() -> None:
    """A command typer knows about but the dispatcher does not would silently become a question."""
    from typer.main import get_group

    registered = set(get_group(app).commands)
    assert registered <= RESERVED
    assert {"setup", "tutorial"} <= registered
    assert RESERVED - registered == {"implement"}


def test_help_documents_the_reserved_word_rule_in_actionable_words(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "runs that command instead" in out
    assert 'o ask "status of the economy"' in out


@responses.activate
def test_streaming_writes_chunks_in_order_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("10", " + ", "10", " is 20"))
    assert main(["--no-web", "what", "is", "10+10"]) == 0
    assert capsys.readouterr().out == "10 + 10 is 20\n"


@responses.activate
def test_streaming_asks_ollama_to_stream(capsys: pytest.CaptureFixture[str]) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("hi"))
    main(["--no-web", "hello"])
    assert _sent()["stream"] is True


@responses.activate
def test_stdout_carries_only_the_answer_and_the_stats_line_goes_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("42"))
    main(["--no-web", "the", "answer"])
    captured = capsys.readouterr()
    assert captured.out == "42\n"
    assert "prompt 10 tok read" in captured.err


@responses.activate
def test_no_stream_sends_one_request_that_is_not_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = {"response": "42", "prompt_eval_count": 10, "eval_count": 1}
    responses.add(responses.POST, GENERATE, json=body)
    assert main(["--no-web", "--no-stream", "the", "answer"]) == 0
    assert _sent()["stream"] is False
    assert capsys.readouterr().out == "42\n"


@responses.activate
def test_stats_adds_the_estimate_and_timing_to_the_stderr_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("42"))
    main(["--no-web", "--stats", "the", "answer"])
    err = capsys.readouterr().err
    assert "estimated" in err
    assert "tok/s" in err


@responses.activate
def test_stats_times_the_first_chunk_not_the_whole_reply(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("4", "2"))
    main(["--no-web", "--stats", "the", "answer"])
    err = capsys.readouterr().err
    assert "first chunk" in err
    assert "first word" in err


@responses.activate
def test_thinking_tokens_are_counted_so_the_silence_before_an_answer_is_explained(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """qwen3.8 emits ~21 thinking tokens first; without this the wait looks like a stall."""
    body = _ndjson(
        {"response": "", "thinking": "The", "done": False},
        {"response": "", "thinking": " user", "done": False},
        {"response": "20", "done": False},
        {"response": "", "done": True, "prompt_eval_count": 10, "eval_count": 3},
    )
    responses.add(responses.POST, GENERATE, body=body)
    main(["--no-web", "--stats", "the", "answer"])
    captured = capsys.readouterr()
    assert captured.out == "20\n"
    assert "thought 2 tok first" in captured.err


@responses.activate
def test_a_reply_that_never_streamed_reports_no_first_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--no-stream has no first chunk to time, and a fabricated figure would be worse than none."""
    body = {"response": "42", "prompt_eval_count": 10, "eval_count": 1}
    responses.add(responses.POST, GENERATE, json=body)
    main(["--no-web", "--no-stream", "--stats", "the", "answer"])
    assert "first" not in capsys.readouterr().err


def test_a_cp1252_stream_is_reconfigured_to_survive_what_a_model_writes() -> None:
    """Measured: an arrow in an answer raised UnicodeEncodeError and truncated the reply."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    _utf8(stream)
    stream.write("10 → 20 — done ✓")
    stream.flush()
    assert stream.encoding == "utf-8"


def test_a_stream_that_cannot_be_reconfigured_is_left_alone() -> None:
    _utf8(object())


@responses.activate
def test_the_bare_path_asks_ollama_not_to_think(capsys: pytest.CaptureFixture[str]) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("20"))
    main(["--no-web", "what", "is", "10+10"])
    assert _sent()["think"] is False


@responses.activate
def test_think_turns_it_back_on(capsys: pytest.CaptureFixture[str]) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("20"))
    main(["--no-web", "--think", "why", "is", "the", "sky", "blue"])
    assert _sent()["think"] is True


@responses.activate
def test_a_bare_question_goes_to_the_small_model(capsys: pytest.CaptureFixture[str]) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("20"))
    main(["--no-web", "what", "is", "10+10"])
    assert _sent()["model"] == REGISTRY[FAST_ALIAS].tag


@responses.activate
def test_audit_goes_to_the_heavy_model_and_does_think(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Screening is multi-step, so it keeps the reasoning the hot path drops."""
    target = tmp_path / "sample.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    responses.add(responses.POST, GENERATE, body=_stream_body("nothing found"))
    with pytest.raises(SystemExit):
        main(["audit", str(target)])
    sent = _sent()
    assert sent["model"] == REGISTRY[HEAVY_ALIAS].tag
    assert sent["think"] is True


@responses.activate
def test_a_two_hundred_carrying_an_error_is_surfaced_not_read_as_a_missing_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ollama reports "does not support thinking" as 200 plus an error key."""
    body = {"error": '"deepseek" does not support thinking'}
    responses.add(responses.POST, GENERATE, json=body, status=200)
    assert main(["--no-web", "--no-stream", "--think", "-m", "deepseek", "hello"]) == 1
    assert "does not support thinking" in capsys.readouterr().err


@responses.activate
def test_a_preflight_refusal_sends_no_request_at_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("never"))
    assert main(["--no-web", "--num-ctx", "1024", "x" * 4096]) == 2
    assert len(responses.calls) == 0
    assert "Nothing was sent" in capsys.readouterr().err


@responses.activate
def test_a_post_hoc_trip_warns_and_exits_non_zero_after_the_text_was_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("bad", prompt_eval_count=16386))
    assert main(["--no-web", "--num-ctx", "32768", "x" * 80_000]) == 2
    captured = capsys.readouterr()
    assert captured.out == "bad\n"
    assert "do not trust this response" in captured.err


@responses.activate
def test_an_error_object_mid_stream_is_surfaced_rather_than_exiting_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = _ndjson({"response": "half", "done": False}, {"error": "model runner has terminated"})
    responses.add(responses.POST, GENERATE, body=body)
    assert main(["--no-web", "hello"]) == 1
    assert "model runner has terminated" in capsys.readouterr().err


@responses.activate
def test_a_transport_failure_is_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, status=500)
    assert main(["--no-web", "hello"]) == 1
    assert "error:" in capsys.readouterr().err


@responses.activate
def test_dry_run_reports_the_plan_and_sends_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("never"))
    assert main(["--dry-run", "-m", "coder", "hello"]) == 0
    out = capsys.readouterr().out
    assert len(responses.calls) == 0
    assert "coder -> qwen3-coder:30b" in out
    assert "window    24576" in out


@responses.activate
def test_dry_run_says_when_the_same_prompt_would_be_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--dry-run", "--num-ctx", "1024", "x" * 4096]) == 0
    assert "would be refused" in capsys.readouterr().out


@responses.activate
def test_piped_stdin_becomes_context(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("ok"))
    monkeypatch.setattr("ollama_stack.__main__.piped_context", lambda: "def f(): pass")
    main(["--no-web", "explain", "this"])
    assert "def f(): pass" in _sent()["prompt"]


@responses.activate
def test_ask_is_the_escape_for_a_question_starting_with_a_command_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`o ask` shares run_query with the bare path, so it takes the same search-enabled route."""
    responses.add(responses.POST, CHAT, body=_chat_body("fine"))
    with pytest.raises(SystemExit):
        main(["ask", "status of the economy"])
    assert _sent()["messages"][0]["content"] == "status of the economy"


@responses.activate
def test_the_default_path_offers_the_search_tool(capsys: pytest.CaptureFixture[str]) -> None:
    """Automatic is the default: the model gets the tool and decides whether to call it."""
    responses.add(responses.POST, CHAT, body=_chat_body("20"))
    assert main(["what", "is", "10+10"]) == 0
    sent = _sent()
    assert [tool["function"]["name"] for tool in sent["tools"]] == ["web_search"]
    assert capsys.readouterr().out == "20\n"


@responses.activate
def test_no_web_withholds_the_tool_so_it_cannot_be_called(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("20"))
    assert main(["--no-web", "what", "is", "10+10"]) == 0
    assert "tools" not in _sent()


@responses.activate
def test_forcing_a_search_puts_the_results_in_front_of_the_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ollama_stack import search

    hit = search.Result("Python 3.14.7", "https://python.org/x", "Latest stable.")
    monkeypatch.setattr(search.DuckDuckGo, "search", lambda self, q, count=5: [hit])
    responses.add(responses.POST, CHAT, body=_chat_body("3.14.7"))
    assert main(["-w", "latest", "python"]) == 0
    captured = capsys.readouterr()
    assert "https://python.org/x" in _sent()["messages"][0]["content"]
    assert "[1] https://python.org/x" in captured.err


@responses.activate
def test_sources_print_to_stderr_and_nothing_prints_when_no_search_happened(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sourced answer must be tellable from a remembered one, without dirtying stdout."""
    responses.add(responses.POST, CHAT, body=_chat_body("from memory"))
    main(["what", "is", "10+10"])
    captured = capsys.readouterr()
    assert captured.out == "from memory\n"
    assert "[1]" not in captured.err


@responses.activate
def test_a_search_provider_failure_never_fails_the_whole_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ollama_stack import search

    def blocked(self: object, query: str, count: int = 5) -> list[search.Result]:
        raise search.SearchError("rate limited")

    monkeypatch.setattr(search.DuckDuckGo, "search", blocked)
    responses.add(responses.POST, CHAT, body=_chat_body("best effort"))
    assert main(["-w", "anything"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "best effort\n"
    assert "rate limited" in captured.err


def test_web_and_no_web_together_is_an_error_not_a_precedence_rule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["-w", "--no-web", "hello"]) == 2
    assert main(["--no-web", "-w", "hello"]) == 2
    assert "contradict" in capsys.readouterr().err


def test_the_web_setting_is_reported_by_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--dry-run", "hello"])
    assert "web       automatic" in capsys.readouterr().out
    main(["--dry-run", "--no-web", "hello"])
    assert "web       off" in capsys.readouterr().out
    main(["--dry-run", "-w", "hello"])
    assert "web       forced" in capsys.readouterr().out


@responses.activate
def test_audit_withholds_the_search_tool(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    responses.add(responses.POST, GENERATE, body=_stream_body("nothing found"))
    with pytest.raises(SystemExit):
        main(["audit", str(target)])
    assert "tools" not in _sent()


def _modules(code: str) -> set[str]:
    """Top-level module names resident after running code in a fresh interpreter."""
    dump = "import sys; print(' '.join(sorted(sys.modules)), file=sys.stderr)"
    done = subprocess.run(
        [sys.executable, "-c", f"{code}\n{dump}"], capture_output=True, text=True, check=True
    )
    return {name.split(".")[0] for name in done.stderr.split()}


def test_the_bare_path_imports_nothing_outside_its_allowlist() -> None:
    """The allowlist is what the settings layer already costs, plus this package, and no more."""
    baseline = _modules("pass")
    reference = "from __future__ import annotations; import dataclasses, typing, tomllib, pathlib"
    allowed = (_modules(reference) - baseline) | {"ollama_stack"}
    bare = _modules(
        "import sys; sys.argv = ['o', '--dry-run', 'what', 'is', '10+10'];"
        "from ollama_stack.__main__ import main; main()"
    )
    extra = (bare - baseline) - allowed
    assert not extra, f"the bare path now imports {sorted(extra)}"
