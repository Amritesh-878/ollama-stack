"""Bare invocation is the whole product, so dispatch and streaming are tested at the CLI edge."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest
import responses

from ollama_stack.__main__ import RESERVED, _parse, main
from ollama_stack.cli import app

GENERATE = "http://localhost:11434/api/generate"


def _ndjson(*chunks: dict[str, Any]) -> str:
    return "\n".join(json.dumps(chunk) for chunk in chunks)


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
    assert opts.model == "qwen"


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
    assert RESERVED - registered == {"setup", "start", "stop", "status", "config", "implement"}


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
    assert main(["what", "is", "10+10"]) == 0
    assert capsys.readouterr().out == "10 + 10 is 20\n"


@responses.activate
def test_streaming_asks_ollama_to_stream(capsys: pytest.CaptureFixture[str]) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("hi"))
    main(["hello"])
    assert _sent()["stream"] is True


@responses.activate
def test_stdout_carries_only_the_answer_and_the_stats_line_goes_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("42"))
    main(["the", "answer"])
    captured = capsys.readouterr()
    assert captured.out == "42\n"
    assert "prompt 10/16384 tok" in captured.err


@responses.activate
def test_no_stream_sends_one_request_that_is_not_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = {"response": "42", "prompt_eval_count": 10, "eval_count": 1}
    responses.add(responses.POST, GENERATE, json=body)
    assert main(["--no-stream", "the", "answer"]) == 0
    assert _sent()["stream"] is False
    assert capsys.readouterr().out == "42\n"


@responses.activate
def test_stats_adds_the_estimate_and_timing_to_the_stderr_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("42"))
    main(["--stats", "the", "answer"])
    err = capsys.readouterr().err
    assert "estimated" in err
    assert "tok/s" in err


@responses.activate
def test_a_preflight_refusal_sends_no_request_at_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("never"))
    assert main(["--num-ctx", "1024", "x" * 4096]) == 2
    assert len(responses.calls) == 0
    assert "Nothing was sent" in capsys.readouterr().err


@responses.activate
def test_a_post_hoc_trip_warns_and_exits_non_zero_after_the_text_was_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("bad", prompt_eval_count=9000))
    assert main(["--num-ctx", "4096", "hello"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "bad\n"
    assert "do not trust this response" in captured.err


@responses.activate
def test_an_error_object_mid_stream_is_surfaced_rather_than_exiting_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = _ndjson({"response": "half", "done": False}, {"error": "model runner has terminated"})
    responses.add(responses.POST, GENERATE, body=body)
    assert main(["hello"]) == 1
    assert "model runner has terminated" in capsys.readouterr().err


@responses.activate
def test_a_transport_failure_is_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, status=500)
    assert main(["hello"]) == 1
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
    assert "window    16384" in out


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
    main(["explain", "this"])
    assert "def f(): pass" in _sent()["prompt"]


@responses.activate
def test_ask_is_the_escape_for_a_question_starting_with_a_command_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses.add(responses.POST, GENERATE, body=_stream_body("fine"))
    with pytest.raises(SystemExit):
        main(["ask", "status of the economy"])
    assert _sent()["prompt"] == "status of the economy"


def _modules(code: str) -> set[str]:
    """Top-level module names resident after running code in a fresh interpreter."""
    dump = "import sys; print(' '.join(sorted(sys.modules)), file=sys.stderr)"
    done = subprocess.run(
        [sys.executable, "-c", f"{code}\n{dump}"], capture_output=True, text=True, check=True
    )
    return {name.split(".")[0] for name in done.stderr.split()}


def test_the_bare_path_imports_nothing_outside_its_allowlist() -> None:
    """The allowlist is whatever dataclasses and typing already cost, plus this package."""
    baseline = _modules("pass")
    reference = "from __future__ import annotations; import dataclasses, typing"
    allowed = (_modules(reference) - baseline) | {"ollama_stack"}
    bare = _modules(
        "import sys; sys.argv = ['o', '--dry-run', 'what', 'is', '10+10'];"
        "from ollama_stack.__main__ import main; main()"
    )
    extra = (bare - baseline) - allowed
    assert not extra, f"the bare path now imports {sorted(extra)}"
