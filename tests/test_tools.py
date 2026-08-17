"""The tool loop is the first stateful thing here, so the cap and the degradation are the tests."""

from __future__ import annotations

import json
from typing import Any

import responses

from ollama_stack.client import OllamaClient
from ollama_stack.search import Result, SearchError
from ollama_stack.tools import WEB_SEARCH, answer_with_search, query_of

CHAT = "http://127.0.0.1:11434/api/chat"

HIT = Result("Python 3.14.7", "https://python.org/downloads/", "The latest stable release.")


class FakeProvider:
    """Records what it was asked, because the loop's job is to ask the right thing."""

    name = "fake"

    def __init__(self, *rounds: list[Result] | Exception) -> None:
        self.rounds = list(rounds)
        self.queries: list[str] = []

    def search(self, query: str, count: int = 5) -> list[Result]:
        self.queries.append(query)
        answer = self.rounds.pop(0) if self.rounds else []
        if isinstance(answer, Exception):
            raise answer
        return answer


def _chunks(*items: dict[str, Any]) -> str:
    return "\n".join(json.dumps(item) for item in items)


def _tool_turn(query: str = "python latest release") -> str:
    call = {"function": {"name": "web_search", "arguments": {"query": query}}}
    return _chunks(
        {"message": {"tool_calls": [call]}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 20, "eval_count": 5},
    )


def _text_turn(*words: str) -> str:
    parts: list[dict[str, Any]] = [{"message": {"content": w}, "done": False} for w in words]
    parts.append(
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 30, "eval_count": 4}
    )
    return _chunks(*parts)


def _sent(index: int) -> dict[str, Any]:
    raw = responses.calls[index].request.body
    body: dict[str, Any] = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    return body


def _run(provider: FakeProvider, **kwargs: Any) -> tuple[Any, list[str]]:
    written: list[str] = []
    outcome = answer_with_search(
        OllamaClient(), "what is the latest python", "fast", provider, written.append, **kwargs
    )
    return outcome, written


def test_a_stringified_arguments_object_still_yields_a_query() -> None:
    """Arguments have arrived as JSON text rather than an object before, per BRAINSTORM 2b."""
    call = {"function": {"name": "web_search", "arguments": '{"query": "hello"}'}}
    assert query_of(call) == "hello"


def test_arguments_that_are_neither_json_nor_an_object_do_not_crash_the_loop() -> None:
    assert query_of({"function": {"name": "web_search", "arguments": "just text"}}) == "just text"
    assert query_of({"function": {"name": "web_search", "arguments": ["a"]}}) is None
    assert query_of({"function": {"name": "web_search", "arguments": {}}}) is None
    assert query_of({"function": {"name": "web_search", "arguments": {"query": "  "}}}) is None
    assert query_of({"nonsense": True}) is None


@responses.activate
def test_the_tool_is_offered_on_the_first_turn() -> None:
    responses.add(responses.POST, CHAT, body=_text_turn("no", " search"))
    _run(FakeProvider())
    assert _sent(0)["tools"] == [WEB_SEARCH]


@responses.activate
def test_a_model_that_does_not_search_just_answers() -> None:
    responses.add(responses.POST, CHAT, body=_text_turn("42"))
    outcome, written = _run(FakeProvider())
    assert "".join(written) == "42"
    assert (outcome.searches, outcome.sources) == (0, [])


@responses.activate
def test_one_search_then_an_answer_feeds_the_results_back_as_a_tool_message() -> None:
    responses.add(responses.POST, CHAT, body=_tool_turn("python latest"))
    responses.add(responses.POST, CHAT, body=_text_turn("3.14.7"))
    provider = FakeProvider([HIT])
    outcome, written = _run(provider)
    assert provider.queries == ["python latest"]
    assert "".join(written) == "3.14.7"
    assert outcome.sources == [HIT]
    second = _sent(1)["messages"]
    assert second[-1]["role"] == "tool"
    assert "python.org" in second[-1]["content"]


@responses.activate
def test_the_loop_stops_after_three_searches_and_answers_without_the_tool() -> None:
    """A model that keeps searching is a hang, so the last turn goes out with no tool at all."""
    for _ in range(3):
        responses.add(responses.POST, CHAT, body=_tool_turn())
    responses.add(responses.POST, CHAT, body=_text_turn("done"))
    outcome, _ = _run(FakeProvider([HIT], [HIT], [HIT]))
    assert outcome.searches == 3
    assert "stopped after 3 searches" in outcome.notes
    assert "tools" not in _sent(3)


@responses.activate
def test_a_provider_failure_degrades_to_answering_without_search() -> None:
    responses.add(responses.POST, CHAT, body=_tool_turn())
    responses.add(responses.POST, CHAT, body=_text_turn("best effort"))
    outcome, written = _run(FakeProvider(SearchError("rate limited")))
    assert "".join(written) == "best effort"
    assert outcome.sources == []
    assert any("rate limited" in note for note in outcome.notes)


@responses.activate
def test_a_provider_returning_nothing_says_so_and_still_answers() -> None:
    responses.add(responses.POST, CHAT, body=_tool_turn("obscure"))
    responses.add(responses.POST, CHAT, body=_text_turn("nothing found"))
    outcome, _ = _run(FakeProvider([]))
    assert any("no results" in note for note in outcome.notes)
    assert "No results were found." in _sent(1)["messages"][-1]["content"]


@responses.activate
def test_a_search_call_with_no_query_is_reported_rather_than_crashing() -> None:
    call = {"function": {"name": "web_search", "arguments": {}}}
    responses.add(
        responses.POST,
        CHAT,
        body=_chunks(
            {"message": {"tool_calls": [call]}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 20, "eval_count": 1},
        ),
    )
    responses.add(responses.POST, CHAT, body=_text_turn("anyway"))
    outcome, _ = _run(FakeProvider())
    assert any("without giving a query" in note for note in outcome.notes)
    assert outcome.searches == 0


@responses.activate
def test_forcing_a_search_puts_results_in_the_first_prompt_and_skips_the_tool_dance() -> None:
    responses.add(responses.POST, CHAT, body=_text_turn("3.14.7"))
    provider = FakeProvider([HIT])
    outcome, _ = _run(provider, force=True)
    assert provider.queries == ["what is the latest python"]
    assert "python.org" in _sent(0)["messages"][0]["content"]
    assert outcome.sources == [HIT]
    assert outcome.searches == 1


@responses.activate
def test_forcing_a_search_that_finds_nothing_still_asks_the_model() -> None:
    responses.add(responses.POST, CHAT, body=_text_turn("from memory"))
    outcome, written = _run(FakeProvider(SearchError("offline")), force=True)
    assert "".join(written) == "from memory"
    assert any("offline" in note for note in outcome.notes)


@responses.activate
def test_the_estimate_counts_the_results_that_were_actually_sent() -> None:
    """The pre-flight guard sees the whole conversation, so the reported estimate must too."""
    responses.add(responses.POST, CHAT, body=_tool_turn())
    responses.add(responses.POST, CHAT, body=_text_turn("ok"))
    big = Result("t", "https://e", "x" * 2000)
    outcome, _ = _run(FakeProvider([big]))
    assert outcome.prompt_estimate > 100
