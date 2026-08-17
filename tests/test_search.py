"""Parsing someone else's HTML is the fragile part, so it is tested against a real capture."""

from __future__ import annotations

import pytest
import responses

from ollama_stack.search import (
    HTML_ENDPOINT,
    LITE_ENDPOINT,
    DuckDuckGo,
    Result,
    SearchError,
    as_prompt,
    parse_lite,
    trim,
)

# Captured from lite.duckduckgo.com on 2026-08-17, shortened. The quoting is theirs, not ours:
# href is double-quoted and class is single-quoted in the same tag.
CAPTURE = """
<table border="0">
  <tr><td valign="top">1.&nbsp;</td><td>
    <a rel="nofollow"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.formula1.com%2Fen%2Fresults&amp;rut=a8b2"
       class='result-link'>2026 RACE RESULTS - Formula 1</a>
  </td></tr>
  <tr><td>&nbsp;</td><td class='result-snippet'>
    Enter the world of Formula 1. Your go-to source for the latest <b>F1</b> news and GP results.
  </td></tr>
  <tr><td>&nbsp;</td><td><span class='link-text'>www.formula1.com/en/results</span></td></tr>
  <tr><td valign="top">2.&nbsp;</td><td>
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bbc.co.uk%2Fsport%2Fformula1"
       class='result-link'>F1 Latest Results - BBC Sport</a>
  </td></tr>
  <tr><td>&nbsp;</td><td class='result-snippet'>Find out the full results.</td></tr>
</table>
"""

BOT_CHECK = """
<html><head><title>DuckDuckGo</title></head>
<body><p>Our systems have detected unusual traffic from your computer network.</p>
<form class="challenge-form"></form></body></html>
"""


def test_a_real_capture_parses_into_results() -> None:
    results = parse_lite(CAPTURE)
    assert len(results) == 2
    assert results[0].title == "2026 RACE RESULTS - Formula 1"
    assert "Formula 1" in results[0].snippet
    assert results[1].title == "F1 Latest Results - BBC Sport"


def test_the_redirect_wrapper_is_unwrapped_to_the_real_url() -> None:
    """Every hit arrives wrapped in duckduckgo.com/l/, which is no use to a reader."""
    results = parse_lite(CAPTURE)
    assert results[0].url == "https://www.formula1.com/en/results"
    assert results[1].url == "https://www.bbc.co.uk/sport/formula1"


def test_a_page_with_no_results_parses_to_nothing_rather_than_raising() -> None:
    assert parse_lite("<html><body>nothing here</body></html>") == []


def test_trimming_caps_the_count_and_the_snippet_length() -> None:
    """Ten pages of scraped text is exactly the failure this project is organised around."""
    many = [Result(f"t{n}", f"https://e/{n}", "word " * 200) for n in range(20)]
    trimmed = trim(many, count=3, snippet_chars=50)
    assert len(trimmed) == 3
    assert all(len(r.snippet) <= 50 for r in trimmed)
    assert trimmed[0].snippet.endswith("...")


def test_trimming_collapses_the_whitespace_that_html_leaves_behind() -> None:
    trimmed = trim([Result("t", "u", "  a\n\n   b  ")], count=1)
    assert trimmed[0].snippet == "a b"


def test_the_prompt_form_numbers_results_so_the_model_can_cite_one() -> None:
    text = as_prompt([Result("Title", "https://example.com", "Snippet")])
    assert text.startswith("[1] Title")
    assert "https://example.com" in text


def test_no_results_says_so_rather_than_sending_an_empty_string() -> None:
    assert as_prompt([]) == "No results were found."


@responses.activate
def test_the_provider_returns_parsed_results() -> None:
    responses.add(responses.POST, LITE_ENDPOINT, body=CAPTURE, status=200)
    results = DuckDuckGo().search("f1 results", count=5)
    assert [r.title for r in results] == [
        "2026 RACE RESULTS - Formula 1",
        "F1 Latest Results - BBC Sport",
    ]


@responses.activate
def test_the_provider_sends_a_browser_agent_because_the_endpoint_requires_one() -> None:
    responses.add(responses.POST, LITE_ENDPOINT, body=CAPTURE, status=200)
    DuckDuckGo().search("f1 results")
    assert "Mozilla" in str(responses.calls[0].request.headers.get("User-Agent"))


@responses.activate
def test_a_bot_check_is_an_error_not_an_empty_result_set() -> None:
    """Rate limiting arrives as a 200; reading it as "no results" would hide it completely."""
    responses.add(responses.POST, LITE_ENDPOINT, body=BOT_CHECK, status=200)
    responses.add(responses.POST, HTML_ENDPOINT, body=BOT_CHECK, status=200)
    with pytest.raises(SearchError, match="rate-limiting"):
        DuckDuckGo().search("anything")


@responses.activate
def test_the_other_mirror_is_tried_when_the_first_is_blocked() -> None:
    """Lite is rate-limited often, and html serves the same results under different markup."""
    responses.add(responses.POST, LITE_ENDPOINT, body=BOT_CHECK, status=200)
    responses.add(responses.POST, HTML_ENDPOINT, body=HTML_PAGE, status=200)
    found = DuckDuckGo().search("python latest version")
    assert found and found[0].url == "https://www.python.org/downloads/"


@responses.activate
def test_a_transport_failure_is_a_search_error_not_a_leaked_requests_error() -> None:
    responses.add(responses.POST, LITE_ENDPOINT, status=503)
    with pytest.raises(SearchError, match="could not be reached"):
        DuckDuckGo().search("anything")


@responses.activate
def test_an_honestly_empty_page_returns_no_results_without_raising() -> None:
    responses.add(responses.POST, LITE_ENDPOINT, body="<html><body></body></html>", status=200)
    assert DuckDuckGo().search("nothing at all") == []


HTML_PAGE = """<div class="links_main links_deep result__body">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://www.python.org/downloads/">Download Python</a>
  </h2>
  <a class="result__snippet">The official home of Python.</a>
</div>"""
