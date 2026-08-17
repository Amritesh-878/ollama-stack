"""Web results for questions past the model's cutoff, from a provider needing no signup."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
# Without a browser agent the lite endpoint answers with an anti-bot page instead of results.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
)
SEARCH_TIMEOUT = 15
DEFAULT_COUNT = 5
SNIPPET_CHARS = 240


class SearchError(RuntimeError):
    """The provider could not be reached or answered with something unusable."""


@dataclass(frozen=True)
class Result:
    """One hit, trimmed to what a model can use without spending the context window on it."""

    title: str
    url: str
    snippet: str

    def as_text(self) -> str:
        return f"{self.title}\n{self.url}\n{self.snippet}"


class SearchProvider(Protocol):
    """Swappable so a keyed provider can replace the free one without touching the tool loop."""

    name: str

    def search(self, query: str, count: int = DEFAULT_COUNT) -> list[Result]: ...


def trim(results: list[Result], count: int, snippet_chars: int = SNIPPET_CHARS) -> list[Result]:
    """Trimmed hard before it reaches the model: blowing the window is this project's own bug."""
    trimmed = []
    for result in results[:count]:
        snippet = " ".join(result.snippet.split())
        if len(snippet) > snippet_chars:
            snippet = snippet[: snippet_chars - 3].rstrip() + "..."
        trimmed.append(Result(title=result.title.strip(), url=result.url, snippet=snippet))
    return trimmed


def as_prompt(results: list[Result]) -> str:
    """What the model sees, numbered so it can cite a result rather than paraphrase all of them."""
    if not results:
        return "No results were found."
    return "\n\n".join(f"[{n}] {r.as_text()}" for n, r in enumerate(results, 1))


def _direct_url(href: str) -> str:
    """The lite endpoint wraps every hit in its own redirect; the real URL is the uddg parameter."""
    if "uddg=" not in href:
        return href
    query = urllib.parse.urlparse(href if "//" in href else f"//{href}").query
    target = urllib.parse.parse_qs(query).get("uddg")
    return urllib.parse.unquote(target[0]) if target else href


class _LiteParser(HTMLParser):
    """A real parser, not a regex: the markup mixes quote styles and has changed shape before."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[Result] = []
        self._href: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_link = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: (value or "") for key, value in attrs}
        classes = values.get("class", "").split()
        if tag == "a" and "result-link" in classes:
            self._flush()
            self._in_link = True
            self._href = values.get("href", "")
        elif tag == "td" and "result-snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_link = False
        elif tag == "td":
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title.append(data)
        elif self._in_snippet:
            self._snippet.append(data)

    def _flush(self) -> None:
        if self._href is None:
            return
        title = "".join(self._title).strip()
        if title:
            self.results.append(
                Result(
                    title=title,
                    url=_direct_url(self._href),
                    snippet="".join(self._snippet).strip(),
                )
            )
        self._href, self._title, self._snippet = None, [], []

    def close(self) -> None:
        super().close()
        self._flush()


def parse_lite(page: str) -> list[Result]:
    """Public so a live capture can be replayed in a test without a network call."""
    parser = _LiteParser()
    parser.feed(page)
    parser.close()
    return parser.results


class DuckDuckGo:
    """The default because it needs no key: a public repo whose search needs a signup is dead."""

    name = "duckduckgo"

    def __init__(self, endpoint: str = LITE_ENDPOINT, timeout: int = SEARCH_TIMEOUT) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, count: int = DEFAULT_COUNT) -> list[Result]:
        import requests

        try:
            response = requests.post(
                self.endpoint,
                data={"q": query},
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SearchError(f"{self.name} could not be reached: {exc}") from exc
        page = response.text
        results = parse_lite(page)
        if not results and _looks_blocked(page):
            raise SearchError(f"{self.name} answered with a bot check instead of results")
        return trim(results, count)


def _looks_blocked(page: str) -> bool:
    """Rate limiting arrives as a normal 200, so the body is the only thing that says so."""
    return bool(re.search(r"anomaly|captcha|unusual traffic|challenge-form", page, re.I))
