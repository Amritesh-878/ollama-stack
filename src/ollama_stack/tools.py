"""The web_search tool and the loop that runs it, capped so a searching model cannot hang."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ollama_stack.search import DEFAULT_COUNT, Result, SearchError, as_prompt

if TYPE_CHECKING:
    from ollama_stack.client import OllamaClient, Reply, StreamRun
    from ollama_stack.search import SearchProvider

MAX_SEARCHES = 3

WEB_SEARCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this only when the answer depends on "
            "events or facts after your training cutoff, or on something specific you do not "
            "know. Do not use it for arithmetic, definitions, code, or anything you already "
            "know the answer to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, as you would type it into a search box.",
                }
            },
            "required": ["query"],
        },
    },
}


@dataclass
class SearchOutcome:
    """What the loop did, so the CLI can attribute the answer and report any degradation."""

    reply: Reply
    sources: list[Result] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    searches: int = 0
    last_run: StreamRun | None = None
    prompt_estimate: int = 0


def query_of(call: dict[str, Any]) -> str | None:
    """Arguments have arrived as a JSON string rather than an object before now, so both parse."""
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return arguments.strip() or None
    if not isinstance(arguments, dict):
        return None
    query = arguments.get("query")
    if query is None:
        return None
    return str(query).strip() or None


def _estimate(messages: list[dict[str, Any]]) -> int:
    """Counted over the whole conversation, so trimmed results show up against the real count."""
    from ollama_stack.client import estimate_tokens

    return estimate_tokens("\n".join(str(m.get("content", "")) for m in messages))


def _search(provider: SearchProvider, query: str, count: int) -> tuple[list[Result], str]:
    """A provider failure degrades to answering without results, never to failing the command."""
    try:
        found = provider.search(query, count)
    except SearchError as exc:
        return [], f"search unavailable, answering without it: {exc}"
    if not found:
        return [], f"no results for {query!r}, answering without them"
    return found, ""


def answer_with_search(
    client: OllamaClient,
    prompt: str,
    model: str,
    provider: SearchProvider,
    write: Callable[[str], None],
    *,
    context: str = "",
    force: bool = False,
    max_searches: int = MAX_SEARCHES,
    count: int = DEFAULT_COUNT,
) -> SearchOutcome:
    """Runs turns until the model answers without asking for another search, or the cap is hit."""
    sources: list[Result] = []
    notes: list[str] = []
    searches = 0
    turns = 0

    if force:
        found, note = _search(provider, prompt, count)
        searches += 1
        if note:
            notes.append(note)
        sources.extend(found)
        if found:
            # Forced results go in as context, not as a tool message with no call to answer.
            context = f"{context}\n\nWeb results:\n{as_prompt(found)}".strip()

    body = f"{context}\n\n{prompt}".strip() if context else prompt
    messages: list[dict[str, Any]] = [{"role": "user", "content": body}]
    tools: list[dict[str, Any]] | None = [WEB_SEARCH]

    while True:
        turns += 1
        run = client.chat_stream(messages, model, tools=tools)
        for piece in run:
            write(piece)
        reply = run.reply
        if not reply.tool_calls or tools is None:
            return SearchOutcome(
                reply,
                sources,
                notes,
                searches,
                last_run=run,
                prompt_estimate=_estimate(messages),
            )
        messages.append(
            {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls}
        )
        for call in reply.tool_calls:
            query = query_of(call)
            if query is None:
                notes.append("the model asked to search without giving a query")
                messages.append({"role": "tool", "content": "No query was given."})
                continue
            found, note = _search(provider, query, count)
            searches += 1
            if note:
                notes.append(note)
            sources.extend(found)
            messages.append({"role": "tool", "content": as_prompt(found)})
        # The final turn goes out without the tool, so the model has to answer with what it has.
        if searches >= max_searches:
            notes.append(f"stopped after {searches} searches")
            tools = None
        elif turns > max_searches:
            # A call with no query never increments searches, so turns must be capped or this loops.
            notes.append(f"stopped after {turns} turns without a usable query")
            tools = None
