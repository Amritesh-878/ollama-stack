"""The entry point, which parses argv itself so asking a question never imports typer."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ollama_stack.models import DEFAULT_ALIAS, DEFAULT_NUM_CTX

if TYPE_CHECKING:
    from ollama_stack.client import Reply, StreamRun

RESERVED = frozenset(
    {
        "setup",
        "start",
        "stop",
        "status",
        "ask",
        "audit",
        "models",
        "which",
        "config",
        "implement",
    }
)

HELP_FLAGS = frozenset({"-h", "--help"})

USAGE = """usage: o <question>          ask the default model, no quotes needed
       o ask "<question>"    when the question starts with a subcommand name
       o --help              subcommands and flags"""


class UsageError(Exception):
    """The argv is malformed, which is the user's problem and not Ollama's."""


@dataclass
class Options:
    """Everything the bare path can be told, parsed without argparse or typer."""

    model: str = DEFAULT_ALIAS
    num_ctx: int = DEFAULT_NUM_CTX
    stream: bool = True
    stats: bool = False
    dry_run: bool = False
    version: bool = False
    think: bool = False


def _value(argv: list[str], index: int, token: str) -> str:
    if index + 1 >= len(argv):
        raise UsageError(f"{token} wants a value")
    return argv[index + 1]


def _parse(argv: list[str]) -> tuple[Options, list[str]]:
    """Anything not a known flag is prompt text, which is what makes bare invocation work."""
    opts = Options()
    words: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("-m", "--model"):
            opts.model = _value(argv, index, token)
            index += 2
        elif token == "--num-ctx":
            raw = _value(argv, index, token)
            if not raw.isdigit() or int(raw) <= 0:
                raise UsageError(f"{token} wants a positive number, got {raw!r}")
            opts.num_ctx = int(raw)
            index += 2
        elif token == "--no-stream":
            opts.stream = False
            index += 1
        elif token == "--stats":
            opts.stats = True
            index += 1
        elif token == "--think":
            opts.think = True
            index += 1
        elif token == "--no-think":
            opts.think = False
            index += 1
        elif token == "--dry-run":
            opts.dry_run = True
            index += 1
        elif token == "--version":
            opts.version = True
            index += 1
        else:
            words.append(token)
            index += 1
    return opts, words


def _utf8(stream: object) -> None:
    """A redirected stream defaults to cp1252 here, which raises on an arrow the model wrote."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        return


def piped_context() -> str:
    """Piped input is context; a terminal is not, and isatty is the only way to tell."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except (AttributeError, ValueError, OSError):
        return ""


def _subcommand(argv: list[str]) -> int:
    """Everything off the fast path pays for typer, which is the reason the fast path exists."""
    from ollama_stack.cli import app

    app(argv, prog_name="o")
    return 0


def _dry_run(opts: Options, prompt: str, context: str) -> int:
    from ollama_stack.client import ContextTruncationError, OllamaClient, estimate_tokens
    from ollama_stack.models import resolve

    client = OllamaClient(num_ctx=opts.num_ctx, think=opts.think)
    full = f"{context}\n\n{prompt}".strip() if context else prompt
    spec = resolve(opts.model)
    print(f"model     {opts.model} -> {spec.tag}")
    print(f"num_ctx   {opts.num_ctx}")
    print(f"window    {client.usable_window} (refused at or above)")
    print(f"estimate  {estimate_tokens(full)} prompt tokens")
    print(f"stream    {'on' if opts.stream else 'off'}")
    print(f"think     {'on' if opts.think else 'off'}")
    try:
        client.preflight(full)
    except ContextTruncationError as exc:
        print(f"would be refused: {exc}")
    return 0


def _timings(run: StreamRun | None, first_word: float | None) -> list[str]:
    """Two figures, because a reasoning model answers long before it says anything."""
    if run is None:
        return []
    timings = []
    if run.first_chunk_seconds is not None:
        timings.append(f"first chunk {run.first_chunk_seconds * 1000:.0f}ms")
    if first_word is not None:
        timings.append(f"first word {first_word * 1000:.0f}ms")
    if run.thinking_tokens:
        timings.append(f"thought {run.thinking_tokens} tok first")
    return timings


def _status_line(
    opts: Options,
    reply: Reply,
    estimate: int,
    seconds: float,
    timings: list[str],
) -> None:
    parts = [
        reply.model,
        f"prompt {reply.prompt_eval_count}/{reply.usable_window} tok",
        f"generated {reply.eval_count}",
    ]
    if opts.stats:
        rate = reply.eval_count / seconds if seconds > 0 else 0.0
        parts[1] += f", estimated {estimate}"
        parts[2] += f" in {seconds:.2f}s wall, {rate:.1f} tok/s"
        # Wall time here starts at the request, so it excludes the ~80ms of process start.
        parts.extend(timings)
    print(f"[{' | '.join(parts)}]", file=sys.stderr)


def run_query(opts: Options, prompt: str, context: str = "") -> int:
    """One implementation for the bare path and for `o ask`, so they cannot behave differently."""
    from ollama_stack.client import (
        ContextTruncationError,
        OllamaClient,
        OllamaError,
        estimate_tokens,
    )

    if opts.dry_run:
        return _dry_run(opts, prompt, context)
    client = OllamaClient(num_ctx=opts.num_ctx, think=opts.think)
    started = time.perf_counter()
    first_word: float | None = None
    run: StreamRun | None = None
    try:
        if opts.stream:
            run = client.stream(prompt, opts.model, context=context)
            try:
                for chunk in run:
                    if first_word is None:
                        first_word = time.perf_counter() - started
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            finally:
                sys.stdout.write("\n")
                sys.stdout.flush()
            reply = run.reply
        else:
            reply = client.generate(prompt, opts.model, context=context)
            print(reply.text)
    except ContextTruncationError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    estimate = estimate_tokens(f"{context}\n\n{prompt}".strip() if context else prompt)
    _status_line(opts, reply, estimate, time.perf_counter() - started, _timings(run, first_word))
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8(sys.stdout)
    _utf8(sys.stderr)
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        opts, words = _parse(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if opts.version:
        from ollama_stack import __version__

        print(__version__)
        return 0
    if words and (words[0] in RESERVED or words[0] in HELP_FLAGS):
        return _subcommand(args)
    if not words:
        print(USAGE, file=sys.stderr)
        return 2
    return run_query(opts, " ".join(words), piped_context())


if __name__ == "__main__":
    raise SystemExit(main())
