"""The entry point, which parses argv itself so asking a question never imports typer."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ollama_stack.models import DEFAULT_ALIAS

if TYPE_CHECKING:
    from ollama_stack.client import OllamaClient, Reply, StreamRun
    from ollama_stack.config import Settings
    from ollama_stack.render import Pretty
    from ollama_stack.tools import SearchOutcome

RESERVED = frozenset(
    {
        "setup",
        "tutorial",
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

_WEB_LABELS = {None: "automatic, the model decides", True: "forced", False: "off"}

USAGE = """usage: o <question>          ask the default model, no quotes needed
       o ask "<question>"    when the question starts with a subcommand name
       o --help              subcommands and flags"""

# Hand-written rather than typer's: typer owns the subcommands but not the bare-path flags, so
# its screen can only list half of them and buries the rest in an epilog paragraph.
HELP = """usage: o <question>              ask a local model, no quotes needed
       o <command> [options]     run a command
       o ask "<question>"        when the question starts with a command name

Ask a local model something from the terminal.

commands:
  start [-m TAG]         pin a model in VRAM so the next question is fast
  stop [-m TAG]          release it and give the card back
  status                 what is loaded, how much VRAM, how long it stays
  ask "<question>"       ask, when the question starts with a command name
  audit <file>           have a model read one file and report defects
  implement <task-file>  drive a local model against a finished task file
  models                 list the model aliases and which have measurements
  which <alias>          show what an alias resolves to
  config                 show settings and where each one came from
  config set <k> <v>     change a setting
  config unset <k>       revert a setting to its default
  setup                  re-run the first-run wizard
  tutorial               nine guided steps, run on your own machine

options:
  -m, --model TAG        alias like `heavy`, or any Ollama tag
      --num-ctx N        context window to request (default 32768)
      --think            reason before answering; slower, better on hard questions
  -w, --web              search the web before answering
      --no-web           never search
      --no-stream        wait for the whole reply instead of streaming
      --stats            timings, token counts, and the pre-flight estimate
      --dry-run          show what would be sent, send nothing
      --version          print the version
  -h, --help             show this

examples:
  o what is 10+10
  o -m heavy explain this stack trace
  cat main.py | o explain this
  o ask "status of the roman empire"

A question whose FIRST word is a command runs that command instead, so
`o status of the economy` reports what is loaded rather than answering.
For those, quote it: o ask "status of the economy"

https://github.com/Amritesh-878/ollama-stack"""


class UsageError(Exception):
    """The argv is malformed, which is the user's problem and not Ollama's."""


@dataclass
class Options:
    """Everything the bare path can be told, parsed without argparse or typer."""

    model: str = DEFAULT_ALIAS
    # None means "not given on the command line", so config gets its turn in the precedence.
    num_ctx: int | None = None
    stream: bool | None = None
    stats: bool = False
    dry_run: bool = False
    version: bool = False
    think: bool = False
    # None is automatic: the model gets the tool and decides. True forces, False withholds it.
    web: bool | None = None


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
        elif token in ("-w", "--web"):
            if opts.web is False:
                raise UsageError("-w and --no-web contradict each other; pick one")
            opts.web = True
            index += 1
        elif token == "--no-web":
            if opts.web is True:
                raise UsageError("-w and --no-web contradict each other; pick one")
            opts.web = False
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


def _dry_run(opts: Options, prompt: str, context: str, settings: Settings) -> int:
    from ollama_stack.client import ContextTruncationError, OllamaClient, estimate_tokens
    from ollama_stack.models import resolve

    num_ctx = _num_ctx(opts, settings)
    client = OllamaClient(settings.host, num_ctx=num_ctx, think=opts.think)
    full = f"{context}\n\n{prompt}".strip() if context else prompt
    spec = resolve(opts.model)
    print(f"model     {opts.model} -> {spec.tag}")
    print(f"num_ctx   {num_ctx} (from {_where(opts, settings)})")
    print(f"window    {client.prompt_budget} (refused at or above)")
    print(f"estimate  {estimate_tokens(full)} prompt tokens")
    print(f"stream    {'on' if _stream(opts, settings) else 'off'}")
    print(f"think     {'on' if opts.think else 'off'}")
    print(f"web       {_WEB_LABELS[opts.web]}")
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
        f"prompt {reply.prompt_eval_count} tok read",
        f"generated {reply.eval_count}",
    ]
    if opts.stats:
        rate = reply.eval_count / seconds if seconds > 0 else 0.0
        # The honest comparison is what was sent against what was read, not against a threshold.
        parts[1] += f", ~{estimate} estimated"
        parts[2] += f" in {seconds:.2f}s wall, {rate:.1f} tok/s"
        # Wall time here starts at the request, so it excludes the ~80ms of process start.
        parts.extend(timings)
    print(f"[{' | '.join(parts)}]", file=sys.stderr)


def _num_ctx(opts: Options, settings: Settings) -> int:
    return opts.num_ctx if opts.num_ctx is not None else settings.num_ctx


def _stream(opts: Options, settings: Settings) -> bool:
    return opts.stream if opts.stream is not None else settings.stream


def _where(opts: Options, settings: Settings) -> str:
    return "flag" if opts.num_ctx is not None else settings.sources["num_ctx"]


def _searching(
    client: OllamaClient,
    opts: Options,
    prompt: str,
    context: str,
    started: float,
    settings: Settings,
) -> tuple[SearchOutcome, float | None]:
    """The tool loop, taken whenever search is not explicitly forbidden."""
    from ollama_stack.search import default_provider
    from ollama_stack.tools import answer_with_search

    first: list[float] = []
    held: list[str] = []
    streaming = _stream(opts, settings)

    pretty = _pretty()

    def write(piece: str) -> None:
        if not first:
            first.append(time.perf_counter() - started)
        if streaming:
            pretty.write(piece)
            sys.stdout.flush()
        else:
            held.append(piece)

    try:
        outcome = answer_with_search(
            client,
            prompt,
            opts.model,
            default_provider(),
            write,
            context=context,
            force=opts.web is True,
        )
    finally:
        # `streaming`, not opts.stream: the latter is None until config resolves it.
        if streaming:
            pretty.close()
            sys.stdout.write("\n")
            sys.stdout.flush()
    if not streaming:
        _tidy("".join(held))
    # Two searches that fail the same way are one fact, not two lines of noise.
    for note in dict.fromkeys(outcome.notes):
        print(f"note: {note}", file=sys.stderr)
    for number, source in enumerate(outcome.sources, 1):
        print(f"[{number}] {source.url}", file=sys.stderr)
    return outcome, (first[0] if first else None)


def _pretty() -> Pretty:
    """Markdown pipes and asterisks are noise in a terminal, so nothing reaches it unrendered."""
    from ollama_stack.render import Pretty as Renderer

    return Renderer(sys.stdout.write)


def _tidy(text: str) -> None:
    """A whole reply rather than a stream, so this one owns its trailing newline."""
    pretty = _pretty()
    pretty.write(text)
    pretty.close()
    sys.stdout.write("\n")


def run_query(opts: Options, prompt: str, context: str = "") -> int:
    """One implementation for the bare path and for `o ask`, so they cannot behave differently."""
    from ollama_stack import config
    from ollama_stack.client import (
        ContextTruncationError,
        OllamaClient,
        OllamaError,
        estimate_tokens,
    )

    settings = config.load()
    config.apply(settings)
    for warning in settings.warnings:
        print(f"config: {warning}", file=sys.stderr)
    if opts.dry_run:
        return _dry_run(opts, prompt, context, settings)
    client = OllamaClient(settings.host, num_ctx=_num_ctx(opts, settings), think=opts.think)
    started = time.perf_counter()
    first_word: float | None = None
    run: StreamRun | None = None
    searched = 0
    estimate = estimate_tokens(f"{context}\n\n{prompt}".strip() if context else prompt)
    try:
        if opts.web is not False:
            outcome, first_word = _searching(client, opts, prompt, context, started, settings)
            reply, run = outcome.reply, outcome.last_run
            # The conversation grew by whatever search returned, so the original estimate is stale.
            estimate, searched = outcome.prompt_estimate, outcome.searches
        elif _stream(opts, settings):
            run = client.stream(prompt, opts.model, context=context)
            pretty = _pretty()
            try:
                for chunk in run:
                    if first_word is None:
                        first_word = time.perf_counter() - started
                    pretty.write(chunk)
                    sys.stdout.flush()
            finally:
                pretty.close()
                sys.stdout.write("\n")
                sys.stdout.flush()
            reply = run.reply
        else:
            reply = client.generate(prompt, opts.model, context=context)
            _tidy(reply.text)
    except ContextTruncationError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    timings = _timings(run, first_word)
    if searched:
        timings.append(f"searched {searched}x")
    if reply.ran_out_of_window:
        print(
            f"note: {reply.model} used the whole {reply.num_ctx}-token window generating and was "
            "cut off before finishing. Try --no-think, a larger --num-ctx, or a different model.",
            file=sys.stderr,
        )
    _status_line(opts, reply, estimate, time.perf_counter() - started, timings)
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
    if words and words[0] in HELP_FLAGS:
        print(HELP)
        return 0
    if words and words[0] in RESERVED:
        return _subcommand(args)
    if not words:
        print(USAGE, file=sys.stderr)
        return 2
    return run_query(opts, " ".join(words), piped_context())


if __name__ == "__main__":
    raise SystemExit(main())
