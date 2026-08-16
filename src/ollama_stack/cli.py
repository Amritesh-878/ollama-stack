"""The `o` subcommands. A bare question never reaches this module, and so never imports typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ollama_stack.__main__ import Options, piped_context, run_query
from ollama_stack.models import DEFAULT_ALIAS, DEFAULT_NUM_CTX, REGISTRY, resolve

app = typer.Typer(
    add_completion=False,
    help="Local Ollama models, one command.",
    epilog=(
        "Ask a bare question with no command and no quotes: `o what is 10+10`. Bare questions "
        "take -m/--model, --num-ctx, --no-stream, --stats, --dry-run and --version; every "
        "other word is the question. A question whose FIRST word is one of the commands above "
        "runs that command instead, so `o status of the economy` reports what is loaded - ask "
        'it as a question with `o ask "status of the economy"`. setup, config and implement '
        "are reserved the same way before they exist."
    ),
)


@app.command()
def ask(
    prompt: str,
    model: str = typer.Option(DEFAULT_ALIAS, "--model", "-m", help="Registry alias or raw tag."),
    num_ctx: int = typer.Option(DEFAULT_NUM_CTX, "--num-ctx", help="Context window to request."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for the whole reply."),
    stats: bool = typer.Option(False, "--stats", help="Add wall time and the token estimate."),
) -> None:
    """Send a prompt to a model. The escape for a question starting with a command name."""
    opts = Options(model=model, num_ctx=num_ctx, stream=not no_stream, stats=stats)
    raise typer.Exit(run_query(opts, prompt, piped_context()))


@app.command()
def audit(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    model: str = typer.Option("coder", "--model", "-m"),
    num_ctx: int = typer.Option(DEFAULT_NUM_CTX, "--num-ctx"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    stats: bool = typer.Option(False, "--stats"),
) -> None:
    """Screen one file. Its silence carries no information - every file it passes is unexamined."""
    body = file.read_text(encoding="utf-8", errors="replace")
    prompt = (
        "Review the file below for defects. Report each as file:line with the concrete failure. "
        "Do not summarize what the code does. If you find nothing, say so plainly."
    )
    opts = Options(model=model, num_ctx=num_ctx, stream=not no_stream, stats=stats)
    raise typer.Exit(run_query(opts, prompt, f"--- {file.name} ---\n{body}"))


@app.command()
def models() -> None:
    """List the routing table, flagging which models have measurements behind them."""
    for alias, spec in REGISTRY.items():
        flag = "measured" if spec.measured else "UNMEASURED"
        typer.echo(f"{alias:10s} {spec.tag:40s} [{flag}] {spec.summary}")


@app.command()
def which(name: str) -> None:
    """Resolve an alias or tag to what would actually run."""
    spec = resolve(name)
    typer.echo(f"{name} -> {spec.tag} ({spec.summary})")
    if not spec.measured:
        typer.secho("no measurements behind this model", fg=typer.colors.YELLOW, err=True)


if __name__ == "__main__":
    app()
