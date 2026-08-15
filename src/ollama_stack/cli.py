"""The `o` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ollama_stack.client import ContextTruncationError, OllamaClient, OllamaError
from ollama_stack.models import DEFAULT_ALIAS, DEFAULT_NUM_CTX, REGISTRY, resolve

app = typer.Typer(add_completion=False, help="Local Ollama models, one command.")


def _run(prompt: str, alias: str, num_ctx: int, context: str = "") -> None:
    client = OllamaClient(num_ctx=num_ctx)
    try:
        reply = client.generate(prompt, alias, context=context)
    except ContextTruncationError as exc:
        typer.secho(f"refused: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    except OllamaError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(reply.text)
    typer.secho(
        f"[{reply.model} | prompt {reply.prompt_eval_count}/{reply.usable_window} tok "
        f"| generated {reply.eval_count}]",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )


@app.command()
def ask(
    prompt: str,
    model: str = typer.Option(DEFAULT_ALIAS, "--model", "-m", help="Registry alias or raw tag."),
    num_ctx: int = typer.Option(DEFAULT_NUM_CTX, "--num-ctx", help="Context window to request."),
) -> None:
    """Send a prompt to a model, by alias or raw tag."""
    _run(prompt, model, num_ctx)


@app.command()
def audit(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    model: str = typer.Option("coder", "--model", "-m"),
    num_ctx: int = typer.Option(DEFAULT_NUM_CTX, "--num-ctx"),
) -> None:
    """Screen one file. Its silence carries no information - every file it passes is unexamined."""
    body = file.read_text(encoding="utf-8", errors="replace")
    prompt = (
        "Review the file below for defects. Report each as file:line with the concrete failure. "
        "Do not summarize what the code does. If you find nothing, say so plainly."
    )
    _run(prompt, model, num_ctx, context=f"--- {file.name} ---\n{body}")


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
