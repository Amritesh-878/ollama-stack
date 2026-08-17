"""The `o` subcommands. A bare question never reaches this module, and so never imports typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ollama_stack import config, lifecycle
from ollama_stack.__main__ import Options, piped_context, run_query
from ollama_stack.client import OllamaClient, OllamaError
from ollama_stack.lifecycle import Resident, Vram
from ollama_stack.models import DEFAULT_ALIAS, HEAVY_ALIAS, REGISTRY, resolve

app = typer.Typer(
    add_completion=False,
    help="Local Ollama models, one command.",
    epilog=(
        "Ask a bare question with no command and no quotes: `o what is 10+10`. Bare questions "
        "take -m/--model, --num-ctx, --no-stream, --stats, --think/--no-think, -w/--no-web, "
        "--dry-run and --version; every other word is the question. Defaults come from "
        "`o config`, and the precedence is flag > env (OLLAMA_STACK_*) > file > built-in. A "
        "question whose FIRST word is one of the commands above runs that command instead, so "
        "`o status of the economy` reports what is loaded - ask it as a question with "
        '`o ask "status of the economy"`. implement is reserved the same way before it exists.'
    ),
)


@app.command()
def ask(
    prompt: str,
    model: str = typer.Option(DEFAULT_ALIAS, "--model", "-m", help="Registry alias or raw tag."),
    num_ctx: int | None = typer.Option(None, "--num-ctx", help="Context window to request."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for the whole reply."),
    stats: bool = typer.Option(False, "--stats", help="Add wall time and the token estimate."),
    think: bool = typer.Option(False, "--think/--no-think", help="Reason before answering."),
    web: bool = typer.Option(True, "--web/--no-web", "-w", help="Allow a web search."),
) -> None:
    """Send a prompt to a model. The escape for a question starting with a command name."""
    opts = Options(
        model=model,
        num_ctx=num_ctx,
        stream=False if no_stream else None,
        stats=stats,
        think=think,
        web=None if web else False,
    )
    raise typer.Exit(run_query(opts, prompt, piped_context()))


@app.command()
def audit(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    model: str = typer.Option(HEAVY_ALIAS, "--model", "-m"),
    num_ctx: int | None = typer.Option(None, "--num-ctx"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    stats: bool = typer.Option(False, "--stats"),
    think: bool = typer.Option(True, "--think/--no-think", help="On here: screening is not a"
                              " one-step question."),
) -> None:
    """Screen one file. Its silence carries no information - every file it passes is unexamined."""
    body = file.read_text(encoding="utf-8", errors="replace")
    prompt = (
        "Review the file below for defects. Report each as file:line with the concrete failure. "
        "Do not summarize what the code does. If you find nothing, say so plainly."
    )
    # Screening reads one file, so a search tool would only invite the model to wander off it.
    opts = Options(
        model=model,
        num_ctx=num_ctx,
        stream=False if no_stream else None,
        stats=stats,
        think=think,
        web=False,
    )
    raise typer.Exit(run_query(opts, prompt, f"--- {file.name} ---\n{body}"))


def _bar(vram: Vram, width: int = 20) -> str:
    if vram.total_mib <= 0:
        return "[" + "?" * width + "]"
    filled = min(width, max(0, round(width * vram.used_mib / vram.total_mib)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _echo_resident(resident: Resident | None) -> None:
    if resident is None:
        typer.echo("  ollama reports it as not loaded, which disagrees with the load it just did")
        return
    split = "unknown" if resident.gpu_percent is None else f"{resident.gpu_percent}% GPU"
    typer.echo(
        f"  {resident.size_mib} MiB total, {resident.vram_mib} MiB on the card ({split}), "
        f"ctx {resident.context_length}, ttl {resident.ttl}"
    )


def _echo_vram(vram: Vram | None) -> None:
    if vram is None:
        typer.echo("  card: unknown, nvidia-smi did not answer")
        return
    typer.echo(f"  card: {vram.used_mib}/{vram.total_mib} MiB used {_bar(vram)}")


def _settings() -> config.Settings:
    """Loaded once per command, and its warnings go to stderr so stdout stays pipeable."""
    settings = config.load()
    config.apply(settings)
    for warning in settings.warnings:
        typer.secho(f"config: {warning}", fg=typer.colors.YELLOW, err=True)
    return settings


def _fail(exc: OllamaError) -> typer.Exit:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


@app.command()
def start(
    model: str = typer.Option(DEFAULT_ALIAS, "--model", "-m", help="Registry alias or raw tag."),
    num_ctx: int | None = typer.Option(None, "--num-ctx", help="Window to pin the model at."),
) -> None:
    """Pin a model in VRAM so the next question is warm rather than a ten second load."""
    settings = _settings()
    window = num_ctx if num_ctx is not None else settings.num_ctx
    try:
        result = lifecycle.start(
            OllamaClient(num_ctx=window), model, keep_alive=settings.keep_alive
        )
    except OllamaError as exc:
        raise _fail(exc) from exc
    if result.already_resident:
        typer.echo(f"{result.model} is already loaded, nothing to do")
    else:
        typer.echo(f"{result.model} ready in {result.seconds:.1f}s")
    _echo_resident(result.resident)
    _echo_vram(result.vram)


@app.command()
def stop(
    model: str = typer.Option("", "--model", "-m", help="Only this one; default releases all."),
) -> None:
    """Release loaded models and give the card back."""
    _settings()
    try:
        result = lifecycle.stop(OllamaClient(), model or None)
    except OllamaError as exc:
        raise _fail(exc) from exc
    if not result.released and not result.still_resident:
        # "nothing was loaded" would be a lie when a different model still holds the card.
        typer.echo(f"{resolve(model).tag} is not loaded" if model else "nothing was loaded")
        return
    for tag in result.released:
        typer.echo(f"released {tag}")
    for tag in result.still_resident:
        typer.secho(
            f"{tag} is still loaded after the unload request", fg=typer.colors.RED, err=True
        )
    if result.still_resident:
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """What is loaded, how much of the card it holds, and how long it stays."""
    _settings()
    try:
        current = lifecycle.status(OllamaClient())
    except OllamaError as exc:
        raise _fail(exc) from exc
    if not current.residents:
        typer.echo("nothing is loaded - run `o start` to pin a model")
    for resident in current.residents:
        typer.echo(resident.name)
        _echo_resident(resident)
    _echo_vram(current.vram)


config_app = typer.Typer(help="Read and write the settings file.")
app.add_typer(config_app, name="config")


@config_app.callback(invoke_without_command=True)
def config_show(ctx: typer.Context) -> None:
    """Print the effective settings and where each value came from."""
    if ctx.invoked_subcommand is not None:
        return
    settings = config.load()
    typer.echo(f"file: {config.config_path()}")
    typer.echo(f"precedence, highest first: {' > '.join(config.PRECEDENCE)}")
    for key in config.DEFAULTS:
        shown = config.shown(key, settings.values[key])
        typer.echo(f"  {key:16s} {shown:32s} [{settings.sources[key]}]")
    for warning in settings.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Write one setting, leaving every other key in the file untouched."""
    try:
        written, notes = config.set_value(key, value)
    except KeyError:
        typer.secho(f"error: unknown key {key!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    except ValueError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    except OSError as exc:
        typer.secho(f"error: writing {config.config_path()}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.echo(f"{key} = {config.shown(key, written)}  -> {config.config_path()}")
    for note in notes:
        typer.secho(f"warning: {note}", fg=typer.colors.YELLOW, err=True)


@config_app.command("unset")
def config_unset(key: str) -> None:
    """Revert one setting to the built-in default by removing it from the file."""
    try:
        removed, dropped = config.unset_value(key)
    except KeyError:
        typer.secho(f"error: unknown key {key!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    except OSError as exc:
        typer.secho(f"error: writing {config.config_path()}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    default = config.shown(key, config.DEFAULTS[key])
    typer.echo(
        f"{key} back to the built-in default: {default}" if removed else f"{key} was not set"
    )
    for note in dropped:
        typer.secho(f"warning: {note}", fg=typer.colors.YELLOW, err=True)


@app.command()
def models() -> None:
    """List the routing table, flagging which models have measurements behind them."""
    _settings()
    roles = {DEFAULT_ALIAS: "bare questions, o start", HEAVY_ALIAS: "o audit"}
    for alias in REGISTRY:
        spec = resolve(alias)
        flag = "measured" if spec.measured else "UNMEASURED"
        role = f"  <- {roles[alias]}" if alias in roles else ""
        typer.echo(f"{alias:10s} {spec.tag:40s} [{flag}] {spec.summary}{role}")


@app.command()
def which(name: str) -> None:
    """Resolve an alias or tag to what would actually run."""
    _settings()
    spec = resolve(name)
    typer.echo(f"{name} -> {spec.tag} ({spec.summary})")
    if not spec.measured:
        typer.secho("no measurements behind this model", fg=typer.colors.YELLOW, err=True)


@app.command()
def setup(
    fast_model: str | None = typer.Option(None, "--fast-model", help="Tag for the fast role."),
    heavy_model: str | None = typer.Option(None, "--heavy-model", help="Tag for the heavy role."),
    search_provider: str | None = typer.Option(None, "--search-provider", help="Search backend."),
    install: bool | None = typer.Option(None, "--install/--no-install", help="Put `o` on PATH."),
    no_pull: bool = typer.Option(False, "--no-pull", help="Choose models but download nothing."),
) -> None:
    """First-run wizard: hardware, models, config, and a verification run on your machine."""
    from ollama_stack import setup as wizard

    answers = wizard.Answers(
        fast_model=fast_model,
        heavy_model=heavy_model,
        search_provider=search_provider,
        install=install,
        pull=not no_pull,
    )
    try:
        code = wizard.run(answers)
    except wizard.MissingAnswerError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    raise typer.Exit(code)


@app.command()
def tutorial() -> None:
    """Nine steps, run for real against your machine. Changes nothing and can be re-run."""
    from ollama_stack import tutorial as guide

    raise typer.Exit(guide.run())


if __name__ == "__main__":
    app()
