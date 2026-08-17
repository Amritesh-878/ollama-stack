"""The first-run wizard: hardware, models, config, and a verification run on their machine."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ollama_stack import config
from ollama_stack.client import OllamaClient, OllamaError, usable_window
from ollama_stack.hardware import Gpu, Model, Tier, detect, shortfall_mib, tier_for

PULL_TIMEOUT = 3600
PROBE_TIMEOUT = 5
NO_HEAVY = "none"
RATE_FLOOR = 20
VERIFY_PROMPT = "List the eight planets of the solar system, one per line, nothing else."

INSTALL_HINT = {
    "Windows": "winget install Ollama.Ollama    (or download from https://ollama.com/download)",
    "Darwin": "brew install ollama    (or download from https://ollama.com/download)",
    "Linux": "curl -fsSL https://ollama.com/install.sh | sh",
}


class SetupError(Exception):
    """Something the user must fix before the wizard can continue."""


class MissingAnswerError(SetupError):
    """No terminal to ask on and no flag supplied, so the flag gets named rather than guessed."""

    def __init__(self, flag: str, question: str) -> None:
        super().__init__(f"no terminal to ask '{question}'. Pass {flag} instead.")
        self.flag = flag


@dataclass
class Answers:
    """Every question the wizard can ask, each with a flag that supplies it non-interactively."""

    fast_model: str | None = None
    heavy_model: str | None = None
    search_provider: str | None = None
    install: bool | None = None
    pull: bool = True


@dataclass
class Step:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    """Reporting is its own concern, so a step that fails does not need to know how to say so."""

    steps: list[Step] = field(default_factory=list)
    path_changed: bool = False

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append(Step(name, ok, detail))

    @property
    def failures(self) -> list[Step]:
        return [step for step in self.steps if not step.ok]


def ascii_only() -> bool:
    """cmd.exe is cp1252 and renders box drawing as T, | and o, so ask before drawing any."""
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" not in encoding


def say(message: str = "") -> None:
    print(message, flush=True)


def rule(width: int = 60) -> None:
    say(("-" if ascii_only() else "─") * width)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _confirm(question: str, flag: str, default: bool, given: bool | None) -> bool:
    if given is not None:
        return given
    if not interactive():
        raise MissingAnswerError(flag, question)
    answer = input(f"{question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def _choose_one(
    question: str, flag: str, options: list[str], default: str, given: str | None
) -> str:
    if given is not None:
        return given
    if not interactive():
        raise MissingAnswerError(flag, question)
    try:
        import questionary

        picked = questionary.select(question, choices=options, default=default).ask()
    except (ImportError, OSError):
        picked = _numbered(question, options, default)
    return picked or default


def _numbered(question: str, options: list[str], default: str) -> str:
    """The fallback when the terminal cannot host a prompt_toolkit screen."""
    say(question)
    for index, option in enumerate(options, 1):
        say(f"  {index}) {option}")
    raw = input(f"Number [{options.index(default) + 1}]: ").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= len(options):
        return default
    return options[int(raw) - 1]


def ollama_binary() -> str | None:
    return shutil.which("ollama")


def install_hint() -> str:
    return INSTALL_HINT.get(platform.system(), "see https://ollama.com/download")


def daemon_reachable(client: OllamaClient) -> bool:
    try:
        client.tags()
    except OllamaError:
        return False
    return True


def start_daemon() -> bool:
    """Ollama serve holds the terminal, so it is spawned detached and then re-probed."""
    binary = ollama_binary()
    if binary is None:
        return False
    try:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    time.sleep(2)
    return True


def local_tags(client: OllamaClient) -> dict[str, int]:
    """Tag to byte size for everything already pulled, so nothing is offered twice."""
    found: dict[str, int] = {}
    try:
        entries = client.tags()
    except OllamaError:
        return found
    for entry in entries:
        name = str(entry.get("name") or entry.get("model") or "")
        if name:
            found[name] = int(entry.get("size", 0))
    return found


def offered(tier: Tier) -> list[Model]:
    seen: dict[str, Model] = {tier.fast.tag: tier.fast}
    for model in tier.heavy:
        seen.setdefault(model.tag, model)
    return list(seen.values())


def describe(model: Model, present: bool, gpu: Gpu, real_size: int | None) -> str:
    """Every offer says its size, whether it is already here, and whether it will actually fit."""
    if present:
        return f"{model.tag}  (already downloaded)"
    short = shortfall_mib(model, gpu, real_size)
    warning = f"  NEEDS {short} MiB MORE THAN IS FREE" if short else ""
    return f"{model.tag}  {model.size_gb:.1f} GB download{warning}"


def choose_models(
    tier: Tier, gpu: Gpu, local: dict[str, int], answers: Answers
) -> tuple[str, str | None]:
    """Fast and heavy are separate questions because they map to separate config keys and flags."""
    candidates = offered(tier)
    fast_options = [m.tag for m in candidates]
    fast = _choose_one(
        "Fast model - answers your questions and is what `o start` pins:",
        "--fast-model",
        fast_options,
        tier.fast.tag,
        answers.fast_model,
    )
    heavy_options = [m.tag for m in tier.heavy] + [NO_HEAVY]
    default_heavy = tier.heavy[0].tag if tier.heavy else NO_HEAVY
    if not tier.heavy and answers.heavy_model is None:
        say(f"  There is no heavy model for {tier.label} - nothing that size would fit.")
        return fast, None
    heavy = _choose_one(
        "Heavy model - for `o audit` and anything you would rather wait for:",
        "--heavy-model",
        heavy_options,
        default_heavy,
        answers.heavy_model,
    )
    return fast, None if heavy == NO_HEAVY else heavy


def to_pull(wanted: list[str], local: dict[str, int], catalogue: dict[str, Model]) -> list[Model]:
    return [catalogue[tag] for tag in wanted if tag in catalogue and tag not in local]


def pull(tag: str, report: Report) -> bool:
    """A failed pull is not a failed setup, so this reports and the caller carries on."""
    binary = ollama_binary()
    if binary is None:
        report.add(f"pull {tag}", False, "ollama is not on PATH")
        return False
    say(f"  pulling {tag} ...")
    try:
        done = subprocess.run([binary, "pull", tag], timeout=PULL_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        report.add(f"pull {tag}", False, str(exc))
        return False
    ok = done.returncode == 0
    report.add(f"pull {tag}", ok, "" if ok else f"ollama pull exited {done.returncode}")
    return ok


def write_config(fast: str, heavy: str | None, provider: str | None, report: Report) -> None:
    """Through the config module, never by hand - it owns the file format and the validation."""
    try:
        config.set_value("fast_model", fast)
        if heavy is not None:
            config.set_value("heavy_model", heavy)
        if provider is not None:
            config.set_value("search_provider", provider)
    except (OSError, KeyError, ValueError) as exc:
        report.add("write config", False, str(exc))
        return
    report.add("write config", True, str(config.config_path()))


def verify(client: OllamaClient, tag: str, report: Report) -> None:
    """Their cold load and their warm reply, because every figure we publish is from one laptop."""
    say(f"  loading {tag} ...")
    started = time.perf_counter()
    try:
        client.load(tag)
    except OllamaError as exc:
        report.add("verify", False, str(exc))
        return
    cold = time.perf_counter() - started
    started = time.perf_counter()
    try:
        reply = client.generate(VERIFY_PROMPT, tag)
    except OllamaError as exc:
        report.add("verify", False, str(exc))
        return
    warm = time.perf_counter() - started
    # Released again: a pin never expires, and setup must not leave the card held silently.
    client.unload(tag)
    measured = f"cold load {cold:.1f}s, warm reply {warm:.2f}s"
    # A rate off a handful of tokens is fixed cost, not throughput, and reads as a broken machine.
    if reply.eval_count >= RATE_FLOOR and warm > 0:
        measured += f", {reply.eval_count / warm:.0f} tok/s"
    report.add("verify", True, measured)
    say(f"  {measured}")


def repo_root() -> Path | None:
    """None when running from a wheel, where there is no working tree to install editable from."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else None


def resolve_on_path() -> str | None:
    return shutil.which("o")


def global_install(report: Report) -> None:
    """--editable on purpose: a plain install snapshots the code and git pull then changes none."""
    uv = shutil.which("uv")
    root = repo_root()
    if uv is None or root is None:
        report.add("global install", False, "needs uv and a working tree; `uv run o` still works")
        return
    before = resolve_on_path()
    try:
        done = subprocess.run(
            [uv, "tool", "install", "--editable", "."],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        report.add("global install", False, str(exc))
        return
    if done.returncode != 0:
        report.add("global install", False, (done.stderr or done.stdout).strip())
        return
    found = resolve_on_path()
    report.path_changed = found != before
    if found is None:
        subprocess.run([uv, "tool", "update-shell"], cwd=root, capture_output=True, check=False)
        report.path_changed = True
        report.add("global install", True, "installed; `o` needs a new terminal to appear on PATH")
        return
    report.add("global install", True, f"`o` resolves at {found}")


def closing(report: Report, settings_num_ctx: int) -> None:
    """One instruction, and it is the one that proves PATH, the install and the config at once."""
    say()
    rule()
    for step in report.steps:
        mark = "ok  " if step.ok else "FAIL"
        say(f"  [{mark}] {step.name}{f' - {step.detail}' if step.detail else ''}")
    rule()
    say()
    if report.failures:
        say(f"{len(report.failures)} step(s) did not complete. Everything else is ready.")
        say()
    say(f"Attached files and pipes are refused past {usable_window(settings_num_ctx)} tokens.")
    say("  Raise it for one command with --num-ctx, or for good with `o config set num_ctx`.")
    say()
    if report.path_changed:
        say("Setup done. Open a new terminal and type:  o tutorial")
    else:
        say("Setup done. Type:  o tutorial")


def run(answers: Answers) -> int:
    """The interview and the actions are kept apart so a no-TTY run fails on a flag, not midway."""
    report = Report()
    say("ollama-stack setup")
    rule(18)
    say()

    if ollama_binary() is None:
        say("Ollama is not installed, and this tool does not install it for you.")
        say(f"  {install_hint()}")
        return 1

    client = OllamaClient()
    if not daemon_reachable(client):
        say("Ollama is installed but not answering on 127.0.0.1:11434.")
        if _confirm("Start it now?", "--no-install", True, None if interactive() else False):
            start_daemon()
        if not daemon_reachable(client):
            say("  Still not reachable. Start it with `ollama serve` and re-run `o setup`.")
            return 1
    report.add("ollama", True, "installed and reachable")

    gpu = detect()
    tier = tier_for(gpu)
    vram = f"{gpu.total_mib} MiB" if gpu.total_mib is not None else "unknown"
    say(f"Hardware: {gpu.detail}, {vram} -> tier {tier.label}")
    report.add("hardware", True, f"{gpu.source}, {vram}, tier {tier.label}")
    say()

    local = local_tags(client)
    catalogue = {m.tag: m for m in offered(tier)}
    for model in offered(tier):
        say(f"  {describe(model, model.tag in local, gpu, local.get(model.tag))}")
    say()

    fast, heavy = choose_models(tier, gpu, local, answers)
    wanted = [tag for tag in (fast, heavy) if tag]
    pending = to_pull(wanted, local, catalogue)
    if pending and answers.pull:
        total = sum(m.size_gb for m in pending)
        say(f"To download: {', '.join(m.tag for m in pending)} - {total:.1f} GB total")
        for model in pending:
            short = shortfall_mib(model, gpu, None)
            if short:
                say(f"  warning: {model.tag} needs {short} MiB more than is free right now.")
            pull(model.tag, report)
    elif pending:
        report.add("pull", False, f"skipped by --no-pull: {', '.join(m.tag for m in pending)}")

    provider = answers.search_provider
    write_config(fast, heavy, provider, report)

    settings = config.load()
    config.apply(settings)
    verify(client, fast, report)

    install = answers.install
    if install is None and interactive():
        install = _confirm("Put `o` on PATH everywhere?", "--install", True, None)
    if install:
        global_install(report)
    elif install is False:
        report.add("global install", True, "declined; `uv run o` works from the repo")

    closing(report, settings.num_ctx)
    return 0
