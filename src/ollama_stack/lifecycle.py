"""Whether a model is resident, which is the whole difference between a 1s answer and a 14s one."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ollama_stack.binaries import on_path
from ollama_stack.client import PIN, OllamaClient, OllamaError
from ollama_stack.models import resolve

SMI = "nvidia-smi"
SMI_TIMEOUT = 5
MIB = 1024 * 1024
PINNED_AFTER_DAYS = 365
UNLOAD_CHECKS = 5
UNLOAD_SETTLE_SECONDS = 0.3


class VramShortfallError(OllamaError):
    """The card cannot hold the model, and loading it anyway evicts whatever is already there."""


@dataclass(frozen=True)
class Vram:
    """What the driver says about the card, which is not what Ollama says about a model."""

    used_mib: int
    total_mib: int

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)


@dataclass(frozen=True)
class Resident:
    """One loaded model as /api/ps describes it, with every field allowed to be missing."""

    name: str
    size_mib: int
    vram_mib: int
    context_length: int
    expires_at: str

    @property
    def gpu_percent(self) -> int | None:
        if self.size_mib <= 0:
            return None
        return round(100 * self.vram_mib / self.size_mib)

    @property
    def ttl(self) -> str:
        """A pinned model gets a year-2318 expiry rather than any field saying it is pinned."""
        expiry = _parsed_time(self.expires_at)
        if expiry is None:
            return "unknown"
        seconds = int((expiry - datetime.now(UTC)).total_seconds())
        if seconds > PINNED_AFTER_DAYS * 86400:
            return "pinned"
        if seconds <= 0:
            return "expired"
        return f"{seconds // 60}m{seconds % 60:02d}s"


@dataclass(frozen=True)
class StartResult:
    model: str
    already_resident: bool
    seconds: float
    resident: Resident | None
    vram: Vram | None


@dataclass(frozen=True)
class StopResult:
    released: list[str]
    still_resident: list[str]


@dataclass(frozen=True)
class Status:
    residents: list[Resident]
    vram: Vram | None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mib(value: Any) -> int:
    return _int(value) // MIB


def _six_digit_fraction(raw: str) -> str:
    match = re.search(r"\.(\d{7,})", raw)
    if match is None:
        return raw
    return raw.replace(match.group(0), "." + match.group(1)[:6], 1)


def _parsed_time(raw: str) -> datetime | None:
    """Ollama sends nanoseconds, which fromisoformat rejected before 3.13."""
    if not raw:
        return None
    for candidate in (raw, _six_digit_fraction(raw)):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def nvidia_vram() -> Vram | None:
    """None when nvidia-smi is absent or says something unparseable, which is a normal state."""
    binary = on_path(SMI)
    if binary is None:
        return None
    try:
        done = subprocess.run(
            [binary, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = done.stdout.strip().splitlines()
    if not lines:
        return None
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2 or not all(field.isdigit() for field in fields):
        return None
    return Vram(int(fields[0]), int(fields[1]))


def residents(client: OllamaClient) -> list[Resident]:
    """Parses /api/ps defensively - the shape is Ollama's, not ours."""
    return [
        Resident(
            name=str(entry.get("name", "unknown")),
            size_mib=_mib(entry.get("size")),
            vram_mib=_mib(entry.get("size_vram")),
            context_length=_int(entry.get("context_length")),
            expires_at=str(entry.get("expires_at", "")),
        )
        for entry in client.ps()
    ]


def _pulled_size_mib(client: OllamaClient, tag: str) -> int | None:
    """The file size, which is a floor on what loading costs and never a prediction of it."""
    for entry in client.tags():
        if tag in (entry.get("name"), entry.get("model")):
            return _mib(entry.get("size"))
    return None


def _refuse_if_it_cannot_fit(client: OllamaClient, tag: str, before: list[Resident]) -> None:
    vram = nvidia_vram()
    needed = _pulled_size_mib(client, tag)
    if vram is None or needed is None or vram.free_mib >= needed:
        return
    holding = ", ".join(f"{r.name} ({r.vram_mib} MiB)" for r in before) or "nothing from Ollama"
    raise VramShortfallError(
        f"{tag} needs at least {needed} MiB and only {vram.free_mib} of {vram.total_mib} MiB "
        f"is free. Holding the card: {holding}. The real requirement is higher than the file "
        "size, because the context cache loads alongside the weights."
    )


def start(client: OllamaClient, alias: str, keep_alive: int = PIN) -> StartResult:
    """Pins a model, refusing first if the card plainly cannot hold it."""
    tag = resolve(alias).tag
    before = residents(client)
    already = next((r for r in before if r.name == tag), None)
    if already is not None:
        return StartResult(tag, True, 0.0, already, nvidia_vram())
    _refuse_if_it_cannot_fit(client, tag, before)
    # A load-only reply carries no durations at all, so time to ready is the caller's wall clock.
    started = time.perf_counter()
    client.load(alias, keep_alive=keep_alive)
    seconds = time.perf_counter() - started
    after = next((r for r in residents(client) if r.name == tag), None)
    return StartResult(tag, False, seconds, after, nvidia_vram())


def _settled(client: OllamaClient, targets: set[str]) -> set[str]:
    """Ollama drops the model a moment after acknowledging, so one immediate read reports a lie."""
    left = targets
    for attempt in range(UNLOAD_CHECKS):
        left = {r.name for r in residents(client)} & targets
        if not left:
            return left
        if attempt + 1 < UNLOAD_CHECKS:
            time.sleep(UNLOAD_SETTLE_SECONDS)
    return left


def stop(client: OllamaClient, alias: str | None = None) -> StopResult:
    """Verified from /api/ps, because the request being accepted is not the model being gone."""
    before = {r.name for r in residents(client)}
    wanted = [resolve(alias).tag] if alias else sorted(before)
    targets = [tag for tag in wanted if tag in before]
    for tag in targets:
        client.unload(tag)
    left = _settled(client, set(targets))
    return StopResult(
        released=[tag for tag in targets if tag not in left],
        still_resident=[tag for tag in targets if tag in left],
    )


def status(client: OllamaClient) -> Status:
    """Nothing loaded is a normal answer here, not an error."""
    return Status(residents=residents(client), vram=nvidia_vram())
