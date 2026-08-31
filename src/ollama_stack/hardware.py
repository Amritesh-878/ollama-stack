"""What the machine can actually run, which is what decides everything the wizard offers."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

from ollama_stack.binaries import on_path
from ollama_stack.lifecycle import MIB, Vram, nvidia_vram

PROBE_TIMEOUT = 5
MIB_PER_GIB = 1024
# The KV cache loads beside the weights, so a file size is a floor on cost and never a prediction.
VRAM_OVERHEAD = 1.25
# A 24 GB card reads 23.9 GiB, so a band needs slack or every card lands one tier too low.
BAND_SLACK_GIB = 0.5


@dataclass(frozen=True)
class Model:
    """One offerable model and its download size, verified against Ollama's library."""

    tag: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def needed_mib(self) -> int:
        return int(self.size_bytes / MIB * VRAM_OVERHEAD)


@dataclass(frozen=True)
class Tier:
    """One VRAM band and what it can hold, drawn so heavy fits at the bottom of the band."""

    label: str
    floor_gib: float
    fast: Model
    heavy: tuple[Model, ...]


@dataclass(frozen=True)
class Gpu:
    """Detected hardware, where unknown is a real answer and never a zero."""

    source: str
    total_mib: int | None
    free_mib: int | None
    detail: str

    @property
    def total_gib(self) -> float | None:
        if self.total_mib is None:
            return None
        return self.total_mib / MIB_PER_GIB


QWEN_08B = Model("qwen3.5:0.8b", 1_040_000_000)
QWEN_2B = Model("qwen3.5:2b", 2_740_000_000)
QWEN_4B = Model("qwen3.5:4b", 3_390_000_000)
QWEN_9B = Model("qwen3.5:9b", 6_590_000_000)
QWEN_35_27B = Model("qwen3.5:27b", 17_420_000_000)
QWEN_38_27B = Model("qwen3.8:27b", 17_740_000_000)
QWEN_CODER = Model("qwen3-coder:30b", 18_560_000_000)

# Ordered high to low: the first band a card clears is the one it belongs to. Every heavy model
# here fits at its band's FLOOR, which is asserted by a test rather than left to the eye.
TIERS: tuple[Tier, ...] = (
    Tier("24 GB+", 24.0, QWEN_4B, (QWEN_38_27B, QWEN_CODER)),
    Tier("21-24 GB", 21.0, QWEN_4B, (QWEN_35_27B,)),
    Tier("10-21 GB", 10.0, QWEN_4B, (QWEN_9B,)),
    Tier("6-10 GB", 6.0, QWEN_2B, (QWEN_4B,)),
    Tier("under 6 GB", 0.0, QWEN_2B, ()),
)

CPU_TIER = Tier("CPU only / unknown", -1.0, QWEN_08B, ())


def _run(command: list[str]) -> str | None:
    """The probe is resolved to a real path first: a bare name would find the repo it is run in."""
    binary = on_path(command[0])
    if binary is None:
        return None
    try:
        done = subprocess.run(
            [binary, *command[1:]], capture_output=True, text=True,
            timeout=PROBE_TIMEOUT, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def _rocm_vram() -> Vram | None:
    """AMD reports bytes here, unlike nvidia-smi's MiB.

    One card at a time. Taking max total and min used across the whole file paired the
    biggest card's size with the emptiest card's usage: two cards at 8 GB idle and 24 GB
    nearly full reported 24460 MiB free when no card had more than 8092.
    """
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not out:
        return None
    cards: list[Vram] = []
    for line in out.splitlines():
        numbers = [int(f) for f in line.split(",") if f.strip().lstrip("-").isdigit()]
        if len(numbers) < 2:
            continue
        total, used = max(numbers) // MIB, min(numbers) // MIB
        if total > 0:
            cards.append(Vram(used, total))
    if not cards:
        return None
    # The card with the most room, because a model loads onto one card, not onto the sum.
    return max(cards, key=lambda card: card.free_mib)


def _metal_total_mib() -> int | None:
    """Apple shares one pool between CPU and GPU, so system memory is the ceiling."""
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out or not out.strip().isdigit():
        return None
    return int(out.strip()) // MIB


def detect() -> Gpu:
    """nvidia-smi, then ROCm, then Metal, then CPU - and unknown degrades, it does not crash."""
    nvidia = nvidia_vram()
    if nvidia is not None:
        return Gpu("nvidia", nvidia.total_mib, nvidia.free_mib, "nvidia-smi")
    rocm = _rocm_vram()
    if rocm is not None:
        return Gpu("rocm", rocm.total_mib, rocm.free_mib, "rocm-smi")
    if platform.system() == "Darwin":
        total = _metal_total_mib()
        if total is not None:
            # Half the shared pool, because the OS and every other process want the rest.
            return Gpu("metal", total // 2, total // 2, "apple unified memory, halved")
    return Gpu("cpu", None, None, "no GPU detected")


def tier_for(gpu: Gpu) -> Tier:
    """An unreadable card is the CPU tier, never a guess at the band it might have been."""
    gib = gpu.total_gib
    if gib is None:
        return CPU_TIER
    for tier in TIERS:
        if gib >= tier.floor_gib - BAND_SLACK_GIB:
            return tier
    return CPU_TIER


def shortfall_mib(model: Model, gpu: Gpu, size_bytes: int | None = None) -> int:
    """How far the card falls short, or 0 when it fits or when there is nothing to compare."""
    if gpu.free_mib is None:
        return 0
    needed = model.needed_mib if size_bytes is None else int(size_bytes / MIB * VRAM_OVERHEAD)
    return max(0, needed - gpu.free_mib)
