"""Tier detection decides what a stranger is offered, so every band gets a test."""

from __future__ import annotations

import pytest

from ollama_stack import hardware
from ollama_stack.hardware import Gpu, detect, shortfall_mib, tier_for
from ollama_stack.lifecycle import Vram


def _blind(monkeypatch: pytest.MonkeyPatch) -> None:
    """No nvidia, no rocm, no sysctl - the state every non-NVIDIA machine is in."""
    monkeypatch.setattr(hardware, "nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_run", lambda command: None)


@pytest.mark.parametrize(
    ("mib", "label"),
    [
        (24463, "24 GB+"),
        (22528, "21-24 GB"),
        (16376, "10-21 GB"),
        (12288, "10-21 GB"),
        (8192, "6-10 GB"),
        (4096, "under 6 GB"),
    ],
)
def test_each_band_catches_the_card_it_is_drawn_for(mib: int, label: str) -> None:
    assert tier_for(Gpu("nvidia", mib, mib, "test")).label == label


def test_a_24gb_card_reporting_23_9_gib_still_lands_in_the_top_band() -> None:
    """Cards report under their nominal size, so a strict boundary demotes every one of them."""
    assert tier_for(Gpu("nvidia", 24463, 24463, "test")).label == "24 GB+"


def test_an_unreadable_card_is_the_cpu_tier_and_not_a_guess() -> None:
    tier = tier_for(Gpu("cpu", None, None, "no GPU detected"))
    assert tier.label == "CPU only / unknown"
    assert tier.heavy == ()


def test_the_two_smallest_bands_offer_no_heavy_model() -> None:
    """X10: a small card must complete setup, not be recommended something that cannot load."""
    for mib in (4096, None):
        assert tier_for(Gpu("nvidia", mib, mib, "test")).heavy == ()


def test_every_band_offers_a_fast_model() -> None:
    for tier in (*hardware.TIERS, hardware.CPU_TIER):
        assert tier.fast.tag
        assert tier.fast.size_bytes > 0


def test_the_heavy_model_fits_at_the_bottom_of_its_own_band() -> None:
    """Caught a 17.4 GB model offered to a 16 GB card - the same shape as the bug it replaced."""
    for tier in hardware.TIERS:
        floor_mib = int(tier.floor_gib * hardware.MIB_PER_GIB)
        for model in tier.heavy:
            assert model.needed_mib <= floor_mib, f"{model.tag} does not fit {tier.label}"


def test_the_fast_model_fits_at_the_bottom_of_its_own_band() -> None:
    """A band that cannot run its own fast model has nothing to offer at all."""
    for tier in hardware.TIERS:
        if tier.floor_gib <= 0:
            continue
        floor_mib = int(tier.floor_gib * hardware.MIB_PER_GIB)
        assert tier.fast.needed_mib <= floor_mib, f"{tier.fast.tag} does not fit {tier.label}"


def test_nvidia_is_preferred_when_it_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "nvidia_vram", lambda: Vram(1000, 24463))
    gpu = detect()
    assert (gpu.source, gpu.total_mib, gpu.free_mib) == ("nvidia", 24463, 23463)


def test_nothing_detected_degrades_to_cpu_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _blind(monkeypatch)
    monkeypatch.setattr("ollama_stack.hardware.platform.system", lambda: "Linux")
    gpu = detect()
    assert (gpu.source, gpu.total_mib) == ("cpu", None)
    assert tier_for(gpu).label == "CPU only / unknown"


def test_junk_from_the_probes_reads_as_unknown_not_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_run", lambda command: "not,a,number\n")
    monkeypatch.setattr("ollama_stack.hardware.platform.system", lambda: "Linux")
    assert detect().total_mib is None


def test_shortfall_counts_the_kv_cache_and_not_just_the_file() -> None:
    """A file size is a floor: the context cache loads beside the weights."""
    model = hardware.Model("test:1b", 4_000_000_000)
    plenty = Gpu("nvidia", 24463, 20000, "test")
    assert shortfall_mib(model, plenty) == 0
    tight = Gpu("nvidia", 6144, 4000, "test")
    assert shortfall_mib(model, tight) > 0


def test_shortfall_is_zero_when_there_is_nothing_to_compare_against() -> None:
    model = hardware.Model("test:1b", 4_000_000_000)
    assert shortfall_mib(model, Gpu("cpu", None, None, "unknown")) == 0


def test_a_real_pulled_size_overrides_the_table_estimate() -> None:
    """The table is what to offer; /api/tags is what decides, once the model is actually here."""
    model = hardware.Model("test:1b", 1_000_000_000)
    card = Gpu("nvidia", 8192, 2000, "test")
    assert shortfall_mib(model, card) == 0
    assert shortfall_mib(model, card, size_bytes=9_000_000_000) > 0


MIB = 1024 * 1024


def _rocm_csv(cards: list[tuple[int, int]]) -> str:
    """rocm-smi reports bytes, one line per card."""
    rows = [f"card{i}, {total * MIB}, {used * MIB}" for i, (total, used) in enumerate(cards)]
    return chr(10).join(rows) + chr(10)


def test_two_amd_cards_are_read_one_at_a_time_not_mixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max total with min used paired the biggest card's size to the emptiest card's usage."""
    cards = [(8192, 100), (24560, 23000)]
    monkeypatch.setattr(hardware, "_run", lambda command: _rocm_csv(cards))
    vram = hardware._rocm_vram()
    assert vram is not None
    # 8092 is what the idle card really has. 24460 was the old answer and no card had it.
    assert vram.free_mib == 8092


def test_the_amd_card_with_the_most_room_is_the_one_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model loads onto one card, so the useful figure is the best single card."""
    cards = [(24560, 765), (16368, 11444)]
    monkeypatch.setattr(hardware, "_run", lambda command: _rocm_csv(cards))
    vram = hardware._rocm_vram()
    assert vram is not None
    assert (vram.total_mib, vram.used_mib) == (24560, 765)


def test_amd_output_with_no_numbers_is_unknown_rather_than_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noise = "device, name" + chr(10) + "card0, gfx1100" + chr(10)
    monkeypatch.setattr(hardware, "_run", lambda command: noise)
    assert hardware._rocm_vram() is None
