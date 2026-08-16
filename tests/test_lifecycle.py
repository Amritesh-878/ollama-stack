"""Every input here is a system we do not own, so the tests are mostly about degrading."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import requests
import responses

from ollama_stack import lifecycle
from ollama_stack.client import OllamaClient, OllamaError
from ollama_stack.lifecycle import Resident, Vram, VramShortfallError

GENERATE = "http://127.0.0.1:11434/api/generate"
PS = "http://127.0.0.1:11434/api/ps"
TAGS = "http://127.0.0.1:11434/api/tags"

# Captured from ollama 0.32.13 with one model pinned, fields and all.
PS_ENTRY: dict[str, Any] = {
    "name": "qwen3.8:27b",
    "model": "qwen3.8:27b",
    "size": 17179869184,
    "size_vram": 17179869184,
    "context_length": 32768,
    "expires_at": "2318-11-26T18:04:46.425846407+05:30",
}


def _posts() -> list[Any]:
    return [call for call in responses.calls if call.request.method == "POST"]


def _smi(monkeypatch: pytest.MonkeyPatch, stdout: str, exc: Exception | None = None) -> None:
    def fake(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake)


def test_nvidia_smi_output_becomes_a_card_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "21348, 24463\n")
    vram = lifecycle.nvidia_vram()
    assert vram == Vram(21348, 24463)
    assert vram.free_mib == 3115


def test_a_missing_nvidia_smi_is_unknown_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "", exc=FileNotFoundError("nvidia-smi"))
    assert lifecycle.nvidia_vram() is None


def test_a_timed_out_nvidia_smi_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "", exc=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5))
    assert lifecycle.nvidia_vram() is None


def test_a_failing_nvidia_smi_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "", exc=subprocess.CalledProcessError(returncode=9, cmd="nvidia-smi"))
    assert lifecycle.nvidia_vram() is None


@pytest.mark.parametrize("junk", ["", "no GPU here\n", "21348\n", "a, b\n", "21348, \n"])
def test_junk_from_nvidia_smi_is_unknown_rather_than_a_wrong_number(
    monkeypatch: pytest.MonkeyPatch, junk: str
) -> None:
    _smi(monkeypatch, junk)
    assert lifecycle.nvidia_vram() is None


def test_a_pinned_model_reads_as_pinned_not_as_three_hundred_years() -> None:
    assert Resident("m", 100, 100, 32768, PS_ENTRY["expires_at"]).ttl == "pinned"


def test_an_unparseable_expiry_is_unknown() -> None:
    assert Resident("m", 100, 100, 0, "whenever").ttl == "unknown"
    assert Resident("m", 100, 100, 0, "").ttl == "unknown"


def test_an_expiry_already_past_reads_as_expired() -> None:
    assert Resident("m", 100, 100, 0, "2020-01-01T00:00:00Z").ttl == "expired"


def test_nanosecond_precision_parses_even_where_fromisoformat_would_refuse() -> None:
    assert lifecycle._parsed_time("2026-08-16T12:47:29.563296712Z") is not None


def test_the_processor_split_comes_from_the_two_sizes() -> None:
    assert Resident("m", 1000, 400, 0, "").gpu_percent == 40
    assert Resident("m", 0, 0, 0, "").gpu_percent is None


@responses.activate
def test_start_pins_the_model_through_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "4000, 24463\n")
    responses.add(responses.GET, PS, json={"models": []})
    responses.add(responses.GET, TAGS, json={"models": [{"name": "qwen3.8:27b", "size": 100}]})
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "load"})
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    result = lifecycle.start(OllamaClient(), "qwen")
    load = next(iter(_posts()))
    assert '"keep_alive": -1' in str(load.request.body)
    assert result.already_resident is False
    assert result.resident is not None and result.resident.name == "qwen3.8:27b"


@responses.activate
def test_start_on_an_already_resident_model_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "21348, 24463\n")
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    result = lifecycle.start(OllamaClient(), "qwen")
    assert result.already_resident is True
    assert not _posts()


@responses.activate
def test_start_refuses_and_names_what_holds_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "23000, 24463\n")
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    responses.add(
        responses.GET, TAGS, json={"models": [{"name": "qwen3-coder:30b", "size": 17179869184}]}
    )
    with pytest.raises(VramShortfallError, match=re.escape("qwen3.8:27b")):
        lifecycle.start(OllamaClient(), "coder")
    assert not _posts()


@responses.activate
def test_start_proceeds_when_the_card_is_unreadable_rather_than_refusing_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smi(monkeypatch, "", exc=FileNotFoundError("nvidia-smi"))
    responses.add(responses.GET, PS, json={"models": []})
    responses.add(responses.GET, TAGS, json={"models": [{"name": "qwen3.8:27b", "size": 100}]})
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "load"})
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    result = lifecycle.start(OllamaClient(), "qwen")
    assert result.vram is None
    assert result.already_resident is False


@responses.activate
def test_stop_unloads_and_confirms_from_ps(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "4000, 24463\n")
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "unload"})
    responses.add(responses.GET, PS, json={"models": []})
    result = lifecycle.stop(OllamaClient())
    unload = next(iter(_posts()))
    assert '"keep_alive": 0' in str(unload.request.body)
    assert result.released == ["qwen3.8:27b"]
    assert result.still_resident == []


@responses.activate
def test_stop_reports_a_model_that_did_not_go(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "4000, 24463\n")
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    responses.add(responses.POST, GENERATE, json={"done": True, "done_reason": "unload"})
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    result = lifecycle.stop(OllamaClient())
    assert result.released == []
    assert result.still_resident == ["qwen3.8:27b"]


@responses.activate
def test_stop_with_nothing_loaded_sends_no_unload() -> None:
    responses.add(responses.GET, PS, json={"models": []})
    result = lifecycle.stop(OllamaClient())
    assert (result.released, result.still_resident) == ([], [])
    assert not _posts()


@responses.activate
def test_status_parses_a_real_ps_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "21348, 24463\n")
    responses.add(responses.GET, PS, json={"models": [PS_ENTRY]})
    current = lifecycle.status(OllamaClient())
    assert len(current.residents) == 1
    resident = current.residents[0]
    assert resident.name == "qwen3.8:27b"
    assert (resident.context_length, resident.ttl) == (32768, "pinned")
    assert resident.gpu_percent == 100


@responses.activate
def test_status_with_nothing_loaded_is_an_empty_list_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smi(monkeypatch, "1000, 24463\n")
    responses.add(responses.GET, PS, json={"models": []})
    assert lifecycle.status(OllamaClient()).residents == []


@responses.activate
def test_a_ps_body_of_an_unexpected_shape_is_empty_rather_than_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smi(monkeypatch, "1000, 24463\n")
    responses.add(responses.GET, PS, json={"models": "not a list"})
    assert lifecycle.status(OllamaClient()).residents == []


@responses.activate
def test_a_ps_entry_missing_every_field_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smi(monkeypatch, "1000, 24463\n")
    responses.add(responses.GET, PS, json={"models": [{}]})
    resident = lifecycle.status(OllamaClient()).residents[0]
    assert (resident.name, resident.size_mib, resident.ttl) == ("unknown", 0, "unknown")


@responses.activate
def test_ollama_being_down_is_a_message_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, "1000, 24463\n")
    responses.add(responses.GET, PS, body=requests.ConnectionError("refused"))
    with pytest.raises(OllamaError, match="failed against"):
        lifecycle.status(OllamaClient())


@responses.activate
def test_a_ps_reply_that_is_not_json_is_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smi(monkeypatch, "1000, 24463\n")
    responses.add(responses.GET, PS, body="<html>proxy error</html>", status=200)
    with pytest.raises(OllamaError, match="failed against"):
        lifecycle.status(OllamaClient())


def test_lifecycle_never_opens_its_own_connection_to_ollama() -> None:
    """The num_ctx defect was acquired three times by call sites that skipped the client."""
    source = Path(lifecycle.__file__).read_text(encoding="utf-8")
    assert "requests" not in source
    assert "11434" not in source
