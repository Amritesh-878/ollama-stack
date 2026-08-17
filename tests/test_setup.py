"""The wizard runs on machines we cannot see, so every failure path is tested and none is live."""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from ollama_stack import config, setup
from ollama_stack.hardware import Gpu, Model
from ollama_stack.setup import Answers, MissingAnswerError, Report
from ollama_stack.setup import verify as real_verify

FULL = Answers(
    fast_model="qwen3.5:4b",
    heavy_model="qwen3.8:27b",
    search_provider="duckduckgo",
    install=False,
    pull=False,
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Never touch the real config, the real card, the real daemon or the real network."""
    for key in config.DEFAULTS:
        monkeypatch.delenv(config.ENV_PREFIX + key.upper(), raising=False)
    monkeypatch.setenv(config.PATH_ENV, str(tmp_path / "config.toml"))
    monkeypatch.setattr(setup, "ollama_binary", lambda: "ollama")
    monkeypatch.setattr(setup, "daemon_reachable", lambda client: True)
    monkeypatch.setattr(setup, "detect", lambda: Gpu("nvidia", 24463, 23000, "nvidia-smi"))
    monkeypatch.setattr(setup, "local_tags", lambda client: {})
    monkeypatch.setattr(setup, "verify", lambda client, tag, report: report.add("verify", True))
    monkeypatch.setattr(setup, "interactive", lambda: False)
    yield


def test_every_flag_supplied_skips_the_interview_entirely() -> None:
    """No terminal and no question asked, which is what makes the wizard scriptable."""
    assert setup.run(FULL) == 0
    assert config.load().values["fast_model"] == "qwen3.5:4b"


def test_a_missing_answer_with_no_tty_names_the_flag_that_supplies_it() -> None:
    with pytest.raises(MissingAnswerError) as caught:
        setup.run(Answers(install=False, pull=False))
    assert caught.value.flag == "--fast-model"
    assert "--fast-model" in str(caught.value)


def test_a_missing_heavy_answer_names_its_own_flag() -> None:
    with pytest.raises(MissingAnswerError) as caught:
        setup.run(Answers(fast_model="qwen3.5:4b", install=False, pull=False))
    assert caught.value.flag == "--heavy-model"


def test_no_prompt_library_is_imported_on_a_flag_driven_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hot path must not pay for questionary, so a scripted run must not reach it."""
    monkeypatch.delitem(sys.modules, "questionary", raising=False)
    setup.run(FULL)
    assert "questionary" not in sys.modules


def test_missing_ollama_names_the_install_command_and_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(setup, "ollama_binary", lambda: None)
    assert setup.run(FULL) == 1
    # Asserting a literal URL made this pass on Windows and fail on Linux, where the hint is
    # the install script rather than the download page.
    assert setup.install_hint() in capsys.readouterr().out


def test_every_platform_has_an_install_hint_that_names_ollama() -> None:
    """CI runs Linux and the dev machine is Windows, so a per-platform string needs all three."""
    for system in ("Windows", "Darwin", "Linux"):
        assert "ollama" in setup.INSTALL_HINT[system].lower(), system


def test_an_unreachable_daemon_is_reported_separately_from_a_missing_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(setup, "daemon_reachable", lambda client: False)
    monkeypatch.setattr(setup, "start_daemon", lambda: False)
    assert setup.run(FULL) == 1
    out = capsys.readouterr().out
    assert "installed but not answering" in out
    assert "ollama serve" in out


def test_the_tier_and_the_vram_that_decided_it_are_both_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A user who disagrees with the recommendation needs to see what produced it."""
    setup.run(FULL)
    out = capsys.readouterr().out
    assert "24463 MiB" in out
    assert "24 GB+" in out


def test_a_small_card_completes_and_says_there_is_no_heavy_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """X10: under 6 GB finishes with a fast model, not a recommendation that cannot run."""
    monkeypatch.setattr(setup, "detect", lambda: Gpu("nvidia", 4096, 3800, "nvidia-smi"))
    answers = Answers(fast_model="qwen3.5:2b", install=False, pull=False)
    assert setup.run(answers) == 0
    assert "no heavy model" in capsys.readouterr().out


def test_a_cpu_only_machine_completes_and_says_there_is_no_heavy_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(setup, "detect", lambda: Gpu("cpu", None, None, "no GPU detected"))
    answers = Answers(fast_model="qwen3.5:0.8b", install=False, pull=False)
    assert setup.run(answers) == 0
    assert "no heavy model" in capsys.readouterr().out


def test_an_already_pulled_model_is_not_offered_for_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = {"qwen3.5:4b": 3_390_000_000}
    catalogue = {"qwen3.5:4b": Model("qwen3.5:4b", 3_390_000_000)}
    assert setup.to_pull(["qwen3.5:4b"], local, catalogue) == []


def test_a_model_that_is_present_says_so_instead_of_a_size() -> None:
    model = Model("qwen3.5:4b", 3_390_000_000)
    gpu = Gpu("nvidia", 24463, 23000, "test")
    assert "already downloaded" in setup.describe(model, True, gpu, 3_390_000_000)
    assert "GB download" in setup.describe(model, False, gpu, None)


def test_a_model_too_large_for_the_card_states_the_shortfall_not_silence() -> None:
    model = Model("qwen3.8:27b", 17_740_000_000)
    tight = Gpu("nvidia", 8192, 7000, "test")
    described = setup.describe(model, False, tight, None)
    assert "MORE THAN IS FREE" in described


def test_no_pull_declines_downloads_without_declining_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def record(tag: str, report: Report) -> bool:
        called.append(tag)
        return True

    monkeypatch.setattr(setup, "pull", record)
    assert setup.run(FULL) == 0
    assert called == []
    assert config.load().values["fast_model"] == "qwen3.5:4b"


def test_a_failed_pull_reports_and_setup_still_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "pull", lambda tag, report: report.add(f"pull {tag}", False, "boom"))
    answers = Answers(
        fast_model="qwen3.5:4b", heavy_model="qwen3.8:27b", install=False, pull=True
    )
    assert setup.run(answers) == 0


def test_an_unwritable_config_is_reported_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(config.PATH_ENV, str(blocker / "config.toml"))
    report = Report()
    setup.write_config("qwen3.5:4b", None, None, report)
    assert report.failures and report.failures[0].name == "write config"


def test_declining_the_global_install_leaves_a_working_uv_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup.run(FULL)
    assert "uv run o" in capsys.readouterr().out


def test_the_global_install_needs_uv_and_a_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """uvx runs from a wheel, where there is no tree to install editable from."""
    monkeypatch.setattr("ollama_stack.setup.shutil.which", lambda name: None)
    report = Report()
    setup.global_install(report)
    assert report.failures


def test_rerunning_shows_the_existing_config_and_does_not_discard_it() -> None:
    config.set_value("num_ctx", "16384")
    setup.run(FULL)
    assert config.load().values["num_ctx"] == 16384


def test_a_cp1252_console_gets_ascii_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "ascii_only", lambda: True)
    setup.rule(4)


def test_nothing_above_ascii_is_written_to_a_cp1252_console(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(setup, "ascii_only", lambda: True)
    setup.run(FULL)
    out = capsys.readouterr().out
    assert all(ord(char) < 128 for char in out), "non-ASCII reached a cp1252 console"


def test_the_wizard_never_hard_codes_a_context_window_number() -> None:
    """TASK-016 moves these, and a literal here would go stale silently."""
    source = Path(setup.__file__).read_text(encoding="utf-8")
    for stale in ("16384", "24576", "32768"):
        assert stale not in source, stale


def test_setup_ends_with_one_instruction_and_it_is_the_tutorial(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup.run(FULL)
    assert "o tutorial" in capsys.readouterr().out.strip().splitlines()[-1]


def test_bootstrap_imports_nothing_outside_the_standard_library() -> None:
    """It runs before anything is installed, so a third-party import would be unimportable."""
    root = Path(__file__).resolve().parents[1] / "bootstrap.py"
    tree = ast.parse(root.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= sys.stdlib_module_names, f"not stdlib: {imported - sys.stdlib_module_names}"


def test_bootstrap_names_the_store_stub_when_running_under_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented Windows front door runs nothing at all under the 0-byte Store alias."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bootstrap

    monkeypatch.setattr(bootstrap, "WINDOWS", True)
    monkeypatch.setattr(
        sys, "executable", r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe"
    )
    warning = bootstrap.store_stub_warning()
    assert warning is not None
    assert "py bootstrap.py" in warning


def test_bootstrap_says_nothing_about_the_stub_on_a_real_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bootstrap

    monkeypatch.setattr(bootstrap, "WINDOWS", True)
    monkeypatch.setattr(sys, "executable", r"C:\Python312\python.exe")
    assert bootstrap.store_stub_warning() is None


def test_verify_releases_the_model_rather_than_leaving_it_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pin never expires, so a wizard that loads and walks away holds the card forever."""
    released: list[str] = []

    class FakeClient:
        def load(self, tag: str) -> None:
            return None

        def generate(self, prompt: str, tag: str) -> object:
            return type("R", (), {"eval_count": 5})()

        def unload(self, tag: str) -> None:
            released.append(tag)

    report = Report()
    real_verify(FakeClient(), "qwen3.5:4b", report)  # type: ignore[arg-type]
    assert released == ["qwen3.5:4b"]
    assert not report.failures


def test_a_tok_per_second_figure_is_withheld_when_the_reply_is_too_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured live: a 2-token reply reported 4 tok/s, which reads as a broken machine."""

    class FakeClient:
        def __init__(self, tokens: int) -> None:
            self.tokens = tokens

        def load(self, tag: str) -> None:
            return None

        def generate(self, prompt: str, tag: str) -> object:
            return type("R", (), {"eval_count": self.tokens})()

        def unload(self, tag: str) -> None:
            return None

    short, long = Report(), Report()
    real_verify(FakeClient(2), "m", short)  # type: ignore[arg-type]
    real_verify(FakeClient(200), "m", long)  # type: ignore[arg-type]
    assert "tok/s" not in short.steps[0].detail
    assert "tok/s" in long.steps[0].detail


def test_declining_the_path_install_is_told_a_command_that_works(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`o` lives only in the venv when the install is declined, so `o tutorial` would not run."""
    report = setup.Report()
    report.o_on_path = False
    setup.closing(report, 32768)
    assert "uv run o tutorial" in capsys.readouterr().out


def test_a_path_changing_install_asks_for_a_new_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = setup.Report()
    report.path_changed = True
    setup.closing(report, 32768)
    out = capsys.readouterr().out
    assert "Open a new terminal" in out
    assert out.strip().splitlines()[-1].strip() == "o tutorial"


def test_bootstrap_help_prints_usage_and_installs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Asking what a script does must not build a 17 MB environment as a side effect."""
    import bootstrap

    assert bootstrap.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "usage: python bootstrap.py" in out
    assert "Environment ready" not in out


def test_the_venv_copy_of_o_does_not_count_as_being_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A friend said Y to the PATH install, was told `o tutorial`, and it did not exist."""
    scripts = tmp_path / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir()
    (scripts / ("o.exe" if os.name == "nt" else "o")).write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("PATH", str(scripts))
    assert setup.resolve_on_path() is None
