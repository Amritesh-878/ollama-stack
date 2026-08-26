"""A settings system whose precedence is untested produces bug reports that are not bugs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ollama_stack import config, models


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Never read or write the real config, and never leave a repointed role behind."""
    for key in config.DEFAULTS:
        monkeypatch.delenv(config.ENV_PREFIX + key.upper(), raising=False)
    monkeypatch.setenv(config.PATH_ENV, str(tmp_path / "config.toml"))
    models.clear_role_tags()
    yield
    models.clear_role_tags()


def _write(text: str) -> Path:
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_missing_file_is_normal_and_nothing_is_created() -> None:
    settings = config.load()
    assert settings.num_ctx == config.DEFAULTS["num_ctx"]
    assert settings.sources["num_ctx"] == "default"
    assert not config.config_path().exists()


def test_the_file_beats_the_built_in_default() -> None:
    _write("num_ctx = 16384\n")
    settings = config.load()
    assert (settings.num_ctx, settings.sources["num_ctx"]) == (16384, "file")


def test_the_environment_beats_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _write("num_ctx = 16384\n")
    monkeypatch.setenv("OLLAMA_STACK_NUM_CTX", "20480")
    settings = config.load()
    assert (settings.num_ctx, settings.sources["num_ctx"]) == (20480, "env")


def test_a_flag_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _write("num_ctx = 16384\n")
    monkeypatch.setenv("OLLAMA_STACK_NUM_CTX", "20480")
    settings = config.load(flags={"num_ctx": 8192})
    assert (settings.num_ctx, settings.sources["num_ctx"]) == (8192, "flag")


def test_all_four_layers_at_once_resolve_to_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of the order is that it holds when every layer disagrees."""
    _write("num_ctx = 16384\nstream = false\n")
    monkeypatch.setenv("OLLAMA_STACK_NUM_CTX", "20480")
    settings = config.load(flags={"num_ctx": 9000})
    assert settings.num_ctx == 9000
    assert settings.sources["num_ctx"] == "flag"
    assert (settings.stream, settings.sources["stream"]) == (False, "file")
    assert settings.sources["keep_alive"] == "default"


def test_a_flag_that_was_not_given_does_not_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset command-line flag arrives as None and must not beat the file."""
    _write("num_ctx = 16384\n")
    settings = config.load(flags={"num_ctx": None})
    assert (settings.num_ctx, settings.sources["num_ctx"]) == (16384, "file")


def test_set_read_unset_read_round_trips() -> None:
    config.set_value("fast_model", "gemma4:e4b")
    assert config.load().values["fast_model"] == "gemma4:e4b"
    assert config.unset_value("fast_model")[0] is True
    assert config.load().values["fast_model"] == config.DEFAULTS["fast_model"]


def test_unsetting_a_key_that_was_never_set_says_so() -> None:
    assert config.unset_value("num_ctx")[0] is False


def test_a_write_says_what_it_dropped_rather_than_deleting_it_quietly(
    tmp_path: Path,
) -> None:
    """A rewrite keeps only keys the file layer parsed, so a hand-edited line can vanish."""
    target = tmp_path / "config.toml"
    target.write_text('num_ctx = 16384\nmy_future_key = "keep me"\n', encoding="utf-8")
    _, notes = config.set_value("stream", "false", target)
    assert any("my_future_key" in note for note in notes)
    assert "my_future_key" not in target.read_text(encoding="utf-8")


def test_writing_one_key_leaves_the_others_alone() -> None:
    config.set_value("num_ctx", "16384")
    config.set_value("keep_alive", "300")
    values = config.load().values
    assert (values["num_ctx"], values["keep_alive"]) == (16384, 300)


def test_booleans_survive_a_round_trip_through_toml() -> None:
    config.set_value("stream", "false")
    assert config.load().stream is False
    config.set_value("stream", "true")
    assert config.load().stream is True


def test_an_unknown_key_is_refused_rather_than_written() -> None:
    with pytest.raises(KeyError):
        config.set_value("favourite_colour", "blue")


def test_a_value_of_the_wrong_type_is_refused_with_the_key_named() -> None:
    with pytest.raises(ValueError, match="num_ctx"):
        config.set_value("num_ctx", "lots")


def test_unparseable_toml_names_the_file_and_does_not_crash() -> None:
    _write("this is not = = toml\n")
    settings = config.load()
    assert settings.num_ctx == config.DEFAULTS["num_ctx"]
    assert any("not valid TOML" in warning for warning in settings.warnings)


def test_an_unknown_key_in_the_file_is_named_and_ignored() -> None:
    _write('favourite_colour = "blue"\nnum_ctx = 16384\n')
    settings = config.load()
    assert settings.num_ctx == 16384
    assert any("favourite_colour" in warning for warning in settings.warnings)


def test_a_wrongly_typed_key_in_the_file_is_named_and_ignored() -> None:
    _write('num_ctx = "wide"\n')
    settings = config.load()
    assert settings.num_ctx == config.DEFAULTS["num_ctx"]
    assert any("num_ctx wants a whole number" in warning for warning in settings.warnings)


def test_a_bad_environment_value_is_named_and_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_STACK_NUM_CTX", "enormous")
    settings = config.load()
    assert settings.num_ctx == config.DEFAULTS["num_ctx"]
    assert any("OLLAMA_STACK_NUM_CTX" in warning for warning in settings.warnings)


def test_an_unwritable_location_raises_oserror_rather_than_failing_silently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(config.PATH_ENV, str(blocker / "config.toml"))
    with pytest.raises(OSError):
        config.set_value("num_ctx", "16384")


def test_a_low_num_ctx_warns_and_says_what_it_costs() -> None:
    settings = config.load(flags={"num_ctx": 4096})
    warning = next(w for w in settings.warnings if "num_ctx is 4096" in w)
    assert "truncates from the FRONT" in warning
    assert "2048" in warning


def test_a_num_ctx_at_or_above_the_threshold_does_not_warn() -> None:
    settings = config.load(flags={"num_ctx": config.LOW_NUM_CTX})
    assert not any("num_ctx is" in warning for warning in settings.warnings)


def test_a_model_outside_the_registry_warns_rather_than_failing() -> None:
    settings = config.load(flags={"fast_model": "llama9:70b"})
    assert any("not a model this tool knows about" in warning for warning in settings.warnings)


def test_a_model_the_wizard_itself_recommends_never_warns() -> None:
    """A friend on an 8 GB card got this warning on every single command for the tier pick."""
    for tag in models.TIER_MODELS:
        settings = config.load(flags={"fast_model": tag})
        assert not [w for w in settings.warnings if "knows about" in w], tag


def test_a_provider_with_no_implementation_warns_and_names_the_ones_that_exist() -> None:
    settings = config.load(flags={"search_provider": "brave"})
    warning = next(w for w in settings.warnings if "no implementation" in w)
    for known in config.PROVIDERS:
        assert known in warning


def test_every_provider_config_accepts_is_one_search_can_actually_build() -> None:
    """Two copies of the list, kept apart so the bare path never imports search. They must agree."""
    from ollama_stack import search

    assert frozenset(search.PROVIDERS) == config.PROVIDERS
    for name in config.PROVIDERS:
        assert search.provider_named(name) is not None


def test_every_setting_reaches_something_that_reads_it() -> None:
    """search_api_key sat here for weeks reading as a feature; no provider ever took a key."""
    from ollama_stack import search

    assert "search_api_key" not in config.DEFAULTS
    assert not hasattr(search, "api_key")


def test_a_key_left_in_an_old_config_file_is_ignored_rather_than_obeyed(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    target.write_text('search_api_key = "sk-live-abcdefghijklmnop"\n', encoding="utf-8")
    values, warnings = config.read_file(target)
    assert values == {}
    assert any("unknown key" in warning for warning in warnings)


def test_config_repoints_a_role_without_editing_the_registry() -> None:
    config.apply(config.load(flags={"fast_model": "qwen3.6:27b"}))
    assert models.resolve("fast").tag == "qwen3.6:27b"
    assert models.REGISTRY["fast"].tag == "qwen3.5:4b"


def test_a_role_repointed_to_something_unknown_is_unmeasured() -> None:
    config.apply(config.load(flags={"heavy_model": "llama9:70b"}))
    spec = models.resolve("heavy")
    assert (spec.tag, spec.measured) == ("llama9:70b", False)


def test_which_and_models_report_the_repointed_role_not_the_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`o which` answering from the registry while the query path uses config is a lie."""
    from ollama_stack.__main__ import main

    config.set_value("fast_model", "qwen3.6:27b")
    for argv in (["which", "fast"], ["models"]):
        with pytest.raises(SystemExit):
            main(argv)
        assert "qwen3.6:27b" in capsys.readouterr().out


def test_an_unrepointed_role_still_resolves_to_the_registry() -> None:
    config.apply(config.load())
    assert models.resolve("fast").tag == "qwen3.5:4b"


def test_the_config_file_is_not_in_the_repo() -> None:
    """`git pull` must never clobber it and an uninstall must not lose it."""
    import os

    os.environ.pop(config.PATH_ENV, None)
    location = str(config.config_path())
    assert "ollama-stack" in location
    assert "site-packages" not in location
    assert os.name != "nt" or "AppData" in location
