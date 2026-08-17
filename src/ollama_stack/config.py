"""Defaults that survive between invocations, under one precedence order stated in one place."""

from __future__ import annotations

import contextlib
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ollama_stack.models import (
    DEFAULT_NUM_CTX,
    FAST_ALIAS,
    HEAVY_ALIAS,
    REGISTRY,
    set_role_tag,
)

ENV_PREFIX = "OLLAMA_STACK_"
PATH_ENV = f"{ENV_PREFIX}CONFIG"
APP_DIR = "ollama-stack"
FILE_NAME = "config.toml"
LOW_NUM_CTX = 8192
SECRET_KEYS = frozenset({"search_api_key"})
PROVIDERS = frozenset({"duckduckgo"})

# Highest first. Stated here, in `o config`, and in the README, because an undocumented
# precedence order produces bug reports that are not bugs.
PRECEDENCE = ("flag", "env", "file", "default")

DEFAULTS: dict[str, Any] = {
    "fast_model": REGISTRY[FAST_ALIAS].tag,
    "heavy_model": REGISTRY[HEAVY_ALIAS].tag,
    "num_ctx": DEFAULT_NUM_CTX,
    "keep_alive": -1,
    "search_provider": "duckduckgo",
    "search_api_key": "",
    "stream": True,
}


def config_path() -> Path:
    """The platform config directory, never the repo, so `git pull` cannot clobber it."""
    override = os.environ.get(PATH_ENV)
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / APP_DIR / FILE_NAME


@dataclass(frozen=True)
class Settings:
    """Resolved values plus where each came from, because `o config` has to show both."""

    values: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    sources: dict[str, str] = field(default_factory=lambda: dict.fromkeys(DEFAULTS, "default"))
    warnings: list[str] = field(default_factory=list)

    @property
    def num_ctx(self) -> int:
        return int(self.values["num_ctx"])

    @property
    def keep_alive(self) -> int:
        return int(self.values["keep_alive"])

    @property
    def stream(self) -> bool:
        return bool(self.values["stream"])

    @property
    def search_provider(self) -> str:
        return str(self.values["search_provider"])

    @property
    def search_api_key(self) -> str:
        return str(self.values["search_api_key"])


def _coerce(key: str, value: Any) -> tuple[Any, str]:
    """TOML is typed, so a wrong type here is the file's fault and is named rather than guessed."""
    want = type(DEFAULTS[key])
    if want is bool:
        if isinstance(value, bool):
            return value, ""
        return None, f"{key} wants true or false, got {value!r}"
    if want is int:
        if isinstance(value, bool) or not isinstance(value, int):
            return None, f"{key} wants a whole number, got {value!r}"
        return value, ""
    return str(value), ""


def coerce_text(key: str, raw: str) -> tuple[Any, str]:
    """Environment variables and `o config set` both arrive as text and need the same reading."""
    want = type(DEFAULTS[key])
    if want is bool:
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True, ""
        if lowered in ("0", "false", "no", "off"):
            return False, ""
        return None, f"{key} wants true or false, got {raw!r}"
    if want is int:
        try:
            return int(raw.strip()), ""
        except ValueError:
            return None, f"{key} wants a whole number, got {raw!r}"
    return raw, ""


def read_file(path: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """A missing file is normal; a broken one names the problem and contributes nothing."""
    target = config_path() if path is None else path
    try:
        raw = target.read_bytes()
    except OSError:
        return {}, []
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return {}, [f"{target} is not valid TOML and was ignored: {exc}"]
    values: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in parsed.items():
        if key not in DEFAULTS:
            warnings.append(f"{target}: unknown key {key!r}, ignored")
            continue
        coerced, problem = _coerce(key, value)
        if problem:
            warnings.append(f"{target}: {problem}")
            continue
        values[key] = coerced
    return values, warnings


def read_env() -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    warnings: list[str] = []
    for key in DEFAULTS:
        raw = os.environ.get(ENV_PREFIX + key.upper())
        if raw is None:
            continue
        coerced, problem = coerce_text(key, raw)
        if problem:
            warnings.append(f"{ENV_PREFIX}{key.upper()}: {problem}")
            continue
        values[key] = coerced
    return values, warnings


def _advice(values: dict[str, Any]) -> list[str]:
    """Warnings about settings that are legal but will cost the user something."""
    notes: list[str] = []
    num_ctx = int(values["num_ctx"])
    if num_ctx < LOW_NUM_CTX:
        notes.append(
            f"num_ctx is {num_ctx}. Ollama truncates from the FRONT without warning, and the "
            f"client refuses any prompt reaching {num_ctx // 2} tokens, so attached files and "
            "search results will be cut or refused. 32768 is a measured floor, not a preference."
        )
    known = {spec.tag for spec in REGISTRY.values()}
    for key in ("fast_model", "heavy_model"):
        tag = str(values[key])
        if tag not in known:
            notes.append(f"{key} is {tag!r}, which is not in the registry - see `o models`")
    provider = str(values["search_provider"])
    if provider not in PROVIDERS:
        notes.append(f"search_provider {provider!r} has no implementation; duckduckgo will be used")
    return notes


def load(flags: dict[str, Any] | None = None, path: Path | None = None) -> Settings:
    """Precedence, highest first: flag, environment, file, built-in default."""
    values = dict(DEFAULTS)
    sources = dict.fromkeys(DEFAULTS, "default")
    file_values, warnings = read_file(path)
    env_values, env_warnings = read_env()
    warnings.extend(env_warnings)
    layers = ((file_values, "file"), (env_values, "env"), (flags or {}, "flag"))
    for layer, name in layers:
        for key, value in layer.items():
            if key not in DEFAULTS or value is None:
                continue
            values[key] = value
            sources[key] = name
    warnings.extend(_advice(values))
    return Settings(values=values, sources=sources, warnings=warnings)


def apply(settings: Settings) -> None:
    """Points the two role aliases at whatever config says, before any name gets resolved."""
    set_role_tag(FAST_ALIAS, str(settings.values["fast_model"]))
    set_role_tag(HEAVY_ALIAS, str(settings.values["heavy_model"]))


def shown(key: str, value: Any) -> str:
    """A key printed in full is a key in a screenshot, so it never leaves here intact."""
    if key not in SECRET_KEYS:
        return str(value)
    text = str(value)
    if not text:
        return "(unset)"
    return f"set, ending {text[-4:]}" if len(text) > 8 else "set"


def _as_toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_file(values: dict[str, Any], path: Path | None = None) -> Path:
    """Only what differs from the built-in default is written, so the file stays readable."""
    target = config_path() if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {_as_toml(values[key])}" for key in DEFAULTS if key in values]
    target.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    # Best effort: a real restriction on POSIX and close to a no-op on Windows.
    with contextlib.suppress(OSError):
        target.chmod(0o600)
    return target


def set_value(key: str, raw: str, path: Path | None = None) -> tuple[Any, list[str]]:
    """Writes one key, leaving every other key in the file exactly as it was."""
    if key not in DEFAULTS:
        raise KeyError(key)
    value, problem = coerce_text(key, raw)
    if problem:
        raise ValueError(problem)
    stored, _ = read_file(path)
    stored[key] = value
    write_file(stored, path)
    return value, _advice(load(path=path).values)


def unset_value(key: str, path: Path | None = None) -> bool:
    """Reverts one key to the built-in default by removing it, not by writing the default in."""
    if key not in DEFAULTS:
        raise KeyError(key)
    stored, _ = read_file(path)
    if key not in stored:
        return False
    del stored[key]
    write_file(stored, path)
    return True
