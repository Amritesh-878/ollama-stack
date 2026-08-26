"""Program lookup that never resolves to the working directory."""

from __future__ import annotations

import os
from pathlib import Path

# What Windows appends to a bare name, in order. Only used to find the file; the caller runs
# whatever comes back by its full path, so nothing is appended twice.
FALLBACK_PATHEXT = (".COM", ".EXE", ".BAT", ".CMD")


def _here() -> Path:
    return Path.cwd().resolve()


def _entries(path: str | None) -> list[Path]:
    """PATH, minus anything the working directory could be reached through.

    Relative entries go too. They are almost unheard of, and one of them means "a directory
    under wherever this happens to be run", which is the thing being defended against.
    """
    raw = os.environ.get("PATH", "") if path is None else path
    here = _here()
    kept: list[Path] = []
    for entry in raw.split(os.pathsep):
        if not entry or entry == os.curdir:
            continue
        directory = Path(entry)
        if not directory.is_absolute():
            continue
        try:
            if directory.resolve() == here:
                continue
        except OSError:
            continue
        kept.append(directory)
    return kept


def _suffixes(name: str) -> tuple[str, ...]:
    if os.name != "nt":
        return ("",)
    listed = tuple(e for e in os.environ.get("PATHEXT", "").split(os.pathsep) if e)
    extensions = listed or FALLBACK_PATHEXT
    lowered = name.lower()
    if any(lowered.endswith(e.lower()) for e in extensions):
        return ("",)
    return extensions


def _runnable(candidate: Path) -> bool:
    return candidate.is_file() and os.access(candidate, os.X_OK)


def on_path(name: str, path: str | None = None) -> str | None:
    """Where `name` really lives, or None. Never a program sitting in the working directory.

    Windows searches the working directory before PATH, and does it twice over: CreateProcess
    resolves a bare name against the parent's directory first, and shutil.which puts os.curdir
    at the front of whatever path it is handed - so filtering PATH and calling which does not
    help, and neither does calling it once per entry. So the search is done here instead. It
    is short because the only interesting part is which directories are allowed to answer.

    Without this, `o status` run inside a cloned repo executes that repo's nvidia-smi.exe, and
    `o implement` its git.exe. Every lookup goes through here, and every caller passes on the
    full path it returns rather than the bare name.
    """
    if os.path.dirname(name):
        # Already a path, so it is the caller's own choice and not a search at all.
        return name if _runnable(Path(name)) else None
    for directory in _entries(path):
        for suffix in _suffixes(name):
            candidate = directory / (name + suffix)
            if _runnable(candidate):
                return str(candidate)
    return None
