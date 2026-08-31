"""The front door: stdlib only, so it runs on a machine where nothing is installed yet."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

NEEDED = (3, 12)
ROOT = Path(__file__).resolve().parent
WINDOWS = platform.system() == "Windows"

# Set by --yes. Consent to fetching and running someone else's installer is not something
# a missing terminal can give on the user's behalf.
ASSUME_YES = False

UV_PS = "irm https://astral.sh/uv/install.ps1 | iex"
UV_INSTALL_WINDOWS = f'powershell -ExecutionPolicy ByPass -c "{UV_PS}"'
UV_INSTALL_UNIX = "curl -LsSf https://astral.sh/uv/install.sh | sh"


def _utf8() -> None:
    """A redirected stream is cp1252 here, and this script prints a model's name back."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def say(message: str = "") -> None:
    print(message, flush=True)


def ask(question: str, default: bool = True, *, installs: bool = False) -> bool:
    """No TTY means take the default rather than blocking on input nobody can give.

    Except when saying yes downloads and runs someone else's code. Piping into this file
    used to fetch and execute the uv installer with nobody having agreed to it, so that
    answer has to be given out loud with --yes.
    """
    if not sys.stdin.isatty():
        if installs and not ASSUME_YES:
            say(f"{question} [no terminal, so no. Pass --yes to agree in advance.]")
            return False
        say(f"{question} [no terminal, assuming {'yes' if default else 'no'}]")
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        say()
        return False
    if not answer:
        return default
    return answer.startswith("y")


def run(command: list[str], where: Path | None = None) -> tuple[int, str]:
    """Returns the real stderr, because a wrapped error is one nobody can act on."""
    try:
        done = subprocess.run(
            command, cwd=where or ROOT, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return 1, str(exc)
    return done.returncode, (done.stderr or done.stdout).strip()


def store_stub_warning() -> str | None:
    """The WindowsApps python is the Store alias, and on a bare machine it runs nothing at all."""
    if not WINDOWS or "WindowsApps" not in sys.executable:
        return None
    return (
        f"This is running under {sys.executable}, which is the Microsoft Store's Python alias.\n"
        "  On a machine with no real Python installed, that alias is a 0-byte stub: it opens the\n"
        "  Store and runs nothing. If anything below behaves strangely, install Python from\n"
        "  python.org and re-run this with:  py bootstrap.py"
    )


def find_uv() -> str | None:
    return shutil.which("uv")


def offer_uv() -> str | None:
    """The one place this script installs anything, and only after asking."""
    command = UV_INSTALL_WINDOWS if WINDOWS else UV_INSTALL_UNIX
    say("uv is not installed. It manages the Python environment for this tool.")
    say(f"  {command}")
    if not ask("Install uv now?", installs=True):
        say("Fine - falling back to venv and pip.")
        return None
    shell = ["powershell", "-ExecutionPolicy", "ByPass", "-c", UV_PS]
    code, error = run(shell if WINDOWS else ["sh", "-c", UV_INSTALL_UNIX])
    if code != 0:
        say(f"uv install failed: {error}")
        return None
    found = find_uv() or shutil.which("uv", path=str(Path.home() / ".local" / "bin"))
    if found is None:
        say("uv installed but is not on PATH yet. Open a new terminal and re-run this script.")
    return found


def python_is_new_enough() -> bool:
    return sys.version_info >= NEEDED


def version_text() -> str:
    found = ".".join(str(part) for part in sys.version_info[:3])
    return f"found Python {found}, need {NEEDED[0]}.{NEEDED[1]} or newer"


def uv_supplies_python(uv: str) -> bool:
    """uv can fetch an interpreter, which is why the uv check runs before the version check."""
    wanted = f"{NEEDED[0]}.{NEEDED[1]}"
    say(f"{version_text()}. uv can install one.")
    if not ask(f"Let uv install Python {wanted}?", installs=True):
        return False
    code, error = run([uv, "python", "install", wanted])
    if code != 0:
        say(f"Could not install Python {wanted}: {error}")
        return False
    say(f"Python {wanted} installed.")
    return True


def sync_with_uv(uv: str) -> bool:
    say("Installing the package with uv ...")
    code, error = run([uv, "sync"])
    if code != 0:
        say(f"uv sync failed: {error}")
        return False
    return True


def sync_with_pip() -> bool:
    """The fallback that makes declining uv a real choice rather than a dead end."""
    venv = ROOT / ".venv"
    if not venv.exists():
        say("Creating a virtual environment ...")
        code, error = run([sys.executable, "-m", "venv", str(venv)])
        if code != 0:
            say(f"Could not create a virtual environment: {error}")
            return False
    python = venv / ("Scripts/python.exe" if WINDOWS else "bin/python")
    say("Installing the package with pip ...")
    code, error = run([str(python), "-m", "pip", "install", "-e", "."])
    if code != 0:
        say(f"pip install failed: {error}")
        return False
    return True


def wizard_command(uv: str | None) -> list[str]:
    if uv is not None:
        return [uv, "run", "o", "setup"]
    binary = ROOT / ".venv" / ("Scripts/o.exe" if WINDOWS else "bin/o")
    return [str(binary), "setup"]


def hand_off(uv: str | None, passthrough: list[str]) -> int:
    """Phase B runs as its own process so it gets the installed package, not this interpreter."""
    command = wizard_command(uv) + passthrough
    say()
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except OSError as exc:
        say(f"Could not start the setup wizard: {exc}")
        say(f"  Try running it yourself:  {' '.join(command)}")
        return 1


USAGE = """usage: python bootstrap.py [OPTIONS]   (on Windows: py bootstrap.py)

Sets up ollama-stack: installs uv if you allow it, builds the environment, then
runs the setup wizard. Uses only the standard library, so it works before
anything is installed.

  --help          show this and do nothing else
  --yes           agree in advance to installing uv and a Python, for use with no
                  terminal. Without it, a run with nothing attached to stdin installs
                  nothing rather than assuming you would have said yes.

Anything else is passed to the wizard, which has its own flags:

  --fast-model TAG      --heavy-model TAG     --search-provider NAME
  --install / --no-install                    --no-pull

Run `o setup --help` once installed for the wizard's own help."""


def main(argv: list[str] | None = None) -> int:
    global ASSUME_YES
    _utf8()
    passthrough = list(sys.argv[1:] if argv is None else argv)
    if "--yes" in passthrough:
        ASSUME_YES = True
        passthrough = [arg for arg in passthrough if arg != "--yes"]
    # Asking what a script does must not install anything, so this returns before any side effect.
    if "--help" in passthrough or "-h" in passthrough:
        print(USAGE)
        return 0
    say("ollama-stack setup")
    say("=" * 18)
    stub = store_stub_warning()
    if stub is not None:
        say()
        say(f"  warning: {stub}")
    say()

    uv = find_uv()
    if uv is None:
        uv = offer_uv()

    if uv is not None:
        if not python_is_new_enough() and not uv_supplies_python(uv):
            say("Without a suitable Python this cannot continue.")
            return 1
        if not sync_with_uv(uv):
            return 1
    else:
        if not python_is_new_enough():
            say(f"No uv, and {version_text()}.")
            say("  Install Python 3.12+ from python.org, or re-run and accept the uv install.")
            return 1
        if not sync_with_pip():
            return 1

    say("Environment ready.")
    return hand_off(uv, passthrough)


if __name__ == "__main__":
    sys.exit(main())
