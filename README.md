# ollama-stack

Ask your local [Ollama](https://ollama.com) models a question from the terminal.

```sh
$ o what is 10+10
20
```

No subcommand, no quotes. The answer streams as it's written, in about a second on a warm model.

> **Early.** Interfaces will change.

---

## Setup

You need [Ollama](https://ollama.com/download) installed. Everything else the setup does for you —
it finds your GPU, recommends models that will actually fit it, pulls them, writes the config, and
finishes by timing a real question on your machine.

```sh
git clone https://github.com/Amritesh-878/ollama-stack.git
cd ollama-stack
py bootstrap.py          # Windows
python3 bootstrap.py     # macOS and Linux
```

`bootstrap.py` uses only the standard library, so it runs before anything is installed. It will
offer to install [uv](https://docs.astral.sh/uv/) and, if your Python is older than 3.12, offer to
let uv fetch one. Both are asked first, and declining uv falls back to `venv` and `pip`.

**On Windows use `py`, not `python`.** Windows ships a `python.exe` in `WindowsApps` that is a
0-byte stub: it opens the Microsoft Store and runs nothing. The `py` launcher comes with a real
Python install and is never a stub.

When it finishes, **open a new terminal** — PATH is read when a shell starts, so the one you ran
setup in cannot see `o` yet — and run:

```sh
o tutorial
```

Nine steps, each one a real command run against your machine, reporting your numbers rather than
ours. It changes nothing and you can quit or re-run it at any point.

### Just want to try it

If you already have `uv`, you can skip the clone:

```sh
uvx --from git+https://github.com/Amritesh-878/ollama-stack o setup
```

This needs uv already installed, which is the barrier `bootstrap.py` exists to remove — so it is
the shortcut, not the main path.

> **Windows is the only platform this has been tested on.** The macOS and Linux paths are written
> and unverified.

---

## Asking things

```sh
o what is 10+10                       # just ask
o -m coder write a bash retry loop    # use a different model
cat main.py | o explain this          # pipe a file in as context
```

| Flag | Does |
| ---- | ---- |
| `-m, --model` | Alias like `heavy` or `coder`, or any Ollama tag like `llama4:8b` |
| `--num-ctx` | Context window to request. Default 32768. |
| `--think` | Let the model reason before answering. Off by default. |
| `-w, --web` | Search the web first, instead of leaving it to the model |
| `--no-web` | Never search; the model answers from training alone |
| `--no-stream` | Wait for the whole reply instead of streaming |
| `--stats` | Wall time, tokens per second, and how long until the first word |
| `--dry-run` | Show what would be sent, send nothing |
| `--version` | Print the version |

Defaults for most of these come from `o config`, and a flag always wins — see
[Settings](#settings).

The answer goes to stdout and the token counts to stderr, so `o write a haiku > poem.txt` gives
you a clean file.

**Two models, two jobs.** `fast` (`qwen3.5:4b`, 4 GB) answers your questions and is what
`o start` pins. `heavy` (`qwen3.8:27b`, 16 GB) is for `o audit` and anything you'd rather wait
for. `-m` overrides either.

**Thinking is off by default, and that is most of the speed.** These models will reason at
length before saying anything, and none of that reasoning is printed — so you watch an empty
terminal. On `qwen3.5:4b`, `what is 10+10` takes **0.45s** to show a word with thinking off and
**1.7s** with it on, because it spends ~170 tokens deciding. Use `--think` when the question
deserves it. `--stats` shows how many tokens went to thinking.

### Web search

The model gets a search tool and decides for itself whether to use it. Ask about something
recent and it looks it up, prints the answer, and lists the source URLs on stderr.

```sh
o who won the F1 race last weekend    # it searches on its own
o -w what is the latest python        # force a search first
o --no-web explain this traceback     # never search
```

Three things to know before you rely on it:

- **It searches when the question sounds current, and misses quiet ones.** "Who won the race last
  weekend" triggers it; "what is the latest stable Python" often doesn't, and you get a confident
  answer from training data instead. `-w` is the only reliable way to make it look.
- **A sourced answer is not a correct answer.** The small model will take a low-quality page over
  a good one. Asked the same forced-search question, `fast` concluded "Python 3.12" from a content
  farm while `heavy` got 3.14.7 from the same five results. Check the URLs it prints.
- **The default provider is DuckDuckGo scraping, with no key and no account, and it rate-limits
  after roughly six searches.** When that happens you get a note on stderr and an answer without
  search — never a failed command. It clears after a few minutes idle.

**If your question starts with a command name**, it runs the command — `o status of the economy`
runs `status`. Use `o ask "status of the economy"` for those.

---

## Commands

| | |
| --- | --- |
| `o setup` | Re-run the wizard: hardware, models, config, and a verification run |
| `o tutorial` | Nine steps, run for real. Changes nothing and can be repeated |
| `o start` | Pin a model in VRAM so the next question is fast |
| `o stop` | Unload it and give the card back |
| `o status` | What's loaded, how much VRAM it's using, how long it stays |
| `o ask "..."` | Ask, when the question starts with a command name |
| `o audit file.py` | Have a model read one file and report defects |
| `o models` | List the aliases |
| `o which coder` | Show what an alias resolves to |
| `o config` | Show your settings and where each one came from |

Ollama drops a model after 5 minutes idle and reloading `fast` costs ~5s, so `o start` is the
difference between a 0.6s answer and a 6s one. `o start` refuses if the card can't hold the
model and tells you what's already on it. Nothing runs in the background.

---

## Settings

`o config` prints what's in effect and where each value came from. Nothing is written until you
set something.

```sh
o config                              # show everything
o config set fast_model gemma4:e4b    # change the model a bare question uses
o config set num_ctx 16384
o config unset num_ctx                # back to the built-in default
```

Settings live in `%APPDATA%\ollama-stack\config.toml` on Windows and
`~/.config/ollama-stack/config.toml` elsewhere — outside the repo, so `git pull` can't clobber
them.

**Four places a value can come from. Highest wins:**

| | | |
| - | --- | --- |
| 1 | Command-line flag | `o --num-ctx 16384 ...` |
| 2 | Environment variable | `OLLAMA_STACK_NUM_CTX=16384` |
| 3 | Config file | `num_ctx = 16384` |
| 4 | Built-in default | `32768` |

Keys: `fast_model`, `heavy_model`, `num_ctx`, `keep_alive`, `search_provider`, `search_api_key`,
`stream`. `search_api_key` is never printed in full.

Setting `num_ctx` below 8192 is allowed and warns, because it is a measured floor rather than a
preference — see below.

---

## Why it refuses sometimes

Ollama's `num_ctx` defaults to 4096 no matter what the model advertises, and a longer prompt is
cut **from the front** with no warning — the model then answers confidently from what's left.

So this tool always sends `num_ctx`, always checks how much of your prompt was actually read,
and stops rather than hand you an answer built on half a question. If it refuses, send less or
raise `--num-ctx`.

**There are two separate checks, and they measure different things.** Before sending, your prompt
has to leave room for the answer — at the default `num_ctx` of 32768 that means about 24576 tokens
of prompt, with 8192 held back to generate into. After the reply, the token counts are compared
against what was sent: if far less was read than sent, the front was dropped and the answer is not
trustworthy.

**Going over `num_ctx` is not a gentle trim.** Measured on this project: a prompt that exceeds the
window comes back cut to just over *half* of it, losing 63% from the front, silently. That is why
the refusal happens before sending rather than after.

**The size estimate is characters over four, and real text varies a lot** — prose measured 5.5
characters per token here and source code measured 3.0. So `--stats` shows the estimate next to
the real count, and the real count is the one that matters.

---

## Using it from Python

```python
from ollama_stack import OllamaClient

client = OllamaClient(num_ctx=32768, think=False)
reply = client.generate("summarise this", "heavy", context=open("notes.md").read())
print(reply.text)
```

`think` is sent on every request, like `num_ctx` — leave it to the model's own default and a
small model can spend a second and a half reasoning about `10+10`.

`stream()`, `chat()`, `start`/`stop` via `load()` and `unload()`. Model aliases live in
`src/ollama_stack/models.py`.

---

## Coming

Interactive setup, images, follow-up questions, and an MCP server so editors can call local
models as tools.

---

## Worth knowing

A local model is useful and it is not a senior engineer. `o audit` finding nothing means the
file is unexamined, not clean.

**Measured, so you can calibrate.** On a real file with three known defects that lost real users'
work, `qwen3.8:27b` named **one of the three** on each of three runs — and so did every other
model tried, including `qwen3.6:27b`, `qwen3-coder:30b` and `deepseek-r1:32b`. **One of the three
has never been found by any model.** They tend to find the obvious defect, explain it well, and
stop. Treat a report as a starting point, never as coverage.

---

## Development

```sh
uv sync --all-extras
uv run ruff check --fix . && uv run mypy . && uv run pytest
```

MIT — see [LICENSE](LICENSE).
