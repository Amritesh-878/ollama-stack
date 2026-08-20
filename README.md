# ollama-stack

Ask your local [Ollama](https://ollama.com) models a question from the terminal.

```sh
$ o what is 10+10
20
```

No subcommand, no quotes. The answer streams as it is written, in about a second on a warm model.

> **Early.** Interfaces will change.

---

## Setup

You need [Ollama](https://ollama.com/download) installed. Setup does the rest: it reads your GPU,
recommends models that fit it, pulls them, writes the config, and finishes by timing a real
question on your machine.

```sh
git clone https://github.com/Amritesh-878/ollama-stack.git
cd ollama-stack
py bootstrap.py          # Windows
python3 bootstrap.py     # macOS and Linux
```

`bootstrap.py` imports only the standard library, so it runs before anything is installed. It
offers to install [uv](https://docs.astral.sh/uv/), and if your Python is older than 3.12, offers
to let uv fetch one. It asks before installing either. Decline uv and it falls back to `venv` and
`pip`.

**On Windows use `py`, not `python`.** Windows ships a `python.exe` in `WindowsApps` that is a
0-byte stub. It opens the Microsoft Store and runs nothing. The `py` launcher comes with a real
Python install and is never a stub.

When setup finishes, **open a new terminal**. PATH is read when a shell starts, so the shell you
ran setup in cannot find `o` yet. Then run:

```sh
o tutorial
```

Nine steps, each one a real command run against your machine, reporting your numbers rather than
ours. It writes nothing, and you can quit or repeat it at any point.

### Skip the clone

If you already have `uv`:

```sh
uvx --from git+https://github.com/Amritesh-878/ollama-stack o setup
```

This requires uv, which is the barrier `bootstrap.py` removes, so it is the shortcut rather than
the main path.

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
| `--think` | Reason before answering. Off by default. |
| `-w, --web` | Search the web before answering |
| `--no-web` | Never search. The model answers from training alone. |
| `--no-stream` | Wait for the whole reply instead of streaming |
| `--stats` | Wall time, tokens per second, and how long until the first word |
| `--dry-run` | Show what would be sent, send nothing |
| `--version` | Print the version |

Defaults for most of these come from `o config`, and a flag always wins. See
[Settings](#settings).

The answer goes to stdout and the token counts to stderr, so `o write a haiku > poem.txt` gives
you a clean file.

**Two models, two jobs.** `fast` (`qwen3.5:4b`, 4 GB) answers your questions and is what `o start`
pins. `heavy` (`qwen3.8:27b`, 16 GB) is for `o audit` and anything you would rather wait for. `-m`
overrides either.

**Thinking is off by default, and that accounts for most of the speed.** These models reason at
length before answering, and none of that reasoning is printed, so the terminal sits empty while
it runs. On `qwen3.5:4b`, `what is 10+10` takes **0.45s** to show a word with thinking off and
**1.7s** with it on, because it spends about 170 tokens deciding. Use `--think` when the question
deserves it. `--stats` reports how many tokens went to thinking.

### Web search

The model is given a search tool and calls it when it treats the question as current. It prints
the answer and lists the source URLs on stderr.

```sh
o who won the F1 race last weekend    # it searches on its own
o -w what is the latest python        # force a search first
o --no-web explain this traceback     # never search
```

Three things to know before you rely on it:

- **It searches when a question sounds current, and misses quiet ones.** "Who won the race last
  weekend" triggers it. "What is the latest stable Python" often does not, and you get a confident
  answer from training data instead. `-w` is the only reliable way to force a search.
- **A sourced answer is not a correct answer.** The small model takes a low-quality page over a
  good one. Asked the same forced-search question, `fast` concluded "Python 3.12" from a content
  farm while `heavy` returned 3.14.7 from the same five results. Check the URLs it prints.
- **The default provider scrapes DuckDuckGo with no key and no account, and it rate-limits after
  roughly six searches.** Two DuckDuckGo mirrors are tried, then Wikipedia, which needs no key and
  does not rate-limit. Wikipedia answers "what is X" and "who is X" well and is no use for this
  week's news. If every provider fails you get a note on stderr and the model is told the search
  did not run, so it says so rather than answering from training data.

**If your question starts with a command name**, it runs the command. `o status of the economy`
runs `status`. Use `o ask "status of the economy"` for those.

---

## Commands

| | |
| --- | --- |
| `o setup` | Re-run the wizard: hardware, models, config, and a verification run |
| `o tutorial` | Nine steps, run for real. Writes nothing and can be repeated. |
| `o start` | Pin a model in VRAM so the next question is fast |
| `o stop` | Unload it and give the card back |
| `o status` | What is loaded, how much VRAM it holds, how long it stays |
| `o ask "..."` | Ask, when the question starts with a command name |
| `o audit file.py` | Have a model read one file and report defects |
| `o models` | List the aliases |
| `o which coder` | Show what an alias resolves to |
| `o config` | Show your settings and where each one came from |

Ollama drops a model after 5 minutes idle and reloading `fast` costs about 5s, so `o start` is the
difference between a 0.6s answer and a 6s one. `o start` refuses if the card cannot hold the model
and names what is already on it. Nothing runs in the background.

---

## Settings

`o config` prints what is in effect and where each value came from. Nothing is written to disk
until you set something.

```sh
o config                              # show everything
o config set fast_model gemma4:e4b    # change the model a bare question uses
o config set num_ctx 16384
o config unset num_ctx                # back to the built-in default
```

The config file is `%APPDATA%\ollama-stack\config.toml` on Windows and
`~/.config/ollama-stack/config.toml` elsewhere. It sits outside the repo, so `git pull` cannot
overwrite it.

**Four places a value can come from. Highest wins:**

| | | |
| - | --- | --- |
| 1 | Command-line flag | `o --num-ctx 16384 ...` |
| 2 | Environment variable | `OLLAMA_STACK_NUM_CTX=16384` |
| 3 | Config file | `num_ctx = 16384` |
| 4 | Built-in default | `32768` |

Keys: `host`, `fast_model`, `heavy_model`, `num_ctx`, `keep_alive`, `search_provider`,
`search_api_key`, `stream`. `host` also reads Ollama's own `OLLAMA_HOST`, below `OLLAMA_STACK_HOST`
in precedence, so pointing the daemon elsewhere moves this tool with it. `search_api_key` is never printed in full.

Setting `num_ctx` below 8192 is allowed and warns, because 32768 is a measured floor rather than a
preference. See below.

---

## Why it refuses sometimes

Ollama's `num_ctx` defaults to 4096 whatever the model advertises, and a longer prompt is cut
**from the front** with no warning. The model then answers confidently from what is left.

So this tool always sends `num_ctx`, always checks how much of your prompt was read, and refuses
rather than return an answer built on half a question. If it refuses, send less or raise
`--num-ctx`.

**Two separate checks run, and they measure different things.** Before sending, the prompt has to
leave room for the answer. At the default `num_ctx` of 32768 that allows about 24576 tokens of
prompt, holding 8192 back to generate into. After the reply, the token counts are compared against
what was sent. A count far below what was sent means the front was dropped and the answer is not
trustworthy.

**Going over `num_ctx` is not a gentle trim.** Measured on this project: a prompt that exceeds the
window comes back cut to just over *half* of it, losing 63% from the front, with no error. That is
why the refusal happens before sending rather than after.

**The size estimate is characters over four, and real text varies.** Prose measured 5.5 characters
per token here and source code measured 3.0. `--stats` shows the estimate next to the real count,
and the real count is the one that matters.

---

## Using it from Python

```python
from ollama_stack import OllamaClient

client = OllamaClient(num_ctx=32768, think=False)
reply = client.generate("summarise this", "heavy", context=open("notes.md").read())
print(reply.text)
```

`think` is sent on every request, like `num_ctx`. Omit it and the model's own default applies,
which on a small model can mean a second and a half of reasoning about `10+10`.

Also available: `stream()`, `chat()`, and `load()` / `unload()` behind `o start` and `o stop`.
Model aliases are defined in `src/ollama_stack/models.py`.

---

## Coming

Images, follow-up questions, and an MCP server so editors can call local models as tools.

---

## Worth knowing

A local model is useful and it is not a senior engineer. `o audit` finding nothing means the file
is unexamined, not clean.

**Measured, so you can calibrate.** On a real file with three known defects that lost real users'
work, `qwen3.8:27b` named **one of the three** on each of three runs, and so did every other model
tried, including `qwen3.6:27b`, `qwen3-coder:30b` and `deepseek-r1:32b`. **One of the three has
never been found by any model.** They find the obvious defect, explain it well, and stop. Treat a
report as a starting point rather than as coverage.

---

## Development

```sh
uv sync --all-extras
uv run ruff check --fix . && uv run mypy . && uv run pytest
```

MIT. See [LICENSE](LICENSE).
