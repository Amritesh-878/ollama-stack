# ollama-stack

A fast command-line front end for local models running on [Ollama](https://ollama.com).

```sh
$ o what is 10+10
20
```

About a second, on a warm model. The point is that asking a local model something should feel
like running a shell command, not like opening an app.

> **Status: early.** `ask`, `audit`, `models` and `which` work today. `setup`, `start`, `stop`,
> `status`, streaming output and web search are designed but not built — see [Roadmap](#roadmap).
> Interfaces will change.

---

## Why this exists

Ollama's HTTP API has one behaviour that quietly ruins results, and every tool built on it has
to handle the same thing:

**`num_ctx` defaults to 4096 regardless of what the model advertises, and Ollama does not warn
you.** A model with a 256K context window gets 4096 tokens unless you pass the option
explicitly. When your prompt is longer, it is truncated **from the front** — and the model
answers confidently from whatever survived, with no indication anything was lost.

Measured here: with a ~7000 token prompt at `num_ctx: 4096`, a marker placed at the top of the
prompt was gone and one at the bottom survived. Asked to report both and to say `MISSING` for
anything absent, the model reported the bottom value for *both*. It filled the hole rather than
reporting it.

So this library makes that impossible to get wrong:

- `num_ctx` is set by the client on every request. Callers cannot forget it.
- `prompt_eval_count` comes back on every reply — the only honest signal for how much of your
  prompt was actually read.
- A reply that reached the usable window **raises** instead of returning. So does a reply that
  omits the count entirely, because unknown is not the same as fine.

If you only take one thing from this repo, take that. It applies whether or not you use this
tool.

---

## Requirements

- [Ollama](https://ollama.com/download) installed and running
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

At least one model pulled:

```sh
ollama pull qwen3.5:4b
```

---

## Install

```sh
git clone https://github.com/Amritesh-878/ollama-stack.git
cd ollama-stack
uv sync
```

To get `o` on your PATH everywhere:

```sh
uv tool install .
```

---

## Usage

```sh
o ask "explain monads"                # ask the default model
o ask "explain monads" -m coder       # pick a model by alias
o audit src/thing.py                  # screen one file for defects
o models                              # list known models
o which coder                         # what an alias resolves to
```

### Options

| Flag | Does |
| ---- | ---- |
| `-m, --model` | Model alias or raw Ollama tag |
| `--num-ctx` | Context window to request. Default 32768. |

Token counts go to stderr, so piping stdout stays clean:

```sh
o ask "write a haiku" > poem.txt
```

### As a library

```python
from ollama_stack import OllamaClient

client = OllamaClient(num_ctx=32768)
reply = client.generate("summarise this", "qwen", context=open("notes.md").read())

print(reply.text)
print(reply.prompt_eval_count)   # how much it actually read
```

`client.chat(messages, model, tools=...)` is there too, for multi-turn and function calling.

Pass `strict=False` if you would rather inspect `reply.suspect_truncation` yourself than have
it raise.

---

## Model aliases

Aliases live in one place (`src/ollama_stack/models.py`) so a rename is a one-line change.
`o models` marks which ones have measurements behind them and which do not — an unmeasured
model is one nobody has benchmarked on this workload, not one that does not work.

You are not limited to the list. Any Ollama tag works:

```sh
o ask "hello" -m llama4:8b
```

---

## Roadmap

| | |
| --- | --- |
| `o setup` | Interactive first run: checks Ollama, reads your VRAM, recommends and pulls models |
| `o start` / `o stop` | Pin a model in VRAM and release it. Cold load costs ~10s; warm is ~1s. |
| Streaming | First token in ~200ms instead of waiting for the whole reply |
| Web search | The model calls out to the web when a question is past its cutoff |
| Images | `-i photo.png` for vision-capable models |
| MCP server | Expose local models as tools to MCP-aware clients |

---

## Development

```sh
uv sync --all-extras
uv run ruff check --fix .
uv run mypy .
uv run pytest
```

Lint, then typecheck, then tests, in that order. The test suite makes no network calls — the
Ollama transport is mocked — so it runs anywhere.

---

## A caveat worth stating plainly

Small local models are useful and they are not a senior engineer. This tool is built on the
assumption that **you can trust what a local model says, but never that it is finished.** A
"nothing found" from `o audit` means nothing at all: the file it passed is unexamined, not
clean. Read the diff yourself.

---

## License

MIT — see [LICENSE](LICENSE).
