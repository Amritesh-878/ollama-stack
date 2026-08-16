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

You need [Ollama](https://ollama.com/download) running, Python 3.12+, and one model pulled
(`ollama pull qwen3.5:4b`).

```sh
git clone https://github.com/Amritesh-878/ollama-stack.git
cd ollama-stack
uv sync
uv tool install .     # puts `o` on your PATH
```

---

## Asking things

```sh
o what is 10+10                       # just ask
o -m coder write a bash retry loop    # use a different model
cat main.py | o explain this          # pipe a file in as context
```

| Flag | Does |
| ---- | ---- |
| `-m, --model` | Alias like `coder`, or any Ollama tag like `llama4:8b` |
| `--num-ctx` | Context window to request. Default 32768. |
| `--no-stream` | Wait for the whole reply instead of streaming |
| `--stats` | Show wall time and tokens per second |
| `--dry-run` | Show what would be sent, send nothing |
| `--version` | Print the version |

The answer goes to stdout and the token counts to stderr, so `o write a haiku > poem.txt` gives
you a clean file.

**If your question starts with a command name**, it runs the command — `o status of the economy`
runs `status`. Use `o ask "status of the economy"` for those.

---

## Commands

| | |
| --- | --- |
| `o start` | Pin a model in VRAM so the next question is fast |
| `o stop` | Unload it and give the card back |
| `o status` | What's loaded, how much VRAM it's using, how long it stays |
| `o ask "..."` | Ask, when the question starts with a command name |
| `o audit file.py` | Have a model read one file and report defects |
| `o models` | List the aliases |
| `o which coder` | Show what an alias resolves to |

A cold model takes ~10s to load and Ollama drops it after 5 minutes idle, so `o start` is the
difference between a fast tool and a slow one. `o start` refuses if the card can't hold the
model and tells you what's already on it. Nothing runs in the background.

---

## Why it refuses sometimes

Ollama's `num_ctx` defaults to 4096 no matter what the model advertises, and a longer prompt is
cut **from the front** with no warning — the model then answers confidently from what's left.

So this tool always sends `num_ctx`, always checks how much of your prompt was actually read,
and stops rather than hand you an answer built on half a question. If it refuses, send less or
raise `--num-ctx`.

---

## Using it from Python

```python
from ollama_stack import OllamaClient

client = OllamaClient(num_ctx=32768)
reply = client.generate("summarise this", "qwen", context=open("notes.md").read())
print(reply.text)
```

`stream()`, `chat()`, `start`/`stop` via `load()` and `unload()`. Model aliases live in
`src/ollama_stack/models.py`.

---

## Coming

Interactive setup, saved config, web search for questions past the model's cutoff, images,
follow-up questions, and an MCP server so editors can call local models as tools.

---

## Worth knowing

A local model is useful and it is not a senior engineer. `o audit` finding nothing means the
file is unexamined, not clean.

---

## Development

```sh
uv sync --all-extras
uv run ruff check --fix . && uv run mypy . && uv run pytest
```

MIT — see [LICENSE](LICENSE).
