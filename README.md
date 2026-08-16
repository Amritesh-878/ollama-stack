# ollama-stack

A fast command-line front end for local models running on [Ollama](https://ollama.com).

```sh
$ o what is 10+10
20
```

No subcommand, no quotes, streamed as it's produced. ~1s on a warm model.

> **Status: early.** Asking, streaming, `start`, `stop`, `status`, `ask`, `audit`, `models`,
> `which`. `setup`, `config`, web search and images are not built yet. Interfaces will change.

---

## Install

```sh
git clone https://github.com/Amritesh-878/ollama-stack.git
cd ollama-stack
uv sync
uv tool install .     # puts `o` on your PATH
```

Needs Ollama running, Python 3.12+, and at least one model pulled.

---

## Usage

```sh
o what is 10+10                       # just ask
o -m coder write a bash retry loop    # pick a model
cat main.py | o explain this          # stdin becomes context
o audit src/thing.py                  # screen one file for defects
o models                              # aliases, and which have measurements behind them
o which coder                         # what an alias resolves to
```

| Flag | Does |
| ---- | ---- |
| `-m, --model` | Alias or raw Ollama tag |
| `--num-ctx` | Context window to request. Default 32768. |
| `--no-stream` | Wait for the whole reply |
| `--stats` | Wall time, token rate, and the pre-flight estimate |
| `--dry-run` | Print what would be sent. Sends nothing. |
| `--version` | Print the version |

Answer goes to stdout, token counts to stderr — `o write a haiku > poem.txt` stays clean.

### Keeping a model warm

Cold load is ~10s, warm is ~1s, and Ollama drops a model after 5 minutes idle.

```sh
o start          # pin the default model in VRAM
o status         # what's loaded and how much of the card it holds
o stop           # release it
```

`o start` refuses if the card can't hold the model and names what's already on it. Nothing runs
in the background — Ollama holds the model itself.

### Bare questions starting with a command name

`o status of the economy` runs `status`. Use `o ask "status of the economy"` instead. Reserved
first words: `setup`, `start`, `stop`, `status`, `ask`, `audit`, `models`, `which`, `config`,
`implement`.

---

## The num_ctx thing

Ollama's `num_ctx` defaults to 4096 whatever the model advertises. Longer prompts are truncated
**from the front**, with no warning, and the model answers confidently from what's left. The
desktop app's *Context length* slider changes that default, so what you get depends on the
machine.

So this client always sends `num_ctx`, always checks `prompt_eval_count` on the way back, and
raises rather than returning a reply that reached the window — or one that omits the count,
since unknown isn't fine. Worth doing whether or not you use this tool.

Two checks: a pre-flight estimate (chars ÷ 4) refuses before sending, exit 2. After streaming,
the real count is checked — you've already seen the text, so it warns and exits non-zero anyway.

**The usable window is half of `num_ctx`**, so the default refuses at 16384. Generation needs
room in the same budget. That threshold rests on one measurement; raise `--num-ctx` if it fires
on a prompt you know is fine.

---

## As a library

```python
from ollama_stack import OllamaClient

client = OllamaClient(num_ctx=32768)
reply = client.generate("summarise this", "qwen", context=open("notes.md").read())
print(reply.text, reply.prompt_eval_count)

run = client.stream("summarise this", "qwen")
for chunk in run:
    print(chunk, end="", flush=True)
print(run.reply.prompt_eval_count)      # counts arrive with the last chunk
```

Also `chat(messages, model, tools=...)`, and `load()` / `unload()` / `ps()` / `tags()` behind the
lifecycle commands. `strict=False` returns `reply.suspect_truncation` instead of raising.

Default host is `127.0.0.1`, not `localhost` — Ollama binds IPv4 only, and resolving `localhost`
cost ~2s per connection here. Pass `host=` if yours is elsewhere.

Aliases live in `src/ollama_stack/models.py`. Any raw tag works too: `o -m llama4:8b hello`.

---

## Roadmap

`o setup` (interactive first run), `o config`, web search past the model's cutoff, `-i` for
images, `-c` for follow-ups, and an MCP server exposing local models as tools.

---

## Development

```sh
uv sync --all-extras
uv run ruff check --fix . && uv run mypy . && uv run pytest
```

Lint, typecheck, tests, in that order. No network calls in the suite.

---

## One caveat

Small local models are useful and they are not a senior engineer. Trust what one says it did,
never that it's finished. A "nothing found" from `o audit` means the file is unexamined, not
clean.

---

MIT — see [LICENSE](LICENSE).
