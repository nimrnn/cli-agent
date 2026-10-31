# cli-agent

A small terminal chat agent with three tools: **web search**, **web fetch**, and **bash** running inside a disposable Docker sandbox.

Built on the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), with [rich](https://github.com/Textualize/rich) streaming markdown replies to the terminal.

## How it works

```
you ──▶ chat loop ──▶ Agent (OpenAI Agents SDK) ──▶ Ollama (OpenAI-compatible API)
                          │
                          ├── web_search   → Ollama /api/web_search
                          ├── web_fetch    → Ollama /api/web_fetch
                          └── bash         → docker exec ──▶ sandboxed container
```

- `Agent`, `Runner.run_streamed`, `@function_tool`, and `SQLiteSession` come from the OpenAI Agents SDK.
- The SDK talks to Ollama through `OpenAIChatCompletionsModel`, which uses the older Chat Completions API that Ollama supports.
- The `rich` `Live` component redraws the reply as text chunks arrive; `Markdown` renders the final formatting.
- Chat history is stored in a SQLite file under `sessions/`, so conversations resume where you left off.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker
- An Ollama API key

## Setup

```bash
git clone https://github.com/nimrnn/cli-agent.git
cd cli-agent
uv sync
export OLLAMA_API_KEY="your-key-here"
```

## Usage

```bash
# Resume the most recent session, or start a new one on first run
uv run app.py

# Resume a specific session by file name (extension optional)
uv run app.py --session 3f2a9c1e-....sqlite
```

In-chat commands:

| Input | Effect |
| --- | --- |
| `exit` / `quit` | End the chat |
| `clear` | Wipe the saved history for this session |
| Ctrl+C / Ctrl+D | End the chat |

The first run downloads the `python:3.11-slim` Docker image, which can take a minute.

## Tools

### `web_search`

Sends a query to Ollama's `/api/web_search` endpoint and returns up to 10 results as `title / url / content`.

### `web_fetch`

Fetches a single URL through `/api/web_fetch`, returns the page content, and appends its links so the agent can follow them.

### `bash`

Runs **one** command inside the sandbox with `docker exec`. It is not a shell, so pipes (`|`), redirects (`>`), and `&&` do not work. Arguments are split with `shlex.split` and passed as a list, which prevents shell injection.

The container is created once per chat and reused for every command:

- no network access (`--network none`)
- 512 MB RAM and 1 CPU limit
- 15-second command timeout
- a temporary host folder mounted at `/workspace`, deleted when the chat ends
- output capped at 10,000 characters per tool call

## Safety

Docker is the real security boundary. On top of that, `app.py` applies a second layer:

- a blocklist of destructive programs (`rm`, `dd`, `mkfs`, `sudo`, shells, ...)
- path checks that reject system folders (`/etc`, `/usr`, `/proc`, ...) and shell startup files, after normalizing `..`

The sandbox container is force-removed when the chat exits, even if it crashes.

## Configuration

Constants live at the top of `app.py`:

| Constant | Purpose |
| --- | --- |
| `OLLAMA_URL` | Ollama API base URL |
| `MODEL_NAME` | Model that answers |
| `MAX_RESULTS_LIMIT` | Cap on search results per call |
| `MAX_CHARS` | Cap on text returned by a tool |
| `HTTP_TIMEOUT` | Web request timeout |
| `COMMAND_TIMEOUT` | Sandbox command timeout |
| `WORKDIR` | Working folder inside the container |

## Project layout

```
app.py                  the whole agent: config, tools, UI, sessions, sandbox
sessions/               local SQLite chat history (gitignored)
src/cli_agent/          package placeholder
pyproject.toml          dependencies and project metadata
```
