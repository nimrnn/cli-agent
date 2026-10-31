"""Terminal chat agent: web search/fetch tools + a Docker-sandboxed command runner, with resumable sessions."""

# ============================================================
# IMPORTS
# ============================================================

# --- Standard library (ships with Python) ---
import argparse  # turns terminal flags like --session into Python values
import asyncio  # lets slow work (network, waiting) run without freezing the program
import os  # reads environment variables (where the API key lives)
import posixpath  # path math with Linux rules; the sandbox is Linux even if your PC isn't
import shlex  # splits a command string into a list of words, respecting quotes
import sqlite3  # imported only for its error type, to catch broken session files
import subprocess  # runs other programs (here: docker)
import sys  # sys.exit() stops the program with a message
import tempfile  # creates a temporary folder that deletes itself
import time  # clock, used to animate the spinner
import uuid  # generates unique random IDs for names
from contextlib import contextmanager  # turns a function into a reusable "with" block
from pathlib import Path  # file paths as objects instead of raw strings

# --- Third-party (installed with pip) ---
import httpx  # HTTP client that supports async
from agents import (  # OpenAI Agents SDK
    Agent,  # the assistant: model + instructions + tools
    OpenAIChatCompletionsModel,  # adapter for servers that speak the "Chat Completions" API (Ollama does)
    Runner,  # executes an agent on a user message
    SQLiteSession,  # stores chat history in a SQLite file so chats can resume
    function_tool,  # decorator that turns a Python function into a tool the AI can call
    set_tracing_disabled,  # turns off the SDK's built-in telemetry
)
from openai import AsyncOpenAI  # client library; works with any OpenAI-compatible server
from openai.types.responses import ResponseTextDeltaEvent  # event type that carries streamed text chunks
from rich.console import Console  # prints styled text to the terminal
from rich.live import Live  # redraws one screen region repeatedly (for streaming)
from rich.markdown import Markdown  # renders markdown (bold, lists, code) in the terminal
from rich.table import Table  # used as a layout grid to align the bullet and the text
from rich.text import Text  # a string that carries styling

# ============================================================
# CONFIGURATION
# ============================================================
# Constants live at the top so you can change a value in one place.
# UPPER_CASE is the Python convention for "never reassigned".

OLLAMA_URL = "https://ollama.com"  # base address of the API
MODEL_NAME = "gpt-oss:120b-cloud"  # which model answers
SESSIONS_DIR = Path("sessions")  # folder that holds saved chats
MAX_RESULTS_LIMIT = 10  # most search results the AI may request
MAX_CHARS = 10000  # most text returned to the AI per tool call (protects its memory)
HTTP_TIMEOUT = 30  # seconds before a web request gives up
COMMAND_TIMEOUT = 15  # seconds before a sandbox command is abandoned
WORKDIR = "/workspace"  # folder inside the container where commands run

# Programs the AI may not run. A set gives fast "is x in here?" checks.
FORBIDDEN_COMMANDS = {"rm", "dd", "mkfs", "chmod", "chown", "sudo", "su", "sh", "bash"}
# System folders the AI may not touch.
SENSITIVE_DIRS = ("/etc", "/bin", "/sbin", "/usr", "/var", "/boot", "/sys", "/proc")
# Shell startup files the AI may not touch, wherever they are.
SENSITIVE_FILES = (".bashrc", ".bash_profile", ".zshrc", ".profile")

# Words that end the chat instead of being sent to the AI.
EXIT_COMMANDS = {"exit", "quit"}
# Docker's "not installed" text lives in one place, so both the startup check
# and the tool error say the same thing.
DOCKER_MISSING = "Docker is not installed or not available in PATH."


# ============================================================
# STARTUP HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments passed when running the script."""
    parser = argparse.ArgumentParser(description="Chat agent with resumable sessions.")
    # "--session" is optional because it starts with "--".
    parser.add_argument(
        "--session",
        metavar="FILE",  # placeholder name shown in --help
        help="session file to resume, e.g. 3f2a9c1e-....sqlite (looked up in sessions/)",
    )
    return parser.parse_args()  # reads sys.argv and returns an object: args.session


def get_api_key() -> str:
    """Retrieves the required API key from environment variables, or exits if missing."""
    # Keys live in the environment, not in code, so they never end up in shared files.
    key = os.environ.get("OLLAMA_API_KEY")  # .get returns None instead of crashing if unset
    if not key:  # catches both None and an empty string
        sys.exit("OLLAMA_API_KEY is not set.")  # prints the message and stops the program
    return key


# ============================================================
# COMMAND SAFETY CHECKS
# ============================================================
# Docker is the real security wall. These checks are a second layer
# that stops obviously bad commands before they reach the container.


def is_sensitive_path(arg: str) -> bool:
    """Returns True if an argument points at a protected file or folder inside the sandbox."""
    # Resolve the argument the way the container would: relative paths start at WORKDIR,
    # and normpath collapses "..", so "../etc" becomes "/etc" and can't dodge the check.
    path = posixpath.normpath(posixpath.join(WORKDIR, arg))
    # Block shell startup files by file name.
    if posixpath.basename(path) in SENSITIVE_FILES:
        return True
    # Block "/" itself, each protected folder, and anything inside one.
    # Adding "/" before startswith stops "/etc_notes" from matching "/etc".
    return path == "/" or any(path == d or path.startswith(d + "/") for d in SENSITIVE_DIRS)


def check_command(args: list[str]) -> str | None:
    """Returns an error message if the command breaks a rule, or None if it is allowed."""
    # basename() reduces "/bin/rm" to "rm", so a full path can't bypass the blocklist.
    program = posixpath.basename(args[0])
    if program in FORBIDDEN_COMMANDS:
        return f"Security Error: Execution of '{program}' is strictly forbidden."
    for arg in args[1:]:  # args[1:] = everything after the program name
        if is_sensitive_path(arg):
            return f"Security Error: Access to sensitive path or file in argument '{arg}' is blocked."
    return None  # None means "no problems found"


# ============================================================
# TOOLS (functions the AI can call)
# ============================================================
# Each tool is built inside a factory function. That lets the tool use
# api_key / container_name without exposing them to the AI as parameters.
# The AI reads each tool's name, parameter types and docstring to decide when to use it,
# so those docstrings are instructions for the AI, not just notes for you.


async def ollama_post(endpoint: str, api_key: str, payload: dict) -> dict:
    """Sends a POST request to an Ollama endpoint and returns the reply as a dict."""
    # "async with" opens the client and guarantees it closes, even if an error occurs.
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http:
        # "await" pauses this function until the server answers, while other code can run.
        resp = await http.post(
            f"{OLLAMA_URL}{endpoint}",
            headers={"Authorization": f"Bearer {api_key}"},  # proves who you are
            json=payload,  # httpx converts the dict to JSON text
        )
        resp.raise_for_status()  # turns HTTP errors (401, 500...) into exceptions instead of bad data
        return resp.json()  # parses the reply's JSON text into a dict


def make_web_search_tool(api_key: str):
    """Builds the search tool with the API key baked in."""

    @function_tool  # registers the function as an AI-callable tool
    async def web_search(query: str, max_results: int = 5) -> str:
        """Performs a web search for a single query and returns relevant results."""
        # Force the number into the range 1..MAX_RESULTS_LIMIT so the AI can't request thousands.
        max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))
        try:
            data = await ollama_post(
                "/api/web_search", api_key, {"query": query, "max_results": max_results}
            )
        except (httpx.HTTPError, ValueError) as e:  # ValueError is what .json() raises on bad JSON
            # Return the error as text instead of crashing, so the AI can read it and adapt.
            return f"Search failed: {e}"

        # Format each result as "title / url / content", separated by blank lines.
        text = "\n\n".join(
            f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}"
            for r in data.get("results", [])  # .get avoids a crash if "results" is missing
        )
        # Cut to MAX_CHARS; "or" supplies a fallback message when the text is empty.
        return text[:MAX_CHARS] or "No results."

    return web_search  # hand the finished tool back to the caller


def make_web_fetch_tool(api_key: str):
    """Builds the page-fetching tool with the API key baked in."""

    @function_tool
    async def web_fetch(url: str) -> str:
        """Fetches a single web page by URL and returns its content."""
        try:
            data = await ollama_post("/api/web_fetch", api_key, {"url": url})
        except (httpx.HTTPError, ValueError) as e:
            return f"Fetch failed: {e}"

        content = data.get("content", "")[:MAX_CHARS]  # trim long pages
        if not content:
            return "No content."
        # Include the page's links so the AI can follow them with another fetch.
        links = "\n".join(data.get("links", []))
        return f"{data.get('title', '')}\n\n{content}\n\nLinks:\n{links}"

    return web_fetch


def make_bash_tool(container_name: str):
    """Builds the command tool, bound to one running sandbox container."""

    @function_tool
    async def bash(cmd: str) -> str:
        """Runs one command inside the persistent Docker sandbox. It is not a shell: pipes (|), redirects (>) and && do not work."""
        # shlex.split turns the string into a list, so "echo 'a b'" becomes ["echo", "a b"].
        # Passing a list (not a string) to subprocess means no shell ever interprets the text,
        # which blocks injection tricks like "ls; rm -rf /".
        try:
            args = shlex.split(cmd)
        except ValueError as e:  # raised on unbalanced quotes
            return f"Parsing error (check your quotes): {e}"

        if not args:  # empty or whitespace-only command
            return "Error: Empty command provided."

        error = check_command(args)
        if error:
            return error  # tell the AI why it was refused

        try:
            # "docker exec" runs inside the container that is already running,
            # which is instant, unlike starting a new container per command.
            # subprocess.run blocks until done, so to_thread moves it to a side thread;
            # that keeps the spinner animating while the command runs.
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "exec", container_name, *args],  # * unpacks the list into separate items
                capture_output=True,  # collect output instead of printing it to your terminal
                text=True,  # give back strings instead of raw bytes
                timeout=COMMAND_TIMEOUT,  # raises TimeoutExpired if exceeded
            )
        except subprocess.TimeoutExpired:
            return "Sandbox Error: Execution timed out."
        except FileNotFoundError:  # the "docker" program itself wasn't found
            return f"Error: {DOCKER_MISSING}"

        # Return both streams: programs often print useful info to stderr even on success.
        output = (result.stdout + result.stderr)[:MAX_CHARS]
        if result.returncode != 0:  # non-zero exit code means the command failed
            output += f"\n[exit code {result.returncode}]"
        return output or "(no output)"  # an empty reply would leave the AI guessing

    return bash


# ============================================================
# AGENT
# ============================================================

# The system prompt: standing instructions the AI sees before every conversation.
SYSTEM_PROMPT = "You are a helpful AI assistant equipped with web tools and a fast, persistent sandboxed bash execution environment. Personalization: Explain concepts directly without filler. Begin with a concise, one-sentence summary of the core idea, followed by an explanation in plain language. Use simple, relatable analogies (e.g., 'X is the phone, Y is the adapter') rather than heavy jargon. Focus on why a concept matters, not just what it is. Keep explanations brief and readable in a single glance. Include a small diagram or flow only when it is necessary to clarify how parts connect. Maintain an intelligent yet accessible tone: avoid defining obvious foundational terms, but ensure logical steps are never skipped. Strictly omit motivational language, hype, or compliments."


def build_agent(api_key: str, container_name: str) -> Agent:
    """Builds and returns the AI agent, binding its tools and configuration model."""
    # Ollama copies OpenAI's API layout, so the OpenAI client works with just a different address.
    client = AsyncOpenAI(base_url=f"{OLLAMA_URL}/v1", api_key=api_key)
    # The SDK defaults to OpenAI's newer API; this adapter makes it use the older one Ollama supports.
    model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)

    return Agent(
        name="Assistant",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=[
            make_web_search_tool(api_key),
            make_web_fetch_tool(api_key),
            make_bash_tool(container_name),
        ],
    )


# ============================================================
# TERMINAL DISPLAY
# ============================================================

# One shared printer. Rich only applies style codes to text that goes through a Console,
# so using the same instance everywhere keeps the output consistent.
console = Console()


class InlineDotSpinner:
    """Animated '● .' → '● ..' → '● ...' shown while waiting for the model."""

    def __init__(self) -> None:
        # monotonic() only moves forward, so a system clock change can't make elapsed negative.
        self._start_time = time.monotonic()  # remember when the wait began

    def __rich__(self) -> Text:
        # Rich calls this method on every refresh to ask "what should I draw right now?".
        elapsed = time.monotonic() - self._start_time
        # elapsed * 3 = three steps per second; % 3 cycles 0,1,2; + 1 makes it 1,2,3 dots.
        dots = "." * (int(elapsed * 3) % 3 + 1)
        return Text(f"● {dots}", style="bold cyan")


def render_reply(text: str) -> Table:
    """Puts the cyan bullet and the markdown reply side by side on one row."""
    # A borderless table acts as a layout grid; padding=(0, 1) = no vertical, 1 space horizontal.
    grid = Table.grid(padding=(0, 1))
    # Each add_row argument becomes one column, so the bullet stays aligned with the first line.
    grid.add_row(Text("●", style="bold cyan"), Markdown(text))
    return grid


async def stream_reply(agent: Agent, session: SQLiteSession, message: str) -> None:
    """Sends one message to the agent and draws the reply as it arrives."""
    # run_streamed returns immediately; the reply is pulled out of it piece by piece below.
    # session= makes the SDK load the old messages first and save the new ones afterward.
    result = Runner.run_streamed(agent, message, session=session)
    buffer = ""  # all reply text received so far

    # Live redraws this region 10 times a second. It starts with the spinner,
    # and update() swaps in the real text once it begins arriving.
    with Live(InlineDotSpinner(), console=console, refresh_per_second=10) as live:
        async for event in result.stream_events():  # yields events as the model produces them
            # Many event types exist (tool calls, etc.); only text chunks are drawn.
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                buffer += event.data.delta  # append the new chunk
                # Rebuild the display from the full text, because markdown needs the whole
                # text to format correctly. Live's refresh thread handles the actual redraw.
                live.update(render_reply(buffer))


async def chat_loop(agent: Agent, session: SQLiteSession) -> None:
    """Manages the interactive command-line chat loop and input parsing."""
    while True:  # repeats until a break
        try:
            # input() blocks until Enter is pressed. to_thread runs it on a side thread
            # so the async event loop stays free instead of freezing.
            message = (await asyncio.to_thread(input, "❯ ")).strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D or Ctrl+C
            console.print()  # move to a fresh line
            break

        if not message:  # empty input: ask again
            continue
        if message.lower() in EXIT_COMMANDS:  # .lower() makes this case-insensitive
            console.print("\nBye-bye!")
            break
        if message.lower() == "clear":
            await session.clear_session()  # wipes saved history so the AI forgets the chat
            console.print("\nConversation history cleared.\n")
            continue  # skip sending this word to the AI

        try:
            await stream_reply(agent, session, message)
        except Exception as e:  # a network or API failure shouldn't kill the whole chat
            # Text() prints the message literally; a plain string could mistake "[...]" for styling codes.
            console.print(Text(f"Error: {e}", style="red"))
        console.print()  # blank line between turns


# ============================================================
# SESSIONS
# ============================================================


def resolve_session_path(name: str | None) -> Path:
    """Resolves and validates the path where chat session databases are stored."""
    # Create the folder up front, before anything tries to read from or write to it.
    SESSIONS_DIR.mkdir(exist_ok=True)  # exist_ok avoids an error if the folder already exists
    if name:  # the user passed --session
        # Path(name).name keeps only the file name, so "../../secret" can't escape SESSIONS_DIR.
        filename = Path(name).name
        if not filename.endswith(".sqlite"):
            filename += ".sqlite"  # lets the user omit the extension
        path = SESSIONS_DIR / filename  # "/" joins paths when the left side is a Path
        if not path.is_file():
            sys.exit(f"\nSession not found: {path}")
        return path

    # No --session given: resume the most recently used session, if any exist.
    existing = [f for f in SESSIONS_DIR.glob("*.sqlite") if f.is_file()]
    if existing:
        # max() with key= picks the file with the latest modification time.
        return max(existing, key=lambda f: f.stat().st_mtime)

    # First run ever: pick a unique name for a new session.
    return SESSIONS_DIR / f"{uuid.uuid4()}.sqlite"


async def open_session(path: Path) -> SQLiteSession:
    """Opens and validates an SQLite session database file."""
    try:
        # First argument = session ID (the file name), second = the file that stores it.
        session = SQLiteSession(path.stem, path)
        # Read one row only to prove the file is a real database.
        await session.get_items(limit=1)
    except sqlite3.DatabaseError as e:  # raised when the file isn't valid SQLite
        sys.exit(f"Cannot use {path} as a session: {e}\nPass --session to pick another file.")
    return session


# ============================================================
# SANDBOX (Docker container)
# ============================================================


def _sandbox_run_args(name: str, scratch_dir: str) -> list[str]:
    """Builds the `docker run` argument list that creates the sandbox container."""
    return [
        "docker", "run",
        "-d",  # detached: start in the background and return immediately
        "--name", name,  # a known name so we can target it later
        "--network", "none",  # no network: code in the sandbox can't reach the internet
        "--memory", "512m",  # cap RAM so a runaway program can't eat your computer
        "--cpus", "1.0",  # cap CPU for the same reason
        "-v", f"{scratch_dir}:{WORKDIR}",  # mount: host folder : container folder
        "-w", WORKDIR,  # commands start in this folder
        "python:3.11-slim",  # the image: a small Linux with Python
        "tail", "-f", "/dev/null",  # a command that never ends, keeping the container alive
    ]


@contextmanager
def sandbox():
    """Starts a throwaway Docker container, yields its name, and always removes it afterward."""
    # In a @contextmanager function, code before "yield" is setup and code after it
    # is cleanup. The cleanup runs even if the chat crashes.
    name = f"agent_sandbox_{uuid.uuid4().hex[:8]}"  # short random suffix avoids name clashes

    # Shared folder between your computer and the container. ignore_cleanup_errors stops a crash
    # when files created by the container (as root) can't be deleted by your user.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch_dir:
        # The first run downloads the image, which can take a while, so tell the user.
        console.print("Starting sandbox (the first run downloads the Docker image)...")
        try:
            proc = subprocess.run(
                _sandbox_run_args(name, scratch_dir), capture_output=True, text=True
            )
        except FileNotFoundError:  # the "docker" program itself wasn't found
            sys.exit(DOCKER_MISSING)

        if proc.returncode != 0:  # 0 = success; anything else = failure
            sys.exit(f"Failed to start Docker sandbox container: {proc.stderr}")

        try:
            yield name  # the "with sandbox() as name" block runs here
        finally:
            # "rm -f" force-removes a running container instantly. A plain "stop" would wait
            # ~10 seconds because "tail" ignores the polite shutdown signal.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ============================================================
# ENTRY POINT
# ============================================================


async def main() -> None:
    """Initializes the session, sets up the sandbox, and starts the chat loop."""
    args = parse_args()
    set_tracing_disabled(True)  # tracing uploads to OpenAI by default; you aren't using OpenAI
    api_key = get_api_key()

    path = resolve_session_path(args.session)
    session = await open_session(path)  # await: this function is async
    console.print(f"Session: {path.stem}\n")  # .stem = file name without ".sqlite"

    # Everything inside this block has a live sandbox. When it ends (or crashes),
    # the container is removed automatically.
    with sandbox() as container_name:
        agent = build_agent(api_key, container_name)
        await chat_loop(agent, session)


if __name__ == "__main__":  # True only when run directly, not when imported by another file
    try:
        asyncio.run(main())  # starts the async engine and runs main() until it finishes
    except KeyboardInterrupt:
        pass  # Ctrl+C mid-reply exits quietly; the sandbox cleanup has already run
