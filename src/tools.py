"""
Tool definitions and execution for the reactive agent.

Each tool has:
  - An OpenAI-compatible function schema (for the LLM).
  - A Python implementation that receives parsed arguments and returns a string result.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Maps tool name -> implementation function
_TOOL_IMPLS: dict[str, Callable[..., str]] = {}

# Base tool definitions sent to the model
BASE_TOOL_DEFINITIONS: list[dict[str, Any]] = []

# Optional orchestration tools (wired by Agent at runtime)
ORCHESTRATION_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sessions_spawn",
            "description": "Spawn a subagent session and optionally run an initial task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Initial task for the subagent."},
                    "provider": {"type": "string", "description": "Optional provider override."},
                    "model": {"type": "string", "description": "Optional model override."},
                    "run_now": {
                        "type": "boolean",
                        "description": "If true, run the task immediately and return a first reply.",
                        "default": True,
                    },
                    "background": {
                        "type": "boolean",
                        "description": "If true and run_now=true, run in background and return immediately.",
                        "default": True,
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subagents",
            "description": "List, steer, or kill existing subagent runs for this session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get_result", "events", "steer", "kill"],
                        "description": "Operation to perform.",
                    },
                    "run_id": {"type": "string", "description": "Subagent run id (required for steer/kill)."},
                    "message": {"type": "string", "description": "Message for steer action."},
                    "background": {
                        "type": "boolean",
                        "description": "If true for steer, run in background and return immediately.",
                        "default": False,
                    },
                },
                "required": ["action"],
            },
        },
    },
]


def _register(
    name: str,
    description: str,
    parameters: dict[str, Any],
    fn: Callable[..., str],
) -> None:
    """Register a tool (schema + implementation)."""
    _TOOL_IMPLS[name] = fn
    BASE_TOOL_DEFINITIONS.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _read_file(path: str) -> str:
    """Read the contents of a file and return them as a string."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return f"Error: file not found: {resolved}"
    try:
        return resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading {resolved}: {exc}"


def _write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {resolved}"
    except Exception as exc:
        return f"Error writing {resolved}: {exc}"


def _list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return f"Error: not a directory: {resolved}"
    try:
        entries = sorted(resolved.iterdir())
        lines: list[str] = []
        for entry in entries:
            prefix = "d " if entry.is_dir() else "f "
            lines.append(f"{prefix}{entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as exc:
        return f"Error listing {resolved}: {exc}"


def _run_cmd(command: str, cwd: str | None = None) -> str:
    """Run a shell command and return combined stdout+stderr."""
    work_dir = cwd or os.getcwd()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=work_dir,
        )
        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s limit)"
    except Exception as exc:
        return f"Error running command: {exc}"


# Default max size for web_fetch to avoid blowing context (chars).
_WEB_FETCH_MAX_CHARS = 80_000
_WEB_FETCH_TIMEOUT_SECONDS = 15
# User-Agent so some sites don't reject the request.
_WEB_FETCH_USER_AGENT = "AgentSpine/1.0 (web_fetch; +https://github.com/bmcool/AgentSpine)"


def _web_fetch(url: str, max_chars: int = _WEB_FETCH_MAX_CHARS) -> str:
    """Fetch a URL and return its body as text. Truncated to max_chars for context safety."""
    if not url or not url.strip():
        return "Error: url is required and must be non-empty."
    url = url.strip()
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return f"Error: invalid url (missing scheme or host): {url}"
    if parsed.scheme not in ("http", "https"):
        return f"Error: only http and https are allowed; got scheme: {parsed.scheme}"
    try:
        req = Request(url, headers={"User-Agent": _WEB_FETCH_USER_AGENT})
        with urlopen(req, timeout=_WEB_FETCH_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
            if "charset" in (resp.headers.get("content-type") or ""):
                # Try to extract charset from header
                ct = (resp.headers.get("content-type") or "").lower()
                m = re.search(r"charset=([\w-]+)", ct)
                encoding = m.group(1) if m else "utf-8"
            else:
                encoding = "utf-8"
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            if len(text) <= max_chars:
                return text
            return (
                text[: max_chars - 200]
                + "\n\n...[truncated: "
                + str(len(text) - max_chars)
                + " chars omitted for context]..."
            )
    except HTTPError as e:
        return f"Error: HTTP {e.code} {e.reason} for {url}"
    except URLError as e:
        return f"Error: request failed for {url}: {e.reason}"
    except TimeoutError:
        return f"Error: request timed out ({_WEB_FETCH_TIMEOUT_SECONDS}s) for {url}"
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


# ---------------------------------------------------------------------------
# Register tools
# ---------------------------------------------------------------------------

_register(
    name="read_file",
    description="Read the full contents of a file at the given path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path to read."},
        },
        "required": ["path"],
    },
    fn=_read_file,
)

_register(
    name="write_file",
    description="Write content to a file. Creates parent directories if they don't exist.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write to."},
            "content": {"type": "string", "description": "Content to write into the file."},
        },
        "required": ["path", "content"],
    },
    fn=_write_file,
)

_register(
    name="list_directory",
    description="List files and subdirectories at the given path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current directory.",
                "default": ".",
            },
        },
        "required": [],
    },
    fn=_list_directory,
)

_register(
    name="run_cmd",
    description="Execute a shell command and return its stdout, stderr, and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "cwd": {
                "type": "string",
                "description": "Working directory for the command. Defaults to the current directory.",
            },
        },
        "required": ["command"],
    },
    fn=_run_cmd,
)

_register(
    name="web_fetch",
    description=(
        "Fetch the content of a URL (http/https) and return it as text. "
        "Use this to read a web page or API response when you don't have a search API: "
        "e.g. fetch a search result page HTML and parse links, then fetch target pages as needed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to fetch (e.g. https://example.com/page)."},
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return; response is truncated if longer. Default 80000.",
            },
        },
        "required": ["url"],
    },
    fn=_web_fetch,
)


# ---------------------------------------------------------------------------
# Execution dispatcher
# ---------------------------------------------------------------------------


def execute_tool(
    name: str,
    arguments_json: str,
    runtime_hooks: dict[str, Callable[..., str]] | None = None,
    extra_handlers: dict[str, Callable[..., str]] | None = None,
) -> str:
    """
    Look up a tool by name, parse its JSON arguments, and run it.
    Returns the result string (or an error message).
    extra_handlers: optional map of name -> handler for embedding-project tools.
    """
    fn = _TOOL_IMPLS.get(name)
    if fn is None and runtime_hooks:
        fn = runtime_hooks.get(name)
    if fn is None and extra_handlers:
        fn = extra_handlers.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        args: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return f"Error: failed to parse tool arguments: {exc}"
    try:
        return fn(**args)
    except TypeError as exc:
        return f"Error: bad arguments for tool '{name}': {exc}"
    except Exception as exc:
        return f"Error: tool '{name}' raised: {exc}"


def get_tool_definitions(
    *,
    include_orchestration: bool,
    extra_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tools = list(BASE_TOOL_DEFINITIONS)
    if include_orchestration:
        tools.extend(ORCHESTRATION_TOOL_DEFINITIONS)
    if extra_tools:
        for t in extra_tools:
            definition = t.get("definition") if isinstance(t, dict) else None
            if isinstance(definition, dict) and definition.get("type") == "function" and "function" in definition:
                tools.append(definition)
    return tools


def get_tool_summaries(
    *,
    include_orchestration: bool,
    extra_tools: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Return a compact [(tool_name, description)] list for prompt building."""
    summaries: list[tuple[str, str]] = []
    for tool in get_tool_definitions(
        include_orchestration=include_orchestration,
        extra_tools=extra_tools,
    ):
        fn = tool.get("function", {})
        name = fn.get("name")
        description = fn.get("description")
        if isinstance(name, str) and isinstance(description, str):
            summaries.append((name, description))
    return summaries
