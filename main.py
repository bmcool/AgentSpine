"""
AgentSpine CLI entry point.

Usage:
  1. Copy .env.example to .env and fill in provider API keys.
  2. pip install -r requirements.txt
  3. python main.py --provider openai --model gpt-4o
     python main.py --provider anthropic --model claude-3-5-sonnet-20241022

Type your message and press Enter. The agent will use tools as needed
and print the final reply. Type "exit" or "quit" to leave.
Type "/reset" to clear conversation history and start fresh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env from the project root
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

from src.agent import Agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentSpine reactive CLI")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--session", dest="session_id", default=None)
    parser.add_argument("--workspace", default=None, help="Workspace root for prompt/runtime context")
    parser.add_argument("--sessions-dir", default=None, help="Directory for JSONL session files")
    parser.add_argument(
        "--thinking",
        dest="thinking_level",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning level (provider/model support dependent).",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming text output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = Agent(
        provider=args.provider,
        model=args.model,
        session_id=args.session_id,
        workspace_dir=args.workspace,
        sessions_dir=args.sessions_dir,
        thinking_level=args.thinking_level,
    )
    print("AgentSpine - reactive coding agent")
    print(f"provider/model: {agent.provider_name}/{agent.model}")
    print(f"session id: {agent.session.meta.session_id}")
    print(f"sessions dir: {agent.store.sessions_dir}")
    print(f"session messages: {len(agent.session)}")
    print('Type your message (or "exit" to quit, "/reset" to clear history).')
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if user_input == "/reset":
            agent.reset()
            print("[session reset]")
            continue

        try:
            streamed = {"used": False}

            def on_delta(delta: str) -> None:
                if not delta:
                    return
                streamed["used"] = True
                print(delta, end="", flush=True)

            if args.no_stream:
                reply = agent.chat(user_input)
            else:
                reply = agent.chat_stream(user_input, on_text_delta=on_delta)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue

        if not args.no_stream and streamed["used"]:
            print("")
        else:
            print(f"\n{reply}")


if __name__ == "__main__":
    main()
