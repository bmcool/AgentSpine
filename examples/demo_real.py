from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentSpine real-provider demo")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--session", dest="session_id", default="demo-real")
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--sessions-dir", default=str(ROOT / "sessions"))
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Use list_directory on the workspace root, then summarize what you found.",
        help="First user prompt.",
    )
    parser.add_argument(
        "--continue-prompt",
        default="Now provide a one-sentence follow-up using the existing context.",
        help="User message appended before continue_run/continue_run_stream.",
    )
    return parser.parse_args()


def print_event(event: dict[str, Any]) -> None:
    et = event.get("type")
    if et == "message_update":
        return
    if et == "tool_execution_update":
        print(f"[event] {et}: {event.get('tool_name')} -> {event.get('partial')}")
        return
    if et in {"tool_execution_start", "tool_execution_end"}:
        print(f"[event] {et}: {json.dumps(event, ensure_ascii=False)}")
        return
    if et in {"turn_start", "turn_end", "agent_start", "agent_end"}:
        print(f"[event] {et}: {json.dumps(event, ensure_ascii=False)}")


def main() -> None:
    args = parse_args()
    print("=== AgentSpine Real Demo ===")
    print("This demo uses a real model provider and logs lifecycle/tool events.")
    print(f"provider/model: {args.provider}/{args.model or '(default)'}")

    agent = Agent(
        provider=args.provider,
        model=args.model,
        session_id=args.session_id,
        workspace_dir=args.workspace,
        sessions_dir=args.sessions_dir,
        on_event=print_event,
    )

    print("\n--- Demo 1: chat / chat_stream ---")
    print(f"[user] {args.prompt}")
    if args.no_stream:
        reply = agent.chat(args.prompt)
    else:
        streamed = {"used": False}

        def on_delta(delta: str) -> None:
            if not delta:
                return
            streamed["used"] = True
            print(delta, end="", flush=True)

        reply = agent.chat_stream(args.prompt, on_text_delta=on_delta)
        if streamed["used"]:
            print("")
    print(f"[assistant] {reply}")

    print("\n--- Demo 2: continue_run / continue_run_stream ---")
    print(f"[inject user message] {args.continue_prompt}")
    agent.session.add_user_message(args.continue_prompt)
    if args.no_stream:
        second = agent.continue_run()
    else:
        second = agent.continue_run_stream(on_text_delta=lambda d: print(d, end="", flush=True) if d else None)
        print("")
    print(f"[assistant] {second}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
