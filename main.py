from __future__ import annotations

import argparse
import os
import sys
import json
from typing import Any, List, Optional, Tuple

from mapgpt.agent import run_agent
from mapgpt.tools import get_tool_names, get_tools_prompt_string

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def serialize_messages(messages: List[Any]) -> List[dict]:
    """
    Converts a list of LangChain message objects to a JSON-serializable format.
    Handles Function Calling messages (ToolMessage & tool_calls).
    """
    serializable_history = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        role = "unknown"
        content = ""

        if isinstance(msg, HumanMessage):
            role = "human"
            content = msg.content
        elif isinstance(msg, AIMessage):
            role = "ai"
            if msg.tool_calls:
                calls_desc = [f"Call Tool: {tc['name']}({tc['args']})" for tc in msg.tool_calls]
                content = "\n".join(calls_desc)
            else:
                content = msg.content
        elif isinstance(msg, ToolMessage):
            role = "tool"
            content = f"Tool Output: {msg.content}"
        else:
            continue

        if not isinstance(content, str):
            content = str(content)

        if content.strip():
            serializable_history.append({"role": role, "content": content})

    return serializable_history


def load_and_prepare_history(filepath: str) -> Optional[List[Tuple[str, str]]]:
    """Loads a session file and prepares it for the agent's prompt."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            saved_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}. Starting fresh.")
            return None

    # Convert to the List[Tuple[str, str]] format expected by _format_chat_history
    formatted_history: List[Tuple[str, str]] = []
    for item in saved_data:
        role = item.get("role")
        content = item.get("content", "")

        if role == "human":
            formatted_role = "User"
        elif role == "ai":
            formatted_role = "Assistant"
        elif role == "tool":
            formatted_role = "System (Tool)"
        else:
            continue

        formatted_history.append((formatted_role, content))
    return formatted_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MapGPT Agent CLI")
    parser.add_argument(
        "-q", "--question", type=str, required=False, default=None,
        help="User instruction in natural language",
    )

    parser.add_argument(
        "-s", "--session-file", type=str, default=None,
        help="Path to a JSON file to load and save the conversation history."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=False, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model name (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=60, help="Max Thought-Action-Observation steps",
    )
    parser.add_argument(
        "--list-tools", action="store_true", help="List available tools and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_tools:
        print("Available tools:")
        print(get_tools_prompt_string())
        return 0

    question = args.question
    if not question:
        print("Please provide a question via -q/--question. Example:")
        print(""" python main.py -q "1.用'/Users/henry/Desktop/Cehui_data/data/房屋_曹杨新村街道.shp'初始化底图;2.设置底图颜色为白色;3.添加'/Users/henry/Desktop/Cehui_data/data/房屋_曹杨新村街道.shp';4.将结果保存到'/Users/henry/Desktop/Cehui_data/data/out.jpg'" """)
        return 1

    chat_history = None
    if args.session_file:
        print(f"--- Loading history from {args.session_file} ---")
        chat_history = load_and_prepare_history(args.session_file)
        if chat_history:
            print(f"--- Loaded {len(chat_history)} previous messages ---")

    try:
        final_answer, final_messages = run_agent(
            question=question,
            chat_history=chat_history,
            llm_model_name=args.model,
            max_steps=args.max_steps,
        )

        print("\n=== Final Answer ===")
        print(final_answer)

        if args.session_file:
            print(f"\n--- Saving final history to {args.session_file} ---")
            serializable_data = serialize_messages(final_messages)
            with open(args.session_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print("--- History saved. ---")

    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())