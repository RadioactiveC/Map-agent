from __future__ import annotations

import argparse
import os
import sys

from mapgpt.agent import run_agent
from mapgpt.tools import get_tool_names, get_tools_prompt_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MapGPT Agent CLI")
    parser.add_argument(
        "-q", "--question", type=str, required=False, default=None,
        help="User instruction in natural language",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=False, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model name (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=15, help="Max Thought-Action-Observation steps",
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
        print("python main.py -q '生成广东省行政地图，高速公路用白色粗线表示，保存到 /workspace/out.png' ")
        return 1

    final_answer = run_agent(
        question=question,
        chat_history=None,
        llm_model_name=args.model,
        max_steps=args.max_steps,
    )

    print("=== Final Answer ===")
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())