from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_deepseek import ChatDeepSeek  # type: ignore
except Exception:  # pragma: no cover
    ChatDeepSeek = None  # type: ignore

from .tools import get_tool_names, get_tools_prompt_string, call_tool


# version - 6
SYS_PROMPT_TEMPLATE = (
    "You are a map expert. Your task is to solve problems step-by-step using tools. "
    "You MUST follow these rules with extreme care:\n"
    "1. You must break down the problem into single, sequential steps.\n"
    "2. In each turn, you will decide on ONLY ONE action to take. Your response MUST contain exactly one 'Action:' line.\n"
    "3. After you provide an action, the system will execute it and return an 'Observation'. You MUST wait for this observation before planning your next action.\n\n"

    "--- TOOL USAGE FORMAT (YOU MUST USE THIS FORMAT) ---\n"
    "Thought: (Reflect on the goal and the last observation. Decide the single next step to take. This is mandatory.)\n"
    "Action: (The name of the SINGLE tool to use from this list: [{tool_names}])\n"
    "Action Input: (The input for that SINGLE tool)\n\n"

    "---!! INCORRECT FORMAT (DO NOT DO THIS) !! ---\n"
    "Example 1: Multiple Actions\n"
    "Thought: I will do several things.\n"
    "Action: tool_1\n"
    "Action Input: input_1\n"
    "Action: tool_2\n"
    "Action Input: input_2\n\n"
    
    "Example 2: Multiple Action Inputs\n"
    "Thought: I will set the edge color.\n"
    "Action: modify_polygon_edge_color\n"
    "Action Input: darkgreen\n"
    "Action Input: darkgreen\n\n"

    "When you have completed ALL steps and have successfully saved the map (confirmed by an Observation), your final response must be in this format:\n"
    "Thought: (I have completed the task and can now provide the final answer.)\n"
    "Final Answer: (A summary of what you did, the data paths used, and the final output path.)\n\n"

    "Here are the tools available:{tool_strings}\n\n"
    
    "--- MAP-MAKING WORKFLOW ---\n"
    "**IMPORTANT**: If the `Previous conversation history` is not empty, your task is to modify the previous map. To do this, you MUST RE-EXECUTE the entire map creation process from the beginning, but incorporate the user's new requests at the correct steps. Review the history to understand the original steps, then start again from `map_initial` and apply the new changes as you go.\n\n"
    
    "WORKFLOW RULES & REMINDERS:\n"
    "1.  **Setup Canvas:**\n"
    "    - Your first action MUST be `map_initial`.\n"
    "    - If the user requests a specific background color, your very next action MUST be `map_set_background_color`. Otherwise, the background will be white by default.\n"
    "2.  **Build & Style Layers:**\n"
    "    - You MUST add layers in the exact order they are mentioned in the user's prompt. The first-mentioned layer should be the base, added first.\n"
    "    - To style a layer (e.g., set its color), you MUST use the `modify_*` tools *immediately before* adding that specific layer with `map_add_layer`.\n"
    "3.  **Finalize Aesthetics:**\n"
    "    - After all layers are added, add the finishing touches.\n"
    "    - Use `map_set_title` for the title.\n"
    "    - Use `map_add_legend` to add a legend. **IMPORTANT RULE**: Leave the Action Input for `map_add_legend` completely empty to use the default settings, unless the user explicitly asks for a specific location (e.g., 'put the legend in the lower right'). Do not invent a location.\n"
    "4.  **Save & Finish:**\n"
    "    - Your final action MUST be `map_save`.\n"
    "    - Once you see 'Map saved to:', your task is complete. Your next and ONLY response MUST be the 'Final Answer' block. Do not call any more tools after saving.\n\n"
    
    "Begin!\n"
    "Previous conversation history: {chat_history}\n"
    "Question: {input}\n"
    "Thought:"
)


class AgentState(TypedDict):
    messages: List[Any]
    step: int
    max_steps: int


def _format_chat_history(history: Optional[List[Tuple[str, str]]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for role, content in history:
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_system_message(question: str, chat_history: Optional[List[Tuple[str, str]]]) -> SystemMessage:
    tool_names = ", ".join(get_tool_names())
    sys_prompt = SYS_PROMPT_TEMPLATE.format(
        tool_names=tool_names,
        tool_strings=get_tools_prompt_string(),
        chat_history=_format_chat_history(chat_history),
        input=question,
    )
    return SystemMessage(content=sys_prompt)


ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\n\s*Action:\s*(?P<action>\w+)\n\s*Action Input:\s*(?P<input>.*?)\n\s*Observation:\s*(?P<observation>.*)",
    re.DOTALL | re.IGNORECASE,
)
FINAL_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\n\s*Final Answer:\s*(?P<final>.*)", re.DOTALL | re.IGNORECASE
)

# Parse the model output and match three modes: 1.Action (with observation, but not recommended, easy to hallucinate) 2.Final 3.partial_action (without observation, to avoid model hallucination)
def parse_model_output(text: str) -> Dict[str, str]:
    match = ACTION_RE.search(text)
    if match:
        return {
            "type": "action",
            "thought": match.group("thought").strip(),
            "action": match.group("action").strip(),
            "action_input": match.group("input").strip(),
            "observation": match.group("observation").strip(),
        }
    match = FINAL_RE.search(text)
    if match:
        return {
            "type": "final",
            "thought": match.group("thought").strip(),
            "final": match.group("final").strip(),
        }
    # If the model only returns Thought/Action/Action Input without Observation, we will produce observation by executing tool
    # Modify (?P<input>.*) to (?P<input>[^\n]*) to avoid illusion output of llm. Parse llm output robustly！
    partial_action = re.search(
        r"Thought:\s*(?P<thought>.*?)\n\s*Action:\s*(?P<action>\w+)\n\s*Action Input:\s*(?P<input>[^\n]*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if partial_action:
        return {
            "type": "partial_action",
            "thought": partial_action.group("thought").strip(),
            "action": partial_action.group("action").strip(),
            "action_input": partial_action.group("input").strip(),
        }
    return {"type": "unknown", "raw": text}


def build_graph(llm_model_name: Optional[str] = None, max_steps: int = 15):
    graph = StateGraph(AgentState)

    def model_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        step = state["step"]
        # Initialize LLM lazily to avoid import/runtime errors when not configured
        llm = None
        if llm_model_name:
            model_lower = llm_model_name.lower()
            if "deepseek" in model_lower and ChatDeepSeek is not None:
                api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
                base_url = os.environ.get("DEEPSEEK_API_BASE")
                try:
                    if api_key or base_url:
                        # Pass explicit credentials/base URL if provided
                        llm = ChatDeepSeek(model=llm_model_name, temperature=0, api_key=api_key, base_url=base_url)  # type: ignore[arg-type]
                    else:
                        llm = ChatDeepSeek(model=llm_model_name, temperature=0)  # type: ignore
                except Exception:
                    llm = None
            elif ChatOpenAI is not None:
                try:
                    llm = ChatOpenAI(model=llm_model_name, temperature=0)
                except Exception:
                    llm = None
        if llm is None:
            raise RuntimeError(
                "LLM is not configured. Provide a model name and API key. For DeepSeek, set DEEPSEEK_API_KEY; for OpenAI, set OPENAI_API_KEY."
            )

        print("--- Calling Model ---")
        ai = llm.invoke(messages)  # type: ignore
        print("--- Model Response ---")
        print(ai.content) # 返回aimessage
        return {"messages": messages + [ai], "step": step + 1, "max_steps": state["max_steps"]}

    def tool_node(state: AgentState) -> AgentState:
        messages: List[Any] = state["messages"]
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        if last_ai is None:
            raise RuntimeError("No AIMessage to parse for Action")

        parsed = parse_model_output(last_ai.content)
        if parsed.get("type") not in {"action", "partial_action"}:
            # Nothing to do
            return state

        action_name = parsed.get("action", "")
        action_input = parsed.get("action_input", "")
        observation = call_tool(action_name, action_input)

        # Feed back the observation as a continuation in the required format
        # If map_save succeeded, nudge the model to finalize immediately
        if "Map saved to:" in observation:
            obs_block = (
                f"Observation: {observation}\n"
                "You have successfully saved the map. Now respond with a single Final Answer per the required format, summarizing the steps, listing all data paths used, and the output path. Do not call any more tools.\n"
                "Thought:"
            )
        else:
            obs_block = (
                f"Observation: {observation}\n"
                "Thought:"
            )
        messages = messages + [HumanMessage(content=obs_block)]
        return {"messages": messages, "step": state["step"], "max_steps": state["max_steps"]}

    def route_after_model(state: AgentState) -> str:
        if state["step"] >= state["max_steps"]:
            return END
        # Check the last AI message to decide next
        messages = state["messages"]
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        if last_ai is None:
            return END
        parsed = parse_model_output(last_ai.content)
        if parsed.get("type") in {"action", "partial_action"}:
            return "tools"
        if parsed.get("type") == "final":
            return END
        # If unknown, stop to avoid infinite loops
        return END

    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("model")
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    compiled = graph.compile()
    return compiled


def run_agent(
    question: str,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    llm_model_name: Optional[str] = None,
    max_steps: int = 30,
) -> tuple[str, list]: # for history
    system_message = _build_system_message(question, chat_history)
    state: AgentState = {
        "messages": [system_message, HumanMessage(content="")],
        "step": 0,
        "max_steps": max_steps,
    }

    graph = build_graph(llm_model_name=llm_model_name, max_steps=max_steps)
    config = {"recursion_limit": max_steps}
    final_state = graph.invoke(state, config=config)

    # Find Final Answer if present
    final_answer = None
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            parsed = parse_model_output(msg.content)
            if parsed.get("type") == "final":
                final_answer = parsed.get("final")
    if final_answer is None:
        # Fallback: if a map was saved, synthesize a concise Final Answer
        saved_path: Optional[str] = None
        for msg in final_state["messages"]:
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str) and "Observation:" in msg.content:
                m = re.search(r"Map saved to:\s*(.*)", msg.content)
                if m:
                    saved_path = m.group(1).strip()
        if saved_path:
            # Import lazily to avoid cycles at module load
            try:
                from .tools import _SESSION as _MAP_SESSION  # type: ignore
                data_paths = _MAP_SESSION.data_paths if getattr(_MAP_SESSION, "data_paths", None) else []
            except Exception:
                data_paths = []
            parts: List[str] = []
            parts.append("地图已生成并保存。")
            if data_paths:
                parts.append("使用的数据路径: " + ", ".join(data_paths))
            parts.append("输出路径: " + saved_path)
            final_answer = "\n".join(parts)
    if final_answer is None:
        final_answer = "No Final Answer produced within step limit."

    return final_answer, final_state["messages"]