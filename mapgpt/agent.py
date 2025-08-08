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


SYS_PROMPT_TEMPLATE = (
    "You are a map expert and you are proficient in generating maps using vector or raster data. "
    "Your task is to answer the question or solve the problem step by step using the tools provided.\n"
    "You can only respond with a single complete \"Thought, Action, Action Input, Observation\" format OR a single \"Final Answer\" format.\n"
    "Complete format:\n"
    "Thought: (reflect on your progress and decide what to do next (based on observation if exist), do not skip)\n"
    "Action: (the action name, should be one of [{tool_names}]. decide the action based on previous Thought and Observation)\n"
    "Action Input: (the input string to the action, decide the input based on previous Thought and Observation)\n"
    "Observation: (the result of the action)\n"
    "(this process can repeat and you can only process one subtask at a time)\n"
    "OR\n"
    "Thought: (review original question and check my total process)\n"
    "Final Answer: (output the final answer to the original input question based on observations and lists all data paths used and generated)\n"
    "Answer the question below using the following tools:\n{tool_strings}\n\n"
    "Your final answer should contain all information necessary to answer the question and subquestions.\n"
    "IMPORTANT: Your first step is to learn and understand the following rules and examples, and plan your steps accordingly:\n"
    "The general process of making a map is: first initialize the map, add map layers, add other map components as needed, and finally generate the map.\n"
    "When making a map, the first step must be to initialize the map, and the last step must be to generate the map using map_save tool. These two steps are indispensable.\n"
    "Do not skip these steps.\n"
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
    partial_action = re.search(
        r"Thought:\s*(?P<thought>.*?)\n\s*Action:\s*(?P<action>\w+)\n\s*Action Input:\s*(?P<input>.*)",
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

        ai = llm.invoke(messages)  # type: ignore
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
    max_steps: int = 15,
) -> str:
    system_message = _build_system_message(question, chat_history)
    state: AgentState = {
        "messages": [system_message, HumanMessage(content="")],
        "step": 0,
        "max_steps": max_steps,
    }

    graph = build_graph(llm_model_name=llm_model_name, max_steps=max_steps)
    final_state = graph.invoke(state)

    # Find Final Answer if present
    final_answer = None
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            parsed = parse_model_output(msg.content)
            if parsed.get("type") == "final":
                final_answer = parsed.get("final")
    if final_answer is None:
        final_answer = "No Final Answer produced within step limit."
    return final_answer