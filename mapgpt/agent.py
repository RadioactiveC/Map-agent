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

# version - 3
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
    "Thought: I will do several things.\n"
    "Action: tool_1\n"
    "Action Input: input_1\n"
    "Action: tool_2\n"
    "Action Input: input_2\n\n"

    "When you have completed ALL steps and have successfully saved the map (confirmed by an Observation), your final response must be in this format:\n"
    "Thought: (I have completed the task and can now provide the final answer.)\n"
    "Final Answer: (A summary of what you did, the data paths used, and the final output path.)\n\n"

    "Here are the tools available:{tool_strings}\n\n"

    "CRITICAL REMINDERS:\n"
    "- Your very first action MUST be `map_initial`.\n"
    "- Your last action before the Final Answer MUST be `map_save`.\n"
    "- Once you see 'Map saved to:' in the Observation, your next and ONLY response MUST be the 'Final Answer' block.\n\n"

    "Begin!\n"
    "Previous conversation history: {chat_history}\n"
    "Question: {input}\n"
    "Thought:"
)

# version - 2 （not so good！cause several actions in one response）
# SYS_PROMPT_TEMPLATE = (
#     "You are a map expert and you are proficient in generating maps using vector or raster data. "
#     "Your task is to answer the question or solve the problem step by step using the tools provided.\n"
#     "You must respond with either a \"Thought, Action, Action Input\" block or a single \"Final Answer\" block.\n"
#     "When you use a tool, use the following format:\n"
#     "Thought: (reflect on your progress and decide what to do next. This is a mandatory step)\n"
#     "Action: (the name of the tool to use, should be one of [{tool_names}])\n"
#     "Action Input: (the input to the tool)\n\n"
#     "After you use a tool, the system will provide an Observation. You will use this Observation to plan your next step.\n\n"
#     "When you have completed the task and saved the map, respond with the final answer in this format:\n"
#     "Thought: (I have now completed all the steps and can provide the final answer.)\n"
#     "Final Answer: (A summary of what you did, the data paths used, and the final output path.)\n\n"
#     "Here are the tools available:{tool_strings}\n\n"
#     "IMPORTANT RULES:\n"
#     "1. The process is: Thought -> Action -> Action Input. The system provides the Observation.\n"
#     "2. The first step must be `map_initial` to create the map canvas.\n"
#     "3. The final step to generate the map must be `map_save`.\n"
#     "4. Once `map_save` is successful (you see 'Map saved to:' in the Observation), you MUST stop and provide the Final Answer.\n\n"
#     "Begin!\n"
#     "Previous conversation history: {chat_history}\n"
#     "Question: {input}\n"
#     "Thought:"
# )

# version - 1 （not ok！！！）
# SYS_PROMPT_TEMPLATE = (
#     "You are a map expert and you are proficient in generating maps using vector or raster data. "
#     "Your task is to answer the question or solve the problem step by step using the tools provided.\n"
#     "You can only respond with a single complete \"Thought, Action, Action Input, Observation\" format OR a single \"Final Answer\" format.\n"
#     "Complete format:\n"
#     "Thought: (reflect on your progress and decide what to do next (based on observation if exist), do not skip)\n"
#     "Action: (the action name, should be one of [{tool_names}]. decide the action based on previous Thought and Observation)\n"
#     "Action Input: (the input string to the action, decide the input based on previous Thought and Observation)\n"
#     "Observation: (the result of the action)\n"
#     "(this process can repeat and you can only process one subtask at a time)\n"
#     "OR\n"
#     "Thought: (review original question and check my total process)\n"
#     "Final Answer: (output the final answer to the original input question based on observations and lists all data paths used and generated)\n\n"
#     "Answer the question below using the following tools:{tool_strings}\n"
#     "Your final answer should contain all information necessary to answer the question and subquestions.\n\n"
#     "IMPORTANT: Your first step is to learn and understand the following rules and examples, and plan your steps accordingly:\n"
#     "The general process of making a map is: first initialize the map, add map layers, add other map components as needed, and finally generate the map.\n"
#     "When making a map, the first step must be to initialize the map, and the last step must be to generate the map using map_save tool. These two steps are indispensable.\n\n"
#     "Do not skip these steps.\n\n"
#     "CRITICAL: Once map_save succeeds (Observation contains 'Map saved to:'), immediately stop calling tools and reply with a single Final Answer block.\n"
#     "Your Final Answer must include: a short summary of what you did, list of all input data paths used, and the output image path(s).\n"
#     "Begin!\n"
#     "Previous conversation history: {chat_history}\n"
#     "Question: {input}\n"
#     "Thought:"
# )


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

# 解析模型输出，匹配三种模式：1.Action（有observation，但不建议，容易幻觉） 2.Final 3.partial_action(无observation，避免模型幻觉)
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
    return final_answer