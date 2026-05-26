from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_deepseek import ChatDeepSeek  # type: ignore
except Exception:  # pragma: no cover
    ChatDeepSeek = None  # type: ignore

from .tools import get_tool_names, get_tools_prompt_string, call_tool, get_tools_list, get_tools_by_name


SYS_PROMPT_TEMPLATE = (
    "You are an expert GIS Map Agent specialized in creating visualization maps using Python and Matplotlib. "
    "Your goal is to satisfy the user's request by calling the appropriate tools in a logical, sequential order.\n\n"

    "--- CORE WORKFLOW & RULES ---\n"
    "1. **Sequential Execution**: You plan your actions step-by-step. While you can call multiple tools in one turn if needed, logical dependency MUST be respected (e.g., you cannot save a map before creating it).\n"
    "2. **State Dependency (CRITICAL)**: The mapping environment is state-based. \n"
    "   - **Styling**: To set the color/width/size of a layer, you MUST call the `modify_*` tools (e.g., `modify_line_color`) **BEFORE** calling `map_add_layer`.\n"
    "   - Once a layer is added, its style is 'baked in' and cannot be changed without restarting.\n"
    "3. **Modification Requires Restart**: If the user asks to modify an existing map (e.g., 'change the blue line to red'), you MUST clear the state and rebuild the map from scratch. Start with `map_initial` and re-apply all steps with the requested changes.\n\n"

    "--- STANDARD MAP-MAKING RECIPE ---\n"
    "Follow this order unless instructed otherwise:\n"
    "1. **Initialize**: Call `map_initial`. (Optional: set background color immediately after).\n"
    "2. **Layer Loop** (Repeat for each data layer):\n"
    "   a. Set Styles: Call `modify_line_color`, `modify_line_width`, etc.\n"
    "   b. Add Layer: Call `map_add_layer` with the file path.\n"
    "3. **Decoration**: Call `map_set_title` and `map_add_legend`.\n"
    "4. **Finalize**: Call `map_save`. Once the tool returns 'Map saved to...', output your Final Answer.\n\n"

    "--- TOOL ARGUMENT GUIDELINES ---\n"
    "- **Data Paths**: Use the exact file paths provided by the user. Do not invent paths.\n"
    "- **Complex Arguments**: \n"
    "  - `map_add_layer`: To add a label for the legend, pass a JSON string as the argument. Example: '{{\"path\": \"/data/road.shp\", \"label\": \"Main Roads\"}}'\n"
    "  - `map_add_legend`: To customize location, pass a JSON string. Example: '{{\"loc\": \"upper left\"}}'. If the user does not specify a location, leave the input empty to use the default position (lower right).\n\n"

    "Begin!\n"
    "Previous conversation history: {chat_history}\n"
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


def _build_system_message(chat_history: Optional[List[Tuple[str, str]]]) -> SystemMessage:
    sys_prompt = SYS_PROMPT_TEMPLATE.format(
        chat_history=_format_chat_history(chat_history),
    )
    return SystemMessage(content=sys_prompt)


def build_graph(llm_model_name: Optional[str] = None, max_steps: int = 15):
    graph = StateGraph(AgentState)

    def model_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        step = state["step"]
        llm = None
        if llm_model_name:
            model_lower = llm_model_name.lower()
            if "deepseek" in model_lower and ChatDeepSeek is not None:
                api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
                base_url = os.environ.get("DEEPSEEK_API_BASE")
                try:
                    if api_key or base_url:
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
        tools = get_tools_list()
        model_with_tools = llm.bind_tools(tools)
        ai = model_with_tools.invoke(messages)
        print("--- Model Response ---")
        print(ai.content)
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

        result = []
        tools_by_name = get_tools_by_name()
        for tool_call in last_ai.tool_calls:
            tool_name = tool_call["name"]
            tool = tools_by_name.get(tool_name)

            if tool is None:
                raw_output = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    raw_output = tool.invoke(tool_call["args"])
                except Exception as e:
                    raw_output = f"Error executing {tool_name}: {str(e)}"

            if tool_call["name"] == "map_save" and "Map saved to" in str(raw_output):
                raw_output += "\n(SYSTEM NOTE: Map generation complete. Please respond with a Final Answer summarizing the work.)"

            result.append(ToolMessage(content=raw_output, tool_call_id=tool_call["id"]))

        messages = messages + result
        return {"messages": messages, "step": state["step"], "max_steps": state["max_steps"]}

    def route_after_model(state: AgentState) -> str:
        if state["step"] >= state["max_steps"]:
            return END
        messages = state["messages"]
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        if last_ai is None:
            return END

        if last_ai.tool_calls:
            return "tools"

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
) -> tuple[str, list]:
    system_message = _build_system_message(chat_history)
    state: AgentState = {
        "messages": [system_message, HumanMessage(content=question)],
        "step": 0,
        "max_steps": max_steps,
    }

    graph = build_graph(llm_model_name=llm_model_name, max_steps=max_steps)
    config = {"recursion_limit": max_steps}
    final_state = graph.invoke(state, config=config)

    final_answer = None

    if final_state["messages"]:
        last_msg = final_state["messages"][-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            final_answer = last_msg.content

    if final_answer is None:
        saved_path: Optional[str] = None
        for msg in final_state["messages"]:
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                m = re.search(r"Map saved to:\s*(.*)", msg.content)
                if m:
                    saved_path = m.group(1).strip()
        if saved_path:
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
