"""
LangGraph agent for ValoTox toxicity detection.

Implements a stateful agent graph with tool-calling capabilities:
- Receives text input
- Calls toxicity classifier tool
- If passive-toxic detected, calls Valorant context checker
- Always calls severity scorer
- Returns structured moderation decision
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from valotox.config import settings

# ── Agent State ──────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """State carried through the LangGraph agent."""

    messages: Annotated[list[BaseMessage], operator.add]
    input_text: str
    toxicity_result: dict | None
    context_result: dict | None
    severity_result: dict | None
    final_decision: dict | None


# ── LangChain Tool wrappers ─────────────────────────────────────────────────


@tool
def classify_text(text: str) -> dict:
    """Classify a text comment for multi-label toxicity (toxic, harassment, gender_attack, slur, passive_toxic)."""
    from valotox.agent.tools import classify_toxicity

    result = classify_toxicity(text)
    return result.model_dump()


@tool
def check_context(text: str) -> dict:
    """Analyse Valorant gaming context to determine if passive-toxic slang is sarcastic vs. genuine."""
    from valotox.agent.tools import check_valorant_context

    result = check_valorant_context(text)
    return result.model_dump()


@tool
def score_severity_tool(text: str) -> dict:
    """Score the severity of a toxic comment: none → passive → moderate → severe → slur."""
    from valotox.agent.tools import score_severity

    return score_severity(text)


TOOLS = [classify_text, check_context, score_severity_tool]


# ── Graph nodes ──────────────────────────────────────────────────────────────

SYSTEM_MSG = SystemMessage(
    content="""You are ValoTox, an AI moderation assistant specialised in
Valorant gaming community toxicity detection. You have access to three tools:

1. classify_text — Run the fine-tuned toxicity classifier on a comment
2. check_context — Analyse Valorant-specific slang context (use when passive_toxic is detected)
3. score_severity_tool — Get a severity rating and moderation recommendation

For every input comment:
1. ALWAYS call classify_text first
2. If passive_toxic confidence > 0.3, ALSO call check_context
3. ALWAYS call score_severity_tool
4. Synthesise results into a clear moderation decision

Be concise. Focus on actionable moderation decisions."""
)


def create_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent."""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    ).bind_tools(TOOLS)

    tool_node = ToolNode(TOOLS)

    # ── Node: Agent reasoning ────────────────────────────────────────────
    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # ── Node: Check if we should continue to tools ───────────────────────
    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # ── Build graph ──────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Convenience function ─────────────────────────────────────────────────────

_compiled_graph = None


def get_agent():
    """Get or create the compiled agent graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph


def run_agent(text: str) -> dict:
    """Run the ValoTox agent on a single text input.

    Parameters
    ----------
    text : str
        The comment to analyse.

    Returns
    -------
    dict
        Full agent state including toxicity results and moderation decision.
    """
    agent = get_agent()

    initial_state: AgentState = {
        "messages": [
            SYSTEM_MSG,
            HumanMessage(content=f'Analyse this Valorant community comment for toxicity:\n\n"{text}"'),
        ],
        "input_text": text,
        "toxicity_result": None,
        "context_result": None,
        "severity_result": None,
        "final_decision": None,
    }

    result = agent.invoke(initial_state)

    # Extract the final AI message as the decision
    final_message = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_message = msg.content
            break

    return {
        "input_text": text,
        "agent_response": final_message,
        "full_state": result,
    }
