"""LangGraph Supervisor Agent node for dynamic query routing, state management, and multi-agent coordination."""

import logging
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph

from agents.memo_synthesizer import MemoSynthesizer
from agents.search_agent import search_agent_node
from agents.sql_agent import sql_agent_node
from agents.state import AgentState
from agents.vision_agent import vision_node
from agents.visual_routing_evaluator import VisualIntentType, VisualRoutingEvaluator
from app.models.vision_schemas import VisualAnalyticalMemoBlock

logger = logging.getLogger(__name__)


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Supervisor node evaluating state progress and determining the next sub-agent to invoke."""
    query = state.get("query", "")
    iteration = state.get("iteration_count", 0) + 1
    
    retrieved_docs = state.get("retrieved_docs", [])
    visual_evidence = state.get("visual_evidence", [])
    sql_results = state.get("sql_results", [])

    # Evaluate query routing intent
    intent = VisualRoutingEvaluator.evaluate_query(query)
    logger.info(f"Supervisor routing iteration {iteration} for intent: {intent.primary_intent.value}")

    # Track which agents have already executed to prevent loops
    messages = state.get("messages", [])
    has_sql_run = (state.get("sql_query") is not None) or bool(sql_results) or any(getattr(m, "name", "") == "SQLAgent" for m in messages)
    has_search_run = bool(retrieved_docs) or any(getattr(m, "name", "") == "SearchAgent" for m in messages)
    has_vision_run = bool(visual_evidence) or any(getattr(m, "name", "") in ("VisionAgent", "VisualAnalyticsIntegratorAgent") for m in messages)

    # Routing logic with hard iteration ceiling
    next_node: str = "synthesizer"

    if iteration >= 4:
        next_node = "synthesizer"
    elif not has_search_run:
        next_node = "search_agent"
    elif intent.requires_visual_agent and not has_vision_run:
        next_node = "vision_agent"
    elif any(k in query.lower() for k in ["price", "p/e", "market cap", "stock", "52-week", "ticker"]) and not has_sql_run:
        next_node = "sql_agent"
    else:
        next_node = "synthesizer"

    ai_msg = AIMessage(
        content=f"Supervisor Decision: Routing execution to `{next_node}` (Iteration {iteration}).",
        name="Supervisor",
    )

    return {
        "messages": [ai_msg],
        "next_agent": next_node,
        "iteration_count": iteration,
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Final synthesizer node that compiles multi-modal findings into an investment research memo."""
    logger.info("Executing synthesizer_node in LangGraph...")
    
    query = state.get("query", "")
    raw_visual_evidence = state.get("visual_evidence", [])
    retrieved_docs = state.get("retrieved_docs", [])
    sql_results = state.get("sql_results", [])

    # Convert evidence dictionaries to memo blocks
    visual_blocks: List[VisualAnalyticalMemoBlock] = []
    for ev in raw_visual_evidence:
        if isinstance(ev, dict) and "markdown_formatted_block" in ev:
            try:
                visual_blocks.append(VisualAnalyticalMemoBlock.model_validate(ev))
            except Exception:
                pass

    text_snippets = [d.get("text", "") for d in retrieved_docs if isinstance(d, dict)]
    sql_dict = sql_results[0] if sql_results and isinstance(sql_results, list) else None

    # Synthesize Final Investment Memo
    final_memo = MemoSynthesizer.synthesize_investment_memo(
        company_name="Target Enterprise Entity",
        analyst_query=query,
        visual_blocks=visual_blocks,
        text_context_snippets=text_snippets,
        sql_metrics=sql_dict,
    )

    ai_msg = AIMessage(
        content=final_memo,
        name="Synthesizer",
    )

    return {
        "messages": [ai_msg],
        "final_response": final_memo,
        "next_agent": "END",
    }


def route_supervisor_edge(state: AgentState) -> Literal["search_agent", "vision_agent", "sql_agent", "synthesizer", "__end__"]:
    """Conditional edge function mapping Supervisor state to the next agent node."""
    next_dest = state.get("next_agent", "synthesizer")
    
    if next_dest == "search_agent":
        return "search_agent"
    elif next_dest == "vision_agent":
        return "vision_agent"
    elif next_dest == "sql_agent":
        return "sql_agent"
    elif next_dest == "synthesizer":
        return "synthesizer"
    return END


def create_supervisor_graph() -> StateGraph:
    """Build and compile the complete LangGraph Multi-Agent StateGraph."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("vision_agent", vision_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Add Conditional Edges from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor_edge,
        {
            "search_agent": "search_agent",
            "vision_agent": "vision_agent",
            "sql_agent": "sql_agent",
            "synthesizer": "synthesizer",
            END: END,
        }
    )

    # Sub-agents loop back to supervisor to update state and report findings
    workflow.add_edge("search_agent", "supervisor")
    workflow.add_edge("vision_agent", "supervisor")
    workflow.add_edge("sql_agent", "supervisor")
    
    # Synthesizer is the terminal node
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
