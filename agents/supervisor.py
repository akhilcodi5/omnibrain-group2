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

    # Routing logic
    next_node: str = "synthesizer"

    # Safety guardrail against infinite looping
    if iteration >= 5:
        logger.warning(f"Supervisor reached max iteration count ({iteration}). Routing to synthesizer.")
        next_node = "synthesizer"
    # If query targets visual figures / charts and we haven't invoked vision agent yet
    elif intent.requires_visual_agent and not visual_evidence:
        next_node = "vision_agent"
    # If query is about stock prices, P/E, market cap or structured metrics and SQL not run
    elif any(k in query.lower() for k in ["price", "p/e", "market cap", "stock", "52-week", "ticker"]) and not sql_results:
        next_node = "sql_agent"
    # If general or verification query and search has not retrieved docs yet
    elif not retrieved_docs and intent.primary_intent in (VisualIntentType.GENERAL_QUERY, VisualIntentType.CROSS_MODAL_VERIFY, VisualIntentType.SYNTHESIZE_MEMO):
        next_node = "search_agent"
    # Otherwise, we have collected sufficient multi-modal evidence -> route to synthesizer
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
