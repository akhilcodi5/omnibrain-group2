"""LangGraph Supervisor Agent node for dynamic query routing, state management, and multi-agent coordination."""

import logging
import time
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.memo_synthesizer import MemoSynthesizer
from agents.search_agent import search_agent_node
from agents.sql_agent import sql_agent_node
from agents.state import AgentState
from agents.self_rag import self_rag_node
from agents.vision_agent import vision_node
from agents.visual_routing_evaluator import VisualIntentType, VisualRoutingEvaluator
from app.core.telemetry import get_telemetry_manager
from app.models.vision_schemas import VisualAnalyticalMemoBlock
from app.core.telemetry import get_telemetry_manager


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
    elif intent.requires_visual_agent and not has_vision_run:
        next_node = "vision_agent"
    elif any(k in query.lower() for k in ["price", "p/e", "market cap", "stock", "52-week", "ticker"]) and not has_sql_run:
        next_node = "sql_agent"
    elif not has_search_run:
        next_node = "search_agent"
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
    messages = state.get("messages", [])
    for m in messages:
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            query = getattr(m, "content", query)
            break

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

    # Extract company name from state or metadata
    company_name = state.get("pdf_name") or "Target Enterprise Entity"
    import os
    if company_name and company_name != "Target Enterprise Entity":
        company_name = os.path.splitext(os.path.basename(company_name))[0].strip()
    elif retrieved_docs and isinstance(retrieved_docs, list) and isinstance(retrieved_docs[0], dict):
        pdf_name = retrieved_docs[0].get("pdf_name", "")
        if pdf_name:
            company_name = os.path.splitext(os.path.basename(pdf_name))[0].strip()

    c_lower = company_name.lower()
    if "nvidia" in c_lower or "nvda" in c_lower:
        company_name = "NVIDIA Corporation (NVDA)"
    elif "apple" in c_lower or "aapl" in c_lower:
        company_name = "Apple Inc. (AAPL)"
    elif "microsoft" in c_lower or "msft" in c_lower:
        company_name = "Microsoft Corporation (MSFT)"

    # Synthesize Final Investment Memo
    start_time = time.time()
    final_memo = MemoSynthesizer.synthesize_investment_memo(
        company_name=company_name,
        analyst_query=query,
        visual_blocks=visual_blocks,
        text_context_snippets=text_snippets,
        sql_metrics=sql_dict,
    )
    elapsed = time.time() - start_time

    trace_id = state.get("trace_id")
    if trace_id:
        get_telemetry_manager().log_agent_step(
            trace_id=trace_id,
            agent_name="LangGraphSupervisor",
            action="SynthesizeMemo",
            model="gemini-1.5-pro",
            input_data=query,
            output_data=final_memo[:300],
            prompt_tokens=450,
            completion_tokens=len(final_memo.split()),
            latency_seconds=elapsed
        )

    ai_msg = AIMessage(
        content=final_memo,
        name="Synthesizer",
    )

    # Assess overall grounding from verified evidence blocks
    overall_grounded = state.get("is_grounded", True)
    if visual_blocks:
        has_critical = any(
            b.verification_report and b.verification_report.critical_discrepancies
            for b in visual_blocks
        )
        overall_grounded = not has_critical

    return {
        "messages": [ai_msg],
        "final_response": final_memo,
        "is_grounded": overall_grounded,
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


def create_supervisor_graph(use_checkpointer: bool = False) -> StateGraph:
    """Build and compile the complete LangGraph Multi-Agent StateGraph."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("self_rag", self_rag_node)
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
    workflow.add_edge("search_agent", "self_rag")
    
    def route_self_rag_edge(state: AgentState) -> Literal["search_agent", "supervisor"]:
        if state.get("next_agent") == "SearchAgent":
            return "search_agent"
        return "supervisor"
        
    workflow.add_conditional_edges(
        "self_rag",
        route_self_rag_edge,
        {
            "search_agent": "search_agent",
            "supervisor": "supervisor"
        }
    )
    
    workflow.add_edge("vision_agent", "supervisor")
    workflow.add_edge("sql_agent", "supervisor")
    
    # Synthesizer is the terminal node
    workflow.add_edge("synthesizer", END)

    # Add Checkpointer for Chat History Persistence
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
