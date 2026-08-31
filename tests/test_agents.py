"""Reasoning Audit Tests for LangGraph Supervisor Agent and Sub-Agents (Week 2 Milestone)."""

import pytest
from agents.sql_agent import SQLAgent
from agents.supervisor import (
    create_supervisor_graph,
    route_supervisor_edge,
    supervisor_node,
)
from storage.sql_db import FinancialDatabase


def test_sql_agent_query_generation_and_execution():
    """Test SQLAgent Text-to-SQL generation and database execution."""
    db = FinancialDatabase(db_path=":memory:")
    agent = SQLAgent(db=db)

    # 1. Stock price query
    res = agent.execute("What is the current stock price and PE ratio of APEX?")
    assert "SELECT" in res["sql_query"]
    assert len(res["sql_results"]) == 1
    assert res["sql_results"][0]["ticker"] == "APEX"
    assert res["sql_results"][0]["current_price"] == 184.50

    # 2. Quarterly financials query
    res_q = agent.execute("Show me quarterly revenue and margins for APEX")
    assert len(res_q["sql_results"]) == 4
    assert res_q["sql_results"][0]["revenue_millions"] == 120.5


def test_supervisor_reasoning_audit_routing_decisions():
    """Reasoning Audit: Prove Supervisor correctly routes between Vision, SQL, and Search."""
    # 1. Visual figure query -> routes to vision_agent
    state_visual = {
        "messages": [],
        "query": "Analyze the operating margin bar chart on page 14.",
        "visual_evidence": [],
        "retrieved_docs": [],
        "sql_results": [],
        "iteration_count": 0,
    }
    decision_v = supervisor_node(state_visual)
    assert decision_v["next_agent"] == "vision_agent"
    assert route_supervisor_edge(decision_v) == "vision_agent"

    # 2. Structured stock query -> routes to sql_agent
    state_sql = {
        "messages": [],
        "query": "What is the 52-week high stock price and market cap for APEX?",
        "visual_evidence": [],
        "retrieved_docs": [],
        "sql_results": [],
        "iteration_count": 0,
    }
    decision_s = supervisor_node(state_sql)
    assert decision_s["next_agent"] == "sql_agent"
    assert route_supervisor_edge(decision_s) == "sql_agent"

    # 3. General semantic query -> routes to search_agent
    state_search = {
        "messages": [],
        "query": "Summarize the qualitative risk disclosures in the annual report.",
        "visual_evidence": [],
        "retrieved_docs": [],
        "sql_results": [],
        "iteration_count": 0,
    }
    decision_sr = supervisor_node(state_search)
    assert decision_sr["next_agent"] == "search_agent"
    assert route_supervisor_edge(decision_sr) == "search_agent"


def test_supervisor_graph_compilation():
    """Test full LangGraph StateGraph instantiation and compilation."""
    app = create_supervisor_graph()
    assert app is not None
