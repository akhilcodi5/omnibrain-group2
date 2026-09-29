"""Unit and integration tests for FastAPI Chat & Agent Orchestrator endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_chat_memo_endpoint():
    """Test POST /api/v1/chat/memo endpoint."""
    client = TestClient(app)
    payload = {
        "company_name": "Apex Holdings",
        "research_query": "Analyze quarterly EBITDA margins and revenue",
        "visual_blocks": [],
        "text_snippets": ["Apex demonstrated strong operating leverage in FY24."],
        "sql_metrics": {"Ticker": "APEX", "Market Cap": "$42.5B"},
    }

    res = client.post("/api/v1/chat/memo", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["company_name"] == "Apex Holdings"
    assert "Investment Research Memorandum" in data["investment_memo_markdown"]
    assert "Apex demonstrated strong operating leverage" in data["investment_memo_markdown"]


def test_chat_query_endpoint():
    """Test POST /api/v1/chat/query multi-agent routing execution."""
    client = TestClient(app)
    payload = {
        "query": "What is the stock price of APEX?",
        "referenced_images": [],
    }

    res = client.post("/api/v1/chat/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "final_response" in data
    assert data["execution_time_seconds"] >= 0.0


def test_health_endpoints():
    """Test /health, /healthz, and /api/v1/health endpoints."""
    client = TestClient(app)
    for path in ["/health", "/healthz", "/api/v1/health"]:
        res = client.get(path)
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

