"""Unit tests for SearchAgent, search_agent_node, and search_vector_store_tool."""

import pytest
from langchain_core.messages import HumanMessage
from agents.search_agent import SearchAgent, search_agent_node, search_vector_store_tool
from storage.vector_store import get_vector_store, DefaultEmbedder, VectorStore


@pytest.fixture
def populated_vector_store():
    """Fixture providing a VectorStore pre-populated with sample financial documents."""
    store = VectorStore(
        collection_name="test_search_agent_coll",
        in_memory=True,
        embedder=DefaultEmbedder(),
    )
    docs = [
        {
            "text": "Gross profit margin for Q3 2026 reached 68.4 percent due to higher software licensing sales.",
            "metadata": {"pdf_name": "q3_financial_report.pdf", "page_number": 4, "section_title": "Profitability"},
            "chunk_id": "chunk_search_01",
        },
        {
            "text": "Total operating expenses increased by 5 percent year-over-year to 12.8 million USD.",
            "metadata": {"pdf_name": "q3_financial_report.pdf", "page_number": 7, "section_title": "Expenses"},
            "chunk_id": "chunk_search_02",
        },
        {
            "text": "Capital expenditures focused primarily on expanding high-density GPU data centers.",
            "metadata": {"pdf_name": "annual_strategy.pdf", "page_number": 15, "section_title": "Infrastructure"},
            "chunk_id": "chunk_search_03",
        },
    ]
    store.add_documents(docs)
    yield store
    store.clear_collection()


def test_search_agent_execute_search(populated_vector_store):
    """Test SearchAgent direct search execution and citation payload generation."""
    agent = SearchAgent(vector_store=populated_vector_store)
    result = agent.execute_search("What was the gross profit margin in Q3?", top_k=2)

    assert "retrieved_docs" in result
    assert "citations" in result
    assert "retrieval_confidence" in result
    assert "summary" in result
    assert len(result["retrieved_docs"]) > 0

    top_doc = result["retrieved_docs"][0]
    assert top_doc["pdf_name"] == "q3_financial_report.pdf"
    assert top_doc["page_number"] == 4

    top_citation = result["citations"][0]
    assert top_citation["pdf_name"] == "q3_financial_report.pdf"
    assert top_citation["page_number"] == 4
    assert "Gross profit margin" in top_citation["snippet"]


def test_search_agent_pdf_filter(populated_vector_store):
    """Test SearchAgent with pdf_name_filter."""
    agent = SearchAgent(vector_store=populated_vector_store)
    result = agent.execute_search(
        query="data centers and infrastructure",
        top_k=5,
        pdf_name_filter="annual_strategy.pdf",
    )
    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["pdf_name"] == "annual_strategy.pdf"


def test_search_agent_node(populated_vector_store):
    """Test search_agent_node function updating LangGraph AgentState."""
    agent = SearchAgent(vector_store=populated_vector_store)
    
    state = {
        "query": "Tell me about Q3 operating expenses",
        "messages": [HumanMessage(content="Tell me about Q3 operating expenses")],
        "next_agent": None,
        "retrieved_docs": [],
        "visual_evidence": [],
        "referenced_images": [],
        "sql_query": None,
        "sql_results": None,
        "is_grounded": False,
        "retrieval_confidence": 0.0,
        "iteration_count": 0,
        "final_response": None,
        "citations": [],
    }

    # Execute node
    update = search_agent_node(state)

    assert "messages" in update
    assert len(update["messages"]) == 1
    assert update["messages"][0].name == "SearchAgent"
    assert "retrieved_docs" in update
    assert len(update["retrieved_docs"]) > 0
    assert "citations" in update
    assert len(update["citations"]) > 0
    assert update["is_grounded"] is True
