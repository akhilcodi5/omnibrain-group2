"""Unit tests for SelfRAGEngine, relevance evaluation, query rewriting, and self_rag_node."""

import pytest
from langchain_core.messages import HumanMessage
from agents.self_rag import SelfRAGEngine, self_rag_node


def test_self_rag_grade_retrieval_relevance():
    """Test grading relevance of retrieved document chunks against a user query."""
    engine = SelfRAGEngine()

    # Case 1: Relevant documents
    query = "What is the revenue growth for Q3 2026?"
    relevant_docs = [
        {
            "text": "Q3 2026 revenue growth reached 18.5% year-over-year.",
            "score": 0.85,
            "pdf_name": "q3_report.pdf",
            "page_number": 2,
        }
    ]
    res1 = engine.grade_retrieval_relevance(query, relevant_docs)
    assert res1["is_relevant"] is True
    assert res1["relevance_score"] >= 0.35

    # Case 2: Irrelevant documents
    irrelevant_docs = [
        {
            "text": "Employee office parking policy and cafeteria guidelines.",
            "score": 0.12,
            "pdf_name": "policy.pdf",
            "page_number": 1,
        }
    ]
    res2 = engine.grade_retrieval_relevance(query, irrelevant_docs, confidence_threshold=0.5)
    assert res2["is_relevant"] is False


def test_self_rag_query_rewrite():
    """Test automatic query rewriting across iterations."""
    engine = SelfRAGEngine()
    original_query = "tell me about what is the revenue"

    rewritten_1 = engine.rewrite_query(original_query, iteration_count=1)
    assert isinstance(rewritten_1, str)
    assert len(rewritten_1) > 0
    assert "revenue" in rewritten_1

    rewritten_2 = engine.rewrite_query(original_query, iteration_count=2)
    assert isinstance(rewritten_2, str)
    assert "revenue" in rewritten_2


def test_self_rag_check_grounding():
    """Test grounding and hallucination check logic."""
    engine = SelfRAGEngine()
    docs = [
        {
            "text": "OmniBrain uses Qdrant vector database and CLIP embeddings for multi-modal document search.",
            "score": 0.9,
        }
    ]

    # Case 1: Grounded answer
    answer_grounded = "OmniBrain utilizes Qdrant vector storage and CLIP embeddings for document search."
    res_g = engine.check_grounding(answer_grounded, docs)
    assert res_g["is_grounded"] is True
    assert res_g["confidence"] > 0.0

    # Case 2: Ungrounded / empty answer
    res_empty = engine.check_grounding("", docs)
    assert res_empty["is_grounded"] is False


def test_self_rag_node_irrelevant_loop():
    """Test self_rag_node re-triggering SearchAgent when retrieval is irrelevant."""
    state = {
        "query": "What is the capital expenditure budget?",
        "messages": [HumanMessage(content="What is the capital expenditure budget?")],
        "next_agent": None,
        "retrieved_docs": [],  # Empty retrieval
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

    update = self_rag_node(state)

    assert update["next_agent"] == "SearchAgent"
    assert update["iteration_count"] == 1
    assert update["is_grounded"] is False
    assert len(update["messages"]) == 1
    assert update["messages"][0].name == "SelfRAG"
    assert "Self-RAG Reflection" in update["messages"][0].content


def test_self_rag_node_relevant_pass():
    """Test self_rag_node approving relevant retrieval and proceeding without loop."""
    state = {
        "query": "What is the capital expenditure budget?",
        "messages": [HumanMessage(content="What is the capital expenditure budget?")],
        "next_agent": None,
        "retrieved_docs": [
            {
                "text": "Capital expenditure budget for fiscal year 2026 is set at 15 million USD.",
                "score": 0.92,
                "pdf_name": "budget.pdf",
                "page_number": 3,
            }
        ],
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

    update = self_rag_node(state)

    assert update["next_agent"] is None
    assert update["is_grounded"] is True
    assert update["iteration_count"] == 0
