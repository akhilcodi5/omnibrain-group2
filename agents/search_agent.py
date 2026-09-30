"""Semantic vector search agent for unstructured text chunk retrieval and grounding."""

import logging
from typing import Any, Dict, List, Optional
import time
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agents.state import AgentState
from storage.vector_store import VectorStore, get_vector_store
from app.core.telemetry import get_telemetry_manager

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, top_k: int = 5, pdf_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search the vector database for semantically relevant document chunks.
    
    Args:
        query: The semantic search query string.
        top_k: Number of top relevant document chunks to return.
        pdf_name: Optional filter to restrict search to a specific PDF document.

    Returns:
        List of matching document chunk dictionaries with text, page_number, pdf_name, and relevance score.
    """
    vector_store = get_vector_store(in_memory=False)
    filter_metadata = {"pdf_name": pdf_name} if pdf_name else None
    
    results = vector_store.similarity_search(
        query=query,
        top_k=top_k,
        filter_metadata=filter_metadata,
    )
    return results


class SearchAgent:
    """Specialist agent responsible for retrieving semantic text chunks from Qdrant vector storage."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store(in_memory=True)

    def execute_search(
        self,
        query: str,
        top_k: int = 5,
        pdf_name_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Execute vector similarity search and construct structured context & citation payloads.
        
        Args:
            query: The user query string
            top_k: Max chunks to retrieve
            pdf_name_filter: Filter results by target PDF filename
            score_threshold: Minimum similarity threshold

        Returns:
            Dictionary containing 'retrieved_docs', 'citations', 'retrieval_confidence', and 'summary'
        """
        filter_metadata = {"pdf_name": pdf_name_filter} if pdf_name_filter else None
        
        hits = self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )

        retrieved_docs = []
        citations = []
        total_score = 0.0

        for hit in hits:
            chunk_id = hit.get("chunk_id", "")
            text = hit.get("text", "")
            score = hit.get("score", 0.0)
            pdf_name = hit.get("pdf_name", "unknown.pdf")
            page_number = hit.get("page_number", 1)
            section_title = hit.get("section_title", "")
            
            total_score += score
            
            retrieved_docs.append({
                "chunk_id": chunk_id,
                "text": text,
                "score": score,
                "pdf_name": pdf_name,
                "page_number": page_number,
                "section_title": section_title,
                "metadata": hit.get("metadata", {}),
            })

            citations.append({
                "citation_id": f"cit_{chunk_id[:8]}",
                "pdf_name": pdf_name,
                "page_number": page_number,
                "section_title": section_title,
                "text": text,
                "snippet": text[:150] + "..." if len(text) > 150 else text,
                "relevance_score": score,
            })

        avg_confidence = (total_score / len(hits)) if hits else 0.0

        summary_lines = [f"Retrieved {len(hits)} relevant document chunks (avg confidence: {avg_confidence:.2f}):"]
        for idx, doc in enumerate(retrieved_docs, 1):
            summary_lines.append(
                f"[{idx}] Page {doc['page_number']} ({doc['pdf_name']}): {doc['text'][:120]}..."
            )

        return {
            "retrieved_docs": retrieved_docs,
            "citations": citations,
            "retrieval_confidence": avg_confidence,
            "summary": "\n".join(summary_lines),
        }


def search_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for executing vector search and updating shared AgentState."""
    query = state.get("query", "")
    if not query and state.get("messages"):
        last_msg = state["messages"][-1]
        query = getattr(last_msg, "content", str(last_msg))

    vector_store = state.get("vector_store") or get_vector_store(in_memory=False)
    trace_id = state.get("trace_id")
    logger.info(f"Executing search_agent_node for query: '{query}'")
    
    start_time = time.time()
    agent = SearchAgent(vector_store=vector_store)
    pdf_name_filter = state.get("pdf_name")
    result = agent.execute_search(query=query, top_k=5, pdf_name_filter=pdf_name_filter)
    elapsed = time.time() - start_time

    summary_text = result["summary"]
    
    if trace_id:
        get_telemetry_manager().log_agent_step(
            trace_id=trace_id,
            agent_name="SearchAgent",
            action="DenseRetrieval",
            model="embedding-local",
            input_data=query,
            output_data=summary_text,
            prompt_tokens=len(query.split()), 
            completion_tokens=len(summary_text.split()),
            latency_seconds=elapsed
        )

    ai_message = AIMessage(
        content=f"Search Agent Retrieval Results:\n\n{summary_text}",
        name="SearchAgent",
    )

    return {
        "messages": [ai_message],
        "retrieved_docs": result["retrieved_docs"],
        "citations": result["citations"],
        "retrieval_confidence": result["retrieval_confidence"],
        "is_grounded": len(result["retrieved_docs"]) > 0,
    }
