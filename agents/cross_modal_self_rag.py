"""Cross-Modal Self-RAG & Visual Fact-Checking Engine (Task 2B).

Performs autonomous visual-guided query rewriting and iterative retrieval loops to fact-check
and cross-reference extracted visual numbers against the vector database.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage

from agents.cross_modal_verifier import CrossModalVerifier
from agents.state import AgentState
from app.models.vision_schemas import (
    CrossModalVerificationReport,
    ExtractedChartData,
    MetricCrossReference,
    VerificationStatus,
)
from storage.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class CrossModalSelfRAG:
    """Iterative fact-checking engine that queries the vector database using visual metric anchors."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store(in_memory=True)
        self.verifier = CrossModalVerifier()

    def generate_targeted_search_queries(
        self,
        chart_data: ExtractedChartData,
        original_query: str,
    ) -> List[str]:
        """Formulate targeted search queries from visual metrics and labels to locate corroborating text."""
        queries = [original_query]
        title = chart_data.title or "Financial Metric"

        for series in chart_data.series:
            for pt in series.data_points:
                if pt.value is not None:
                    # Create focused query: e.g. "Operating Revenue Q3 $148.8M"
                    raw_str = pt.raw_value or f"{pt.value}"
                    q = f"{title} {series.series_name} {pt.label} {raw_str}"
                    queries.append(q)

        return list(dict.fromkeys(queries))[:5]  # Deduplicate, cap at 5 queries

    def execute_visual_fact_check_loop(
        self,
        chart_data: ExtractedChartData,
        original_query: str,
        initial_text_context: Optional[str] = None,
        max_search_iterations: int = 3,
        pdf_name: Optional[str] = None,
    ) -> CrossModalVerificationReport:
        """Run iterative Self-RAG loop:
        1. Attempt verification with initial text context.
        2. If unverified metrics remain, rewrite queries using visual numbers and retrieve targeted chunks.
        3. Re-verify until max iterations or all metrics verified.
        """
        accumulated_text = initial_text_context or ""
        
        # Initial verification pass
        report = self.verifier.verify_chart_against_text(chart_data, accumulated_text)
        
        # If all metrics matched or no vector store data available, return early
        if report.discrepancy_count == 0 and report.matched_metrics_count > 0:
            return report

        # Identify missing / unverified metrics
        unverified_refs = [
            cr for cr in report.cross_references 
            if cr.status in (VerificationStatus.VISUAL_MISSING_IN_TEXT, VerificationStatus.UNVERIFIABLE)
        ]

        if not unverified_refs:
            return report

        # Iterative visual query rewrite and retrieval
        targeted_queries = self.generate_targeted_search_queries(chart_data, original_query)
        logger.info(f"CrossModalSelfRAG executing {len(targeted_queries)} targeted vector searches (pdf_name: '{pdf_name}')...")

        filter_meta = {"pdf_name": pdf_name} if pdf_name else None
        new_chunks = []
        for q in targeted_queries[:max_search_iterations]:
            hits = self.vector_store.similarity_search(query=q, top_k=3, filter_metadata=filter_meta)
            for hit in hits:
                chunk_text = hit.get("text", "")
                if chunk_text and chunk_text not in accumulated_text:
                    new_chunks.append(chunk_text)
                    accumulated_text += "\n\n" + chunk_text

        # Re-run cross-modal verification over enriched accumulated text
        final_report = self.verifier.verify_chart_against_text(chart_data, accumulated_text)
        return final_report


def cross_modal_self_rag_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node handler for Cross-Modal Self-RAG fact-checking loop."""
    logger.info("Executing cross_modal_self_rag_node in LangGraph workflow...")

    query = state.get("query", "")
    pdf_name = state.get("pdf_name")
    visual_evidence = state.get("visual_evidence", [])
    retrieved_docs = state.get("retrieved_docs", [])

    vector_store = state.get("vector_store") or get_vector_store(in_memory=True)
    self_rag = CrossModalSelfRAG(vector_store=vector_store)

    text_context = " ".join([d.get("text", "") or d.get("content", "") for d in retrieved_docs if isinstance(d, dict)])
    
    updated_evidence = []
    total_grounding = 1.0

    for ev in visual_evidence:
        if isinstance(ev, dict) and "chart_data" in ev and ev["chart_data"]:
            try:
                chart_data = ExtractedChartData.model_validate(ev["chart_data"])
                report = self_rag.execute_visual_fact_check_loop(
                    chart_data=chart_data,
                    original_query=query,
                    initial_text_context=text_context,
                    pdf_name=pdf_name,
                )
                ev["verification_report"] = report.model_dump()
                total_grounding = report.grounding_score
            except Exception as e:
                logger.warning(f"Error in self-rag node for evidence: {e}")
        updated_evidence.append(ev)

    ai_message = AIMessage(
        content=f"Cross-Modal Self-RAG Fact-Checking complete. Overall grounding score: {int(total_grounding * 100)}%.",
        name="CrossModalSelfRAG",
    )

    return {
        "messages": [ai_message],
        "visual_evidence": updated_evidence,
        "is_grounded": total_grounding >= 0.75,
        "next_agent": "Supervisor",
    }
