"""LangGraph node adapter for the reusable Self-RAG pipeline."""
from typing import Any

from .retrieval import SelfRAGPipeline


def rag_node(state: dict[str, Any]):
    """Run the configured pipeline; callers inject it as ``state['rag_pipeline']``."""
    pipeline = state.get("rag_pipeline")
    if not isinstance(pipeline, SelfRAGPipeline):
        return {"final_answer": "Document retrieval is not configured.", "retrieved_context": []}
    result = pipeline.run(state.get("user_query", ""), state.get("metadata_filter"))
    return {"final_answer": result["answer"], "retrieved_context": result.get("context", []),
            "citations": result["citations"], "retrieval_trace": result["attempts"]}
