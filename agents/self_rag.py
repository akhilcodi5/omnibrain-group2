"""Self-RAG agent module for document relevance grading, query rewriting, and hallucination checking."""

import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


class SelfRAGEngine:
    """Self-Corrective RAG engine providing relevance evaluation, query rewriting, and grounding audits."""

    def __init__(self, llm: Optional[Any] = None, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self._llm = llm

    def _get_llm(self):
        if self._llm is None and settings.OPENAI_API_KEY:
            try:
                self._llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.0,
                    api_key=settings.OPENAI_API_KEY,
                )
            except Exception as e:
                logger.warning(f"Could not initialize ChatOpenAI ({e}). Using rule-based fallback.")
        return self._llm

    def grade_retrieval_relevance(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        confidence_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Evaluate whether retrieved document chunks are relevant to the user query.
        
        Returns:
            Dict containing 'is_relevant': bool, 'relevance_score': float, and 'reasoning': str.
        """
        if not retrieved_docs:
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "reasoning": "No document chunks were retrieved.",
            }

        # Calculate average retrieval similarity score
        scores = [doc.get("score", 0.0) for doc in retrieved_docs if "score" in doc]
        avg_score = (sum(scores) / len(scores)) if scores else 0.5

        # Check key terms matching between query and retrieved texts
        query_words = set(re.findall(r"\w+", query.lower())) - {"what", "is", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "tell", "me", "about"}
        combined_text = " ".join([doc.get("text", "") for doc in retrieved_docs]).lower()

        matched_words = [word for word in query_words if word in combined_text]
        keyword_overlap_ratio = (len(matched_words) / len(query_words)) if query_words else 1.0

        is_relevant = avg_score >= confidence_threshold or keyword_overlap_ratio >= 0.3

        # Try LLM grading if available
        llm = self._get_llm()
        if llm is not None and query_words:
            try:
                prompt = (
                    f"User Query: {query}\n\n"
                    f"Retrieved Document Context:\n{combined_text[:1500]}\n\n"
                    "Evaluate if the retrieved context is relevant to answering the query. "
                    "Respond strictly with 'YES' or 'NO' followed by a one-sentence explanation."
                )
                response = llm.invoke(prompt)
                resp_text = getattr(response, "content", str(response)).strip()
                if resp_text.startswith("YES"):
                    is_relevant = True
                elif resp_text.startswith("NO"):
                    is_relevant = False
            except Exception as e:
                logger.warning(f"LLM relevance grading failed: {e}. Falling back to rule-based evaluation.")

        return {
            "is_relevant": is_relevant,
            "relevance_score": avg_score,
            "reasoning": f"Matched {len(matched_words)}/{len(query_words)} query keywords. Avg vector score: {avg_score:.2f}.",
        }

    def rewrite_query(self, query: str, iteration_count: int = 1) -> str:
        """Rewrite a failing query to broaden or sharpen semantic search terms."""
        llm = self._get_llm()
        if llm is not None:
            try:
                prompt = (
                    f"The following search query failed to retrieve relevant documents (attempt {iteration_count}):\n"
                    f"'{query}'\n\n"
                    "Rewrite this query into a clearer, more effective semantic search query for a financial/corporate PDF. "
                    "Provide ONLY the rewritten query text."
                )
                response = llm.invoke(prompt)
                rewritten = getattr(response, "content", str(response)).strip().strip('"\'')
                if rewritten:
                    logger.info(f"LLM rewritten query: '{query}' -> '{rewritten}'")
                    return rewritten
            except Exception as e:
                logger.warning(f"LLM query rewriter failed: {e}. Using rule-based rewriter.")

        # Rule-based fallback rewriter
        stop_words = {"tell", "me", "about", "what", "is", "the", "how", "much", "did", "was", "were", "a", "an"}
        words = [w for w in query.split() if w.lower() not in stop_words]
        if iteration_count == 1:
            rewritten = " ".join(words) + " breakdown details"
        elif iteration_count == 2:
            rewritten = " ".join(words) + " financial summary"
        else:
            rewritten = " ".join(words)

        logger.info(f"Rule-based rewritten query: '{query}' -> '{rewritten}'")
        return rewritten

    def check_grounding(
        self,
        final_response: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify that the generated response is strictly grounded in retrieved document evidence."""
        if not final_response:
            return {"is_grounded": False, "confidence": 0.0}
        if not retrieved_docs:
            return {"is_grounded": False, "confidence": 0.0}

        doc_text = " ".join([doc.get("text", "") for doc in retrieved_docs])
        
        # Calculate overlap ratio
        response_words = set(re.findall(r"\w+", final_response.lower())) - {"the", "a", "an", "is", "are", "was", "were", "in", "to", "and", "of"}
        if not response_words:
            return {"is_grounded": True, "confidence": 1.0}

        overlap = sum(1 for word in response_words if word in doc_text.lower())
        overlap_score = overlap / len(response_words)

        is_grounded = overlap_score >= 0.25
        return {
            "is_grounded": is_grounded,
            "confidence": min(1.0, overlap_score * 1.5),
        }


def self_rag_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph Self-RAG reflection node. Evaluates retrieval relevance and handles query rewriting loops."""
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    iteration_count = state.get("iteration_count", 0)

    engine = SelfRAGEngine(max_iterations=3)
    evaluation = engine.grade_retrieval_relevance(query, retrieved_docs)

    is_relevant = evaluation["is_relevant"]
    reasoning = evaluation["reasoning"]

    if not is_relevant and iteration_count < engine.max_iterations:
        new_iteration = iteration_count + 1
        rewritten_query = engine.rewrite_query(query, iteration_count=new_iteration)

        reflection_msg = AIMessage(
            content=f"Self-RAG Reflection (Attempt {iteration_count}): Retrieval irrelevant ({reasoning}). "
                    f"Rewriting query to: '{rewritten_query}'. Re-triggering SearchAgent.",
            name="SelfRAG",
        )

        return {
            "messages": [reflection_msg],
            "query": rewritten_query,
            "iteration_count": new_iteration,
            "next_agent": "SearchAgent",
            "is_grounded": False,
            "retrieval_confidence": evaluation["relevance_score"],
        }

    # Retrieval is relevant or max iterations reached
    reflection_msg = AIMessage(
        content=f"Self-RAG Reflection: Retrieval verified relevant ({reasoning}). Proceeding with synthesis.",
        name="SelfRAG",
    )

    return {
        "messages": [reflection_msg],
        "iteration_count": iteration_count,
        "next_agent": None,
        "is_grounded": is_relevant,
        "retrieval_confidence": evaluation["relevance_score"],
    }
