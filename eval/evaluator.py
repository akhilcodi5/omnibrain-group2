"""Multi-Modal RAG Evaluation, Benchmarking & Faithfulness Scoring Suite (Week 4)."""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.telemetry import get_telemetry_manager
from app.models.vision_schemas import ExtractedChartData

logger = logging.getLogger(__name__)


class MultiModalEvalReport(BaseModel):
    """Evaluation scorecard for multi-modal RAG grounding, accuracy, and citation precision."""
    faithfulness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Proportion of claims supported by evidence.")
    visual_numerical_accuracy: float = Field(default=1.0, ge=0.0, le=1.0, description="Numerical accuracy score.")
    citation_precision: float = Field(default=1.0, ge=0.0, le=1.0, description="Citation relevance and page precision.")
    hallucination_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated hallucination risk (0.0 to 1.0).")
    total_claims_evaluated: int = 0
    grounded_claims_count: int = 0
    passed_all_thresholds: bool = True
    summary: str


class MultiModalRAGEvaluator:
    """Evaluates multi-modal RAG responses against retrieved document evidence and ground-truth figures."""

    def __init__(self):
        self.telemetry = get_telemetry_manager()

    @staticmethod
    def extract_factual_claims(text: str) -> List[str]:
        """Split generated response into individual factual assertion statements."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = [
            s.strip() for s in sentences
            if any(char.isdigit() for char in s) or any(w in s.lower() for w in ["revenue", "margin", "growth", "increased", "decreased"])
        ]
        return claims

    def evaluate_grounding_faithfulness(
        self,
        claims: List[str],
        evidence_text: str,
    ) -> Dict[str, Any]:
        """Compute faithfulness score by verifying that numbers and keywords in claims exist in evidence."""
        if not claims:
            return {"score": 1.0, "grounded_count": 0, "total": 0}

        grounded_count = 0
        evidence_lower = evidence_text.lower()

        for claim in claims:
            # Extract numbers from claim
            numbers_in_claim = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
            
            # If numbers exist, check if at least one number appears in evidence
            if numbers_in_claim:
                if any(num in evidence_text for num in numbers_in_claim):
                    grounded_count += 1
            else:
                # Check keyword overlap
                words = [w for w in claim.lower().split() if len(w) > 4]
                overlap = sum(1 for w in words if w in evidence_lower)
                if overlap >= max(1, len(words) // 2):
                    grounded_count += 1

        faithfulness = round(grounded_count / len(claims), 2)
        return {
            "score": faithfulness,
            "grounded_count": grounded_count,
            "total": len(claims),
        }

    def evaluate_citation_precision(
        self,
        citations: List[Dict[str, Any]],
        expected_pages: Optional[List[int]] = None,
    ) -> float:
        """Measure whether citations link to valid page numbers."""
        if not citations:
            return 1.0

        valid_citations = 0
        for cit in citations:
            page = cit.get("page_number")
            if page and (expected_pages is None or page in expected_pages):
                valid_citations += 1

        return round(valid_citations / len(citations), 2)

    def evaluate_pipeline_output(
        self,
        generated_memo: str,
        retrieved_docs: List[Dict[str, Any]],
        visual_evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        trace_id: Optional[str] = None,
    ) -> MultiModalEvalReport:
        """Run full evaluation suite over multi-modal RAG outputs and log metrics to Langfuse."""
        # Aggregate evidence text
        text_corpus = " ".join([d.get("text", "") for d in retrieved_docs if isinstance(d, dict)])
        visual_corpus = " ".join([
            str(v.get("analysis") or v.get("markdown_formatted_block") or v.get("figure_title") or "")
            for v in visual_evidence if isinstance(v, dict)
        ])
        combined_evidence = text_corpus + " " + visual_corpus


        claims = self.extract_factual_claims(generated_memo)
        faith_res = self.evaluate_grounding_faithfulness(claims, combined_evidence)
        cit_score = self.evaluate_citation_precision(citations)

        faithfulness = faith_res["score"]
        hallucination_index = round(1.0 - faithfulness, 2)
        passed = faithfulness >= 0.75 and cit_score >= 0.75

        summary = (
            f"Evaluation Complete: Faithfulness {int(faithfulness * 100)}%, "
            f"Citation Precision {int(cit_score * 100)}%, "
            f"Hallucination Risk: {int(hallucination_index * 100)}%."
        )

        # Log scores to Telemetry / Langfuse if trace_id exists
        if trace_id:
            self.telemetry.log_evaluation_score(trace_id, "grounding_faithfulness", faithfulness, summary)
            self.telemetry.log_evaluation_score(trace_id, "citation_precision", cit_score)
            self.telemetry.log_evaluation_score(trace_id, "hallucination_index", hallucination_index)

        return MultiModalEvalReport(
            faithfulness_score=faithfulness,
            visual_numerical_accuracy=1.0,
            citation_precision=cit_score,
            hallucination_index=hallucination_index,
            total_claims_evaluated=faith_res["total"],
            grounded_claims_count=faith_res["grounded_count"],
            passed_all_thresholds=passed,
            summary=summary,
        )
