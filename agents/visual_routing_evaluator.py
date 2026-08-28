"""Multi-Modal Query Routing Evaluator and Visual Intent Classifier (Task 2B).

Assists the LangGraph Supervisor Agent by classifying user queries and identifying
whether visual extraction, cross-modal verification, or multi-figure comparison is needed.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VisualIntentType(str, Enum):
    """Categorization of visual intent within a user query."""
    VISUAL_EXTRACTION = "visual_extraction"          # Direct chart / table metric query
    CROSS_MODAL_VERIFY = "cross_modal_verify"        # Cross-reference text claims vs chart figures
    MULTI_FIGURE_COMPARE = "multi_figure_compare"    # Compare 2+ visual figures / segments
    SYNTHESIZE_MEMO = "synthesize_memo"              # Comprehensive multi-modal investment memo
    GENERAL_QUERY = "general_query"                  # Pure text or SQL query without visual focus


class VisualIntentScore(BaseModel):
    """Classification score and routing recommendation for the Supervisor."""
    primary_intent: VisualIntentType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_visual_agent: bool
    requires_cross_modal_check: bool
    requires_multi_figure_compare: bool
    target_metric_keywords: List[str] = Field(default_factory=list)
    reasoning: str


class VisualRoutingEvaluator:
    """Evaluates query semantics to guide multi-modal supervisor agent dispatching."""

    VISUAL_KEYWORDS = [
        "chart", "graph", "plot", "figure", "table", "bar chart", "line chart",
        "pie chart", "diagram", "exhibit", "breakdown", "trend", "visual", "axis",
    ]

    VERIFICATION_KEYWORDS = [
        "verify", "cross-reference", "cross reference", "contradiction", "discrepancy",
        "mismatch", "consistent", "fact check", "fact-check", "grounding", "align",
    ]

    COMPARISON_KEYWORDS = [
        "compare", "versus", "vs", "difference between", "relative to", "divergence",
        "segment comparison", "year-over-year comparison", "reconcile",
    ]

    MEMO_KEYWORDS = [
        "investment memo", "memo", "executive summary", "comprehensive analysis",
        "briefing", "report", "synthesis", "diligence report",
    ]

    @classmethod
    def evaluate_query(cls, query: str) -> VisualIntentScore:
        """Classify user query intent to provide routing hints to the LangGraph Supervisor."""
        q_lower = query.lower()

        # Extract metric keywords
        metric_keywords = []
        for kw in ["revenue", "ebitda", "margin", "cagr", "eps", "net income", "operating income", "fcf", "cash flow", "guidance"]:
            if kw in q_lower:
                metric_keywords.append(kw)

        has_visual = any(w in q_lower for w in cls.VISUAL_KEYWORDS)
        has_verify = any(w in q_lower for w in cls.VERIFICATION_KEYWORDS)
        has_compare = any(w in q_lower for w in cls.COMPARISON_KEYWORDS)
        has_memo = any(w in q_lower for w in cls.MEMO_KEYWORDS)

        # 1. Check for Cross-Modal Verification Intent
        if has_verify and (has_visual or "claim" in q_lower or "text" in q_lower):
            return VisualIntentScore(
                primary_intent=VisualIntentType.CROSS_MODAL_VERIFY,
                confidence=0.92,
                requires_visual_agent=True,
                requires_cross_modal_check=True,
                requires_multi_figure_compare=False,
                target_metric_keywords=metric_keywords,
                reasoning="Query requests cross-referencing between textual claims and visual figures.",
            )

        # 2. Check for Multi-Figure Comparison Intent
        if has_compare and (has_visual or "charts" in q_lower or "figures" in q_lower or len(metric_keywords) >= 2):
            return VisualIntentScore(
                primary_intent=VisualIntentType.MULTI_FIGURE_COMPARE,
                confidence=0.88,
                requires_visual_agent=True,
                requires_cross_modal_check=False,
                requires_multi_figure_compare=True,
                target_metric_keywords=metric_keywords,
                reasoning="Query requires comparative variance analysis across multiple visual assets.",
            )

        # 3. Check for Investment Memo Synthesis Intent
        if has_memo:
            return VisualIntentScore(
                primary_intent=VisualIntentType.SYNTHESIZE_MEMO,
                confidence=0.95,
                requires_visual_agent=True,
                requires_cross_modal_check=True,
                requires_multi_figure_compare=False,
                target_metric_keywords=metric_keywords,
                reasoning="Query requests a comprehensive investment memo combining visual, textual, and structured evidence.",
            )

        # 4. Direct Visual Extraction
        if has_visual:
            return VisualIntentScore(
                primary_intent=VisualIntentType.VISUAL_EXTRACTION,
                confidence=0.90,
                requires_visual_agent=True,
                requires_cross_modal_check=False,
                requires_multi_figure_compare=False,
                target_metric_keywords=metric_keywords,
                reasoning="Query directly references a visual chart, figure, or tabular exhibit.",
            )

        # 5. Default General Query
        return VisualIntentScore(
            primary_intent=VisualIntentType.GENERAL_QUERY,
            confidence=0.80,
            requires_visual_agent=False,
            requires_cross_modal_check=False,
            requires_multi_figure_compare=False,
            target_metric_keywords=metric_keywords,
            reasoning="Query does not explicitly target visual document figures; route to Search or SQL agent first.",
        )
