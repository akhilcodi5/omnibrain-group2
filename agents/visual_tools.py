"""Multi-Modal Tool Integrations and Memo Formatter for LangGraph Supervisor Agent."""

import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from agents.cross_modal_verifier import CrossModalVerifier
from agents.visual_analytics import VisualAnalyticsEngine
from app.models.vision_schemas import (
    CrossModalVerificationReport,
    ExtractedChartData,
    VisualAnalyticalMemoBlock,
    VisualTrendAnalysis,
)

logger = logging.getLogger(__name__)


class VisualMemoFormatter:
    """Formats verified multi-modal visual findings into structured executive investment memo sections."""

    @staticmethod
    def format_memo_block(
        figure_id: str,
        figure_title: str,
        chart_data: ExtractedChartData,
        trends: List[VisualTrendAnalysis],
        verification: Optional[CrossModalVerificationReport] = None,
        page_number: Optional[int] = None,
    ) -> VisualAnalyticalMemoBlock:
        """Construct a full VisualAnalyticalMemoBlock ready for inclusion in the final investment memo."""
        page_str = f"Page {page_number}" if page_number else "Extracted Figure"
        citation_tag = f"[{figure_title} - {page_str}]"

        # Build Markdown content
        md_lines = [
            f"### 📊 Visual Analytical Evidence: {figure_title}",
            f"**Source Citation**: `{citation_tag}` | **Figure Type**: `{chart_data.chart_type.value.upper()}`\n",
            f"**Executive Takeaway**:\n{chart_data.summary}\n",
            "**Quantitative Trend Analysis**:"
        ]

        takeaways = []
        risk_warnings = []

        for t in trends:
            cagr_str = f" (CAGR: {t.cagr_percentage}%)" if t.cagr_percentage is not None else ""
            delta_str = f"{'+' if t.total_percentage_change and t.total_percentage_change > 0 else ''}{t.total_percentage_change}%"
            md_lines.append(f"- **{t.series_name}**: Trajectory is `{t.trend_direction.value.upper()}` with total delta of **{delta_str}**{cagr_str}.")
            md_lines.append(f"  - *Analysis*: {t.analytical_takeaway}")
            takeaways.append(f"{t.series_name}: {t.trend_direction.value.upper()} trajectory ({delta_str})")

            for anomaly in t.detected_anomalies:
                risk_warnings.append(f"[{figure_title}] {anomaly}")
                md_lines.append(f"  - ⚠️ *Anomaly Alert*: {anomaly}")

        if verification:
            badge = "🟢 VERIFIED" if verification.grounding_score >= 0.8 else "🟡 REVIEW REQUIRED"
            md_lines.append(f"\n**Cross-Modal Verification Status**: `{badge}` (Grounding Confidence: {int(verification.grounding_score * 100)}%)")
            md_lines.append(f"- *Summary*: {verification.summary_assessment}")
            
            if verification.critical_discrepancies:
                md_lines.append("- *Flagged Contradictions*:")
                for disc in verification.critical_discrepancies:
                    md_lines.append(f"  - 🔴 {disc}")
                    risk_warnings.append(disc)

        formatted_md = "\n".join(md_lines)

        return VisualAnalyticalMemoBlock(
            figure_id=figure_id,
            figure_title=figure_title,
            page_number=page_number,
            executive_summary=chart_data.summary,
            trend_analytics=trends,
            verification_report=verification,
            key_takeaways=takeaways,
            risk_warnings=risk_warnings,
            citation_tag=citation_tag,
            markdown_formatted_block=formatted_md,
        )


# =====================================================================
# LangGraph / LangChain Tools for Supervisor & Sub-Agents
# =====================================================================

@tool
def verify_visual_numbers_against_text(chart_json_str: str, text_context: str) -> str:
    """Cross-references parsed visual chart numbers against textual document context to detect reporting discrepancies.
    
    Args:
        chart_json_str: JSON string conforming to ExtractedChartData schema.
        text_context: Relevant textual sentences or paragraphs retrieved from document chunks.
        
    Returns:
        JSON string containing the CrossModalVerificationReport.
    """
    try:
        data = json.loads(chart_json_str)
        chart_data = ExtractedChartData.model_validate(data)
        report = CrossModalVerifier.verify_chart_against_text(chart_data, text_context)
        return report.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error in verify_visual_numbers_against_text tool: {e}")
        return json.dumps({"error": f"Failed verification: {str(e)}"})


@tool
def compute_visual_figure_trends(chart_json_str: str) -> str:
    """Computes quantitative CAGRs, period-over-period delta growth rates, and statistical anomalies from a chart dataset.
    
    Args:
        chart_json_str: JSON string conforming to ExtractedChartData schema.
        
    Returns:
        JSON string containing list of VisualTrendAnalysis records.
    """
    try:
        data = json.loads(chart_json_str)
        chart_data = ExtractedChartData.model_validate(data)
        trends = VisualAnalyticsEngine.analyze_chart_dataset(chart_data)
        return json.dumps([t.model_dump() for t in trends], indent=2)
    except Exception as e:
        logger.error(f"Error in compute_visual_figure_trends tool: {e}")
        return json.dumps({"error": f"Failed trend computation: {str(e)}"})


@tool
def format_visual_memo_section_tool(
    figure_title: str,
    chart_json_str: str,
    text_context: Optional[str] = None,
    page_number: Optional[int] = None,
) -> str:
    """Synthesizes verified visual findings, calculated trends, and citations into an executive investment memo block.
    
    Args:
        figure_title: Name of the chart or figure (e.g. 'FY24 Quarterly Revenue Growth').
        chart_json_str: JSON string conforming to ExtractedChartData schema.
        text_context: Optional document text context to cross-reference.
        page_number: Optional PDF page number for citation tagging.
        
    Returns:
        Formatted markdown block ready for the final investment memo.
    """
    try:
        data = json.loads(chart_json_str)
        chart_data = ExtractedChartData.model_validate(data)
        trends = VisualAnalyticsEngine.analyze_chart_dataset(chart_data)
        
        verification = None
        if text_context:
            verification = CrossModalVerifier.verify_chart_against_text(chart_data, text_context)
            
        memo_block = VisualMemoFormatter.format_memo_block(
            figure_id=figure_title.lower().replace(" ", "_"),
            figure_title=figure_title,
            chart_data=chart_data,
            trends=trends,
            verification=verification,
            page_number=page_number,
        )
        return memo_block.markdown_formatted_block
    except Exception as e:
        logger.error(f"Error in format_visual_memo_section_tool: {e}")
        return f"### Visual Analytical Error\nFailed to format memo section: {str(e)}"
