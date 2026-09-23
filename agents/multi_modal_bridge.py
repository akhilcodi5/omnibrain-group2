"""Multi-Modal Supervisor Bridge & LangGraph Tool Registry (Task 2B).

Provides a centralized registry of visual analytical tools and routing hooks for the
LangGraph Supervisor Agent and other collaborative pods.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool, tool
from PIL import Image

from agents.cross_modal_verifier import CrossModalVerifier
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_comparator import VisualComparator
from agents.visual_tools import (
    VisualMemoFormatter,
    compute_visual_figure_trends,
    format_visual_memo_section_tool,
    verify_visual_numbers_against_text,
)
from app.models.vision_schemas import (
    BoundingBox,
    ExtractedChartData,
    VerificationStatus,
)
from app.services.citation_renderer import CitationRenderer

logger = logging.getLogger(__name__)


@tool
def compare_two_visual_figures(
    chart_a_json: str,
    chart_b_json: str,
    comparison_label: str = "Cross-Period Variance",
) -> str:
    """Compares two extracted visual charts/tables to identify variances, metric divergence, and reporting shifts.
    
    Args:
        chart_a_json: JSON string of primary ExtractedChartData.
        chart_b_json: JSON string of secondary ExtractedChartData to compare against.
        comparison_label: Description of comparison (e.g. 'FY23 vs FY24 Segment Revenue').
        
    Returns:
        JSON string containing the MultiFigureComparisonReport.
    """
    try:
        data_a = json.loads(chart_a_json)
        data_b = json.loads(chart_b_json)
        chart_a = ExtractedChartData.model_validate(data_a)
        chart_b = ExtractedChartData.model_validate(data_b)

        report = VisualComparator.compare_two_charts(chart_a, chart_b, comparison_label)
        return report.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error in compare_two_visual_figures tool: {e}")
        return json.dumps({"error": f"Comparison failed: {str(e)}"})


@tool
def generate_visual_citation_overlay_tool(
    figure_title: str,
    page_number: int,
    bbox_ymin: float,
    bbox_xmin: float,
    bbox_ymax: float,
    bbox_xmax: float,
    status_str: str = "verified_match",
) -> str:
    """Generates visual citation bounding box metadata and thumbnail bundle for Streamlit citation preview.
    
    Args:
        figure_title: Title/name of the visual asset.
        page_number: PDF page number.
        bbox_ymin: Top boundary (normalized 0.0 - 1.0).
        bbox_xmin: Left boundary (normalized 0.0 - 1.0).
        bbox_ymax: Bottom boundary (normalized 0.0 - 1.0).
        bbox_xmax: Right boundary (normalized 0.0 - 1.0).
        status_str: 'verified_match' or 'discrepancy_detected'.
        
    Returns:
        JSON string containing VisualCitationPayload metadata.
    """
    try:
        # Create blank preview canvas for thumbnail rendering
        canvas = Image.new("RGB", (800, 1000), color="white")
        box = BoundingBox(ymin=bbox_ymin, xmin=bbox_xmin, ymax=bbox_ymax, xmax=bbox_xmax, is_normalized=True)
        status = VerificationStatus(status_str) if status_str in VerificationStatus._value2member_map_ else VerificationStatus.VERIFIED_MATCH

        payload = CitationRenderer.render_citation_overlay(
            image=canvas,
            bounding_box=box,
            figure_title=figure_title,
            page_number=page_number,
            status=status,
            grounding_confidence=1.0 if status == VerificationStatus.VERIFIED_MATCH else 0.7,
        )
        return payload.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error generating visual citation: {e}")
        return json.dumps({"error": f"Citation generation failed: {str(e)}"})


def get_visual_analytics_tools() -> List[BaseTool]:
    """Retrieve the complete tool suite provided by Pod Role 2B for the LangGraph Supervisor."""
    return [
        verify_visual_numbers_against_text,
        compute_visual_figure_trends,
        compare_two_visual_figures,
        format_visual_memo_section_tool,
        generate_visual_citation_overlay_tool,
    ]
