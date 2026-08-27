"""FastAPI endpoints for Visual Analytics, Cross-Modal Verification, and Citation Rendering (Task 2B)."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.cross_modal_verifier import CrossModalVerifier
from agents.multi_modal_bridge import get_visual_analytics_tools
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_comparator import VisualComparator
from agents.visual_tools import VisualMemoFormatter
from app.models.vision_schemas import (
    CrossModalVerificationReport,
    ExtractedChartData,
    MultiFigureComparisonReport,
    VisualAnalyticalMemoBlock,
    VisualTrendAnalysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/visual", tags=["Visual Analytics & Tool Integration"])


class VerifyCrossModalRequest(BaseModel):
    chart_data: ExtractedChartData
    text_context: str
    tolerance_percentage: float = 2.0


class ComputeTrendsRequest(BaseModel):
    chart_data: ExtractedChartData


class CompareFiguresRequest(BaseModel):
    chart_a: ExtractedChartData
    chart_b: ExtractedChartData
    comparison_label: str = "Cross-Figure Performance Variance"


class FormatMemoBlockRequest(BaseModel):
    figure_id: str
    figure_title: str
    chart_data: ExtractedChartData
    text_context: Optional[str] = None
    page_number: Optional[int] = None


@router.post("/verify-cross-modal", response_model=CrossModalVerificationReport)
async def verify_cross_modal_endpoint(req: VerifyCrossModalRequest):
    """Cross-reference numerical figures from a visual chart against textual context."""
    try:
        report = CrossModalVerifier.verify_chart_against_text(
            req.chart_data, req.text_context, req.tolerance_percentage
        )
        return report
    except Exception as e:
        logger.error(f"Verification endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute-trends", response_model=List[VisualTrendAnalysis])
async def compute_trends_endpoint(req: ComputeTrendsRequest):
    """Calculate CAGRs, period-to-period percentage deltas, and anomalies from visual chart data."""
    try:
        trends = VisualAnalyticsEngine.analyze_chart_dataset(req.chart_data)
        return trends
    except Exception as e:
        logger.error(f"Trend computation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-figures", response_model=MultiFigureComparisonReport)
async def compare_figures_endpoint(req: CompareFiguresRequest):
    """Perform comparative variance analysis between two extracted visual charts."""
    try:
        report = VisualComparator.compare_two_charts(
            req.chart_a, req.chart_b, req.comparison_label
        )
        return report
    except Exception as e:
        logger.error(f"Comparative analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/format-memo-block", response_model=VisualAnalyticalMemoBlock)
async def format_memo_block_endpoint(req: FormatMemoBlockRequest):
    """Format verified visual findings into an executive investment memo block."""
    try:
        trends = VisualAnalyticsEngine.analyze_chart_dataset(req.chart_data)
        verification = None
        if req.text_context:
            verification = CrossModalVerifier.verify_chart_against_text(req.chart_data, req.text_context)

        block = VisualMemoFormatter.format_memo_block(
            figure_id=req.figure_id,
            figure_title=req.figure_title,
            chart_data=req.chart_data,
            trends=trends,
            verification=verification,
            page_number=req.page_number,
        )
        return block
    except Exception as e:
        logger.error(f"Memo block formatting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=List[Dict[str, Any]])
async def list_visual_tools_endpoint():
    """List all registered multi-modal visual tools available for the Supervisor Agent."""
    tools = get_visual_analytics_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "args_schema": str(t.args_schema),
        }
        for t in tools
    ]
