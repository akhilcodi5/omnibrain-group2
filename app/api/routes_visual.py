"""FastAPI endpoints for Visual Analytics, Cross-Modal Verification, Citation Rendering, and Benchmarking (Task 2B)."""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from agents.cross_modal_verifier import CrossModalVerifier
from agents.multi_modal_bridge import get_visual_analytics_tools
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_comparator import VisualComparator
from agents.visual_tools import VisualMemoFormatter
from app.models.vision_schemas import (
    BoundingBox,
    ChartType,
    CrossModalVerificationReport,
    ExtractedChartData,
    ExtractedTableData,
    MultiFigureComparisonReport,
    VerificationStatus,
    VisionExtractionRequest,
    VisionExtractionResponse,
    VisualAnalyticalMemoBlock,
    VisualCitationPayload,
    VisualTrendAnalysis,
)
from app.services.citation_renderer import CitationRenderer
from app.services.vision_service import VisionService
from eval.benchmark_runner import BenchmarkRunner
from storage.image_store import get_image_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/visual", tags=["Multi-Modal Vision & Analytics"])


class ExtractChartRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    query_context: Optional[str] = None


class ExtractTableRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    query_context: Optional[str] = None


class SaveAssetRequest(BaseModel):
    image_base64: str
    doc_id: str
    page_number: int
    figure_name: str = "figure"
    figure_type: str = "chart"
    bounding_box: Optional[BoundingBox] = None


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


class RenderOverlayRequest(BaseModel):
    figure_title: str
    page_number: Optional[int] = None
    bounding_box: BoundingBox
    status: VerificationStatus = VerificationStatus.VERIFIED_MATCH
    grounding_confidence: float = 1.0


@router.post("/verify-cross-modal", response_model=CrossModalVerificationReport)
async def verify_cross_modal_endpoint(req: VerifyCrossModalRequest):
    """Cross-reference numerical figures from a visual chart against textual context."""
    try:
        report = CrossModalVerifier.verify_chart_against_text(
            req.chart_data, req.text_context, req.tolerance_percentage
        )
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute-trends", response_model=List[VisualTrendAnalysis])
async def compute_trends_endpoint(req: ComputeTrendsRequest):
    """Calculate CAGRs, period-to-period percentage deltas, and anomalies from visual chart data."""
    try:
        trends = VisualAnalyticsEngine.analyze_chart_dataset(req.chart_data)
        return trends
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memo block formatting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/render-overlay", response_model=VisualCitationPayload)
async def render_citation_overlay_endpoint(req: RenderOverlayRequest):
    """Render a dynamic highlighted bounding-box overlay image for citations."""
    try:
        canvas = Image.new("RGB", (800, 1000), color="white")
        payload = CitationRenderer.render_citation_overlay(
            image=canvas,
            bounding_box=req.bounding_box,
            figure_title=req.figure_title,
            page_number=req.page_number,
            status=req.status,
            grounding_confidence=req.grounding_confidence,
        )
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Overlay rendering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark", response_model=Dict[str, Any])
async def run_benchmark_endpoint():
    """Execute automated multi-modal accuracy and grounding benchmark test cases."""
    try:
        runner = BenchmarkRunner()
        card = runner.run_all_benchmarks()
        return card
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract", response_model=VisionExtractionResponse)
async def extract_visual_figure_endpoint(req: VisionExtractionRequest):
    """Direct visual extraction and multimodal reasoning over an image/figure using the configured VLM."""
    try:
        service = VisionService()
        response = await service.analyze_figure(req)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-chart", response_model=ExtractedChartData)
async def extract_chart_endpoint(req: ExtractChartRequest):
    """Direct structured chart extraction returning validated ExtractedChartData schema."""
    try:
        service = VisionService()
        image_input = req.image_path or req.image_base64
        if not image_input:
            raise HTTPException(status_code=400, detail="Must provide either image_path or image_base64")
        chart_data = await service.extract_structured_chart(image_input, query_context=req.query_context)
        return chart_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-table", response_model=ExtractedTableData)
async def extract_table_endpoint(req: ExtractTableRequest):
    """Direct structured financial table extraction returning validated ExtractedTableData schema."""
    try:
        service = VisionService()
        image_input = req.image_path or req.image_base64
        if not image_input:
            raise HTTPException(status_code=400, detail="Must provide either image_path or image_base64")
        table_data = await service.extract_structured_table(image_input, query_context=req.query_context)
        return table_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Table extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assets", response_model=Dict[str, Any])
async def save_visual_asset_endpoint(req: SaveAssetRequest):
    """Save an extracted visual chart/table asset and cache thumbnail."""
    try:
        store = get_image_store()
        meta = store.save_figure_asset(
            image_input=req.image_base64,
            doc_id=req.doc_id,
            page_number=req.page_number,
            figure_name=req.figure_name,
            figure_type=req.figure_type,
            bounding_box=req.bounding_box,
        )
        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset saving error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/{doc_id}", response_model=List[Dict[str, Any]])
async def list_visual_assets_endpoint(doc_id: str):
    """List all extracted visual assets and thumbnails for a document."""
    try:
        store = get_image_store()
        assets = store.list_document_assets(doc_id)
        return assets
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset listing error: {e}")
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


@router.get("/artifacts/recent")
async def get_recent_artifacts(doc_name: Optional[str] = None):
    """List recent visual artifacts from storage/extracted_images for instant frontend hydration."""
    extracted_dir = "storage/extracted_images"
    if not os.path.exists(extracted_dir):
        return {"artifacts": [], "pdf_name": None, "total_pages": 0}

    files = [
        f for f in os.listdir(extracted_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not files:
        return {"artifacts": [], "pdf_name": None, "total_pages": 0}

    # Group by document name
    doc_map = {}
    for f in files:
        m = re.match(r"(?:chart|img|table)_(.+)_[pP](\d+)_", f)
        if m:
            doc = m.group(1)
            mtime = os.path.getmtime(os.path.join(extracted_dir, f))
            if doc not in doc_map or mtime > doc_map[doc]["mtime"]:
                doc_map[doc] = {"mtime": mtime, "has_charts": f.startswith("chart_")}

    target_doc = None
    if doc_name:
        clean_name = doc_name.replace(".pdf", "").replace(".PDF", "")
        for d in doc_map:
            if clean_name.lower() in d.lower():
                target_doc = d
                break

    if not target_doc:
        sorted_docs = sorted(
            doc_map.keys(),
            key=lambda d: (
                doc_map[d]["has_charts"],
                not d.startswith("q3_"),
                doc_map[d]["mtime"]
            ),
            reverse=True
        )
        target_doc = sorted_docs[0] if sorted_docs else None

    doc_files = [f for f in files if target_doc and target_doc in f] if target_doc else files[:20]

    # Deduplicate multiple extractions of same page and chart number
    doc_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(extracted_dir, f)),
        reverse=True
    )
    seen_exhibits = set()
    deduped_files = []
    for fname in doc_files:
        p_num_match = re.search(r"_[pP](\d+)_(\d+)_", fname)
        if p_num_match:
            key = (int(p_num_match.group(1)), int(p_num_match.group(2)))
            if key in seen_exhibits:
                continue
            seen_exhibits.add(key)
        deduped_files.append(fname)

    # Natural sort by page number
    def sort_key(fname):
        p_match = re.search(r"_[pP](\d+)_", fname)
        page = int(p_match.group(1)) if p_match else 0
        return (page, fname)

    deduped_files.sort(key=sort_key)

    artifacts = []
    max_page = 1
    table_count = 0
    chart_count = 0
    for idx, fname in enumerate(deduped_files):
        p_match = re.search(r"_[pP](\d+)_", fname)
        page = int(p_match.group(1)) if p_match else 1
        if page > max_page:
            max_page = page

        is_table = fname.startswith("table_")
        if is_table:
            table_count += 1
            label = f"Extracted Table {table_count}"
        else:
            chart_count += 1
            label = f"Extracted Chart {chart_count}"

        artifacts.append({
            "index": idx,
            "filename": fname,
            "url": f"/api/v1/images/{fname}",
            "title": label,
            "type": "table" if is_table else "chart",
            "page": page,
        })

    readable_doc_name = (
        (target_doc + ".pdf")
        if target_doc and not target_doc.lower().endswith(".pdf")
        else (target_doc or "Active Document")
    )
    return {
        "pdf_name": readable_doc_name,
        "total_pages": max_page,
        "artifacts": artifacts
    }

