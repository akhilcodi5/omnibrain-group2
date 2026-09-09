"""Visual Analytics & Multi-Modal Tool Integrator Agent.

Handles downstream reasoning, mathematical trend analysis, cross-modal grounding verification,
iterative Self-RAG fact-checking, and executive memo formatting for visual figures in LangGraph.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import AIMessage
from PIL import Image

from agents.cross_modal_self_rag import CrossModalSelfRAG
from agents.cross_modal_verifier import CrossModalVerifier
from agents.state import AgentState
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_tools import VisualMemoFormatter
from app.models.vision_schemas import (
    BoundingBox,
    ChartType,
    CrossModalVerificationReport,
    ExtractedChartData,
    ExtractedTableData,
    VisionExtractionRequest,
    VisionExtractionResponse,
    VisualAnalyticalMemoBlock,
    VisualTrendAnalysis,
)
from app.services.vision_service import VisionService
from storage.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class VisualAnalyticsIntegratorAgent:
    """Specialist agent responsible for downstream reasoning, mathematical validation, 
    cross-referencing visual data against text context, and preparing memo blocks.
    """

    def __init__(
        self,
        vision_service: Optional[VisionService] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.service = vision_service or VisionService()
        self.analytics_engine = VisualAnalyticsEngine()
        self.verifier = CrossModalVerifier()
        self.self_rag = CrossModalSelfRAG(vector_store=vector_store or get_vector_store(in_memory=True))

    async def analyze_visual_asset(
        self,
        image_input: Union[str, bytes, Image.Image],
        query_context: Optional[str] = None,
        expected_type: Optional[ChartType] = None,
        crop_box: Optional[BoundingBox] = None,
    ) -> VisionExtractionResponse:
        """Role 2A Core: Extract structured visual data and markdown interpretation."""
        if isinstance(image_input, str) and image_input.startswith(("data:image", "http://", "https://")):
            request = VisionExtractionRequest(
                image_base64=image_input,
                query_context=query_context,
                expected_type=expected_type or ChartType.BAR,
                crop_box=crop_box,
            )
        elif isinstance(image_input, str):
            request = VisionExtractionRequest(
                image_path=image_input,
                query_context=query_context,
                expected_type=expected_type or ChartType.BAR,
                crop_box=crop_box,
            )
        else:
            base64_str = self.service.encode_image_to_base64(image_input)
            request = VisionExtractionRequest(
                image_base64=base64_str,
                query_context=query_context,
                expected_type=expected_type or ChartType.BAR,
                crop_box=crop_box,
            )
        return await self.service.analyze_figure(request)

    async def analyze_and_verify_figure(
        self,
        image_input: Union[str, bytes, Image.Image],
        text_context: Optional[str] = None,
        query_context: Optional[str] = None,
        expected_type: Optional[ChartType] = None,
        crop_box: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        use_self_rag_loop: bool = True,
    ) -> VisualAnalyticalMemoBlock:
        """Comprehensive downstream pipeline:
        1. Extract chart data
        2. Compute quantitative trends (CAGR, YoY deltas, anomalies)
        3. Cross-reference against text context (with autonomous Self-RAG vector search)
        4. Synthesize into an executive memo block
        """
        # 1. Base extraction
        request = VisionExtractionRequest(
            image_input=image_input,
            query_context=query_context,
            expected_type=expected_type or ChartType.BAR,
            crop_box=crop_box,
            page_number=page_number,
        )

        extraction_resp = await self.service.analyze_figure(request)
        
        # 2. Extract or resolve structured chart data
        chart_data = extraction_resp.chart_data
        if not chart_data:
            try:
                chart_data = await self.service.extract_structured_chart(image_input, query_context)
            except Exception as e:
                logger.warning(f"Could not parse structured chart: {e}")
                chart_data = ExtractedChartData(
                    title="Extracted Visual Figure",
                    summary=extraction_resp.raw_markdown,
                )

        # 3. Compute Quantitative Trend Analytics
        trends = self.analytics_engine.analyze_chart_dataset(chart_data)

        # 4. Cross-Reference against Document Text with Self-RAG Loop
        verification_report = None
        if use_self_rag_loop:
            verification_report = self.self_rag.execute_visual_fact_check_loop(
                chart_data=chart_data,
                original_query=query_context or "",
                initial_text_context=text_context,
            )
        elif text_context:
            verification_report = self.verifier.verify_chart_against_text(chart_data, text_context)

        # 5. Format Executive Investment Memo Block
        title = chart_data.title or "Financial Figure"
        memo_block = VisualMemoFormatter.format_memo_block(
            figure_id=extraction_resp.image_id or "figure_asset",
            figure_title=title,
            chart_data=chart_data,
            trends=trends,
            verification=verification_report,
            page_number=page_number,
        )

        return memo_block

    async def compute_chart_trends(self, chart_data: ExtractedChartData) -> List[VisualTrendAnalysis]:
        """Perform quantitative series trend analysis."""
        return self.analytics_engine.analyze_chart_dataset(chart_data)

    def cross_reference_with_text(
        self, 
        chart_data: ExtractedChartData, 
        text_context: str
    ) -> CrossModalVerificationReport:
        """Verify visual numbers against text context."""
        return self.verifier.verify_chart_against_text(chart_data, text_context)


# Alias for backward compatibility
VisionAgent = VisualAnalyticsIntegratorAgent


async def vision_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node handler for Visual Analytics & Multi-Modal Tool Integrator.
    
    Executes when the Supervisor routes visual analytical tasks to Pod Role 2B.
    Cross-references extracted visual figures against retrieved text docs and formats memo blocks.
    """
    logger.info("Executing Visual Analytics & Multi-Modal Tool Integrator Node in LangGraph...")
    
    vector_store = state.get("vector_store") or get_vector_store(in_memory=True)
    agent = VisualAnalyticsIntegratorAgent(vector_store=vector_store)
    query = state.get("query", "")
    referenced_images: List[str] = state.get("referenced_images", [])
    current_evidence: List[Dict[str, Any]] = state.get("visual_evidence", [])
    retrieved_docs: List[Dict[str, Any]] = state.get("retrieved_docs", [])
    
    # Aggregate text context from retrieved documents
    text_context = " ".join([
        doc.get("content", "") or doc.get("text", "") 
        for doc in retrieved_docs 
        if isinstance(doc, dict)
    ])

    memo_blocks: List[VisualAnalyticalMemoBlock] = []
    citations = []

    if referenced_images:
        for idx, img_path in enumerate(referenced_images):
            memo_block = await agent.analyze_and_verify_figure(
                image_input=img_path,
                text_context=text_context if text_context else None,
                query_context=query,
                page_number=idx + 1,
            )
            memo_blocks.append(memo_block)
            citations.append({
                "source": img_path,
                "citation_tag": memo_block.citation_tag,
                "grounding_score": memo_block.verification_report.grounding_score if memo_block.verification_report else 1.0,
            })
    else:
        logger.info("Vision node called with query context (no explicit image paths attached).")
        memo_block = await agent.analyze_and_verify_figure(
            image_input="sample_figure",
            text_context=text_context if text_context else None,
            query_context=query,
        )
        memo_blocks.append(memo_block)

    # Format synthesized message for LangGraph supervisor
    formatted_sections = "\n\n---\n\n".join([b.markdown_formatted_block for b in memo_blocks])
    ai_message = AIMessage(
        content=f"**[Visual Analytics & Multi-Modal Integration Report]**\n\n{formatted_sections}",
        name="VisualAnalyticsIntegratorAgent"
    )

    # Assess overall grounding
    avg_grounding = 1.0
    if memo_blocks and any(b.verification_report for b in memo_blocks):
        valid_scores = [b.verification_report.grounding_score for b in memo_blocks if b.verification_report]
        avg_grounding = sum(valid_scores) / len(valid_scores)

    return {
        "messages": [ai_message],
        "visual_evidence": current_evidence + [b.model_dump() for b in memo_blocks],
        "citations": state.get("citations", []) + citations,
        "is_grounded": avg_grounding >= 0.75,
        "next_agent": "Supervisor",
    }


class VisionAgent(VisualAnalyticsIntegratorAgent):
    """Subclass/alias for VisionAgent for backward compatibility."""

    async def analyze_visual_asset(
        self,
        image_input: Union[str, bytes, Image.Image],
        query_context: Optional[str] = None,
        expected_type: Optional[ChartType] = None,
        crop_box: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
    ) -> VisualAnalyticalMemoBlock:
        """Alias for analyze_and_verify_figure."""
        return await self.analyze_and_verify_figure(
            image_input=image_input,
            query_context=query_context,
            expected_type=expected_type,
            crop_box=crop_box,
            page_number=page_number,
        )
