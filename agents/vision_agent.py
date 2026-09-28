"""Visual Analytics & Multi-Modal Tool Integrator Agent.

Handles downstream reasoning, mathematical trend analysis, cross-modal grounding verification,
iterative Self-RAG fact-checking, and executive memo formatting for visual figures in LangGraph.
"""

import base64
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import AIMessage
from PIL import Image

from app.core.telemetry import get_telemetry_manager


from agents.cross_modal_self_rag import CrossModalSelfRAG
from agents.cross_modal_verifier import CrossModalVerifier
from agents.state import AgentState
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_tools import VisualMemoFormatter
from app.core.telemetry import get_telemetry_manager
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
        kwargs = {
            "query_context": query_context,
            "expected_type": expected_type or ChartType.BAR,
            "crop_box": crop_box,
            "page_number": page_number,
        }
        
        if isinstance(image_input, str):
            if image_input.startswith(("data:image", "http://", "https://")) or not os.path.exists(image_input):
                kwargs["image_base64"] = image_input
            else:
                kwargs["image_path"] = image_input
        elif isinstance(image_input, bytes):
            kwargs["image_base64"] = base64.b64encode(image_input).decode('utf-8')
        elif isinstance(image_input, Image.Image):
            kwargs["image_base64"] = self.service.encode_image_to_base64(image_input)
            
        request = VisionExtractionRequest(**kwargs)

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
    
    start_time = time.time()
    vector_store = state.get("vector_store") or get_vector_store(in_memory=False)
    agent = VisualAnalyticsIntegratorAgent(vector_store=vector_store)
    query = state.get("query", "")
    pdf_name = state.get("pdf_name")
    referenced_images: List[str] = state.get("referenced_images", [])
    current_evidence: List[Dict[str, Any]] = state.get("visual_evidence", [])
    retrieved_docs: List[Dict[str, Any]] = state.get("retrieved_docs", [])

    # 1. Document-Level Isolation: Retain only images originating from the active PDF
    if pdf_name and referenced_images:
        clean_doc = os.path.splitext(os.path.basename(pdf_name))[0].replace(" ", "_").lower()
        scoped = [img for img in referenced_images if clean_doc in img.lower()]
        if scoped:
            referenced_images = scoped
        else:
            # If active PDF is specified but has no matching images, do not use old PDF images
            referenced_images = []

    # 2. Smart Relevance & Chart Prioritization
    q_lower = query.lower()
    is_chart_query = any(k in q_lower for k in ["bar chart", "chart", "graph", "plot", "figure", "visual", "cagr", "trajectory"])
    
    if referenced_images:
        # Separate charts from tables and general images
        chart_images = [img for img in referenced_images if "chart_" in os.path.basename(img).lower()]
        table_images = [img for img in referenced_images if "table_" in os.path.basename(img).lower()]
        other_images = [img for img in referenced_images if img not in chart_images and img not in table_images]

        if is_chart_query and chart_images:
            # Prioritize genuine charts over data tables
            referenced_images = chart_images + table_images
        elif "table" in q_lower and table_images:
            referenced_images = table_images + chart_images

        # Filter by relevant pages from retrieved text docs if available
        if retrieved_docs:
            relevant_pages = {doc.get("page_number") for doc in retrieved_docs if doc.get("page_number") is not None}
            if relevant_pages:
                page_matched = []
                for img in referenced_images:
                    match = re.search(r'_p(\d+)_', img)
                    if match and int(match.group(1)) in relevant_pages:
                        page_matched.append(img)
                if page_matched:
                    referenced_images = page_matched

        # Concurrency & Rate Limit Safety: Limit to top 2 most relevant visual figures
        referenced_images = referenced_images[:2]

    # 3. Graceful Absence Handling: If a visual chart was requested but none exist in the active document
    if is_chart_query and not referenced_images:
        doc_label = os.path.basename(pdf_name) if pdf_name else "the uploaded document"
        notice_content = (
            f"**[Visual Analytics Notice]**\n\n"
            f"⚠️ **No Financial Bar Charts Detected**: No quarterly revenue bar charts or visual financial exhibits "
            f"were found in `{doc_label}`.\n\n"
            f"- **Active Document**: `{doc_label}`\n"
            f"- **Guidance**: To extract numerical figures from bar charts, calculate CAGR, and assess YoY trajectories, "
            f"please upload a corporate financial PDF (such as a 10-K filing or quarterly earnings release with embedded charts)."
        )
        ai_message = AIMessage(
            content=notice_content,
            name="VisualAnalyticsIntegratorAgent"
        )
        return {
            "messages": [ai_message],
            "visual_evidence": current_evidence,
            "citations": state.get("citations", []),
            "is_grounded": True,
            "next_agent": "Supervisor",
        }

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
            img_page = idx + 1
            match = re.search(r'_p(\d+)_', img_path)
            if match:
                img_page = int(match.group(1))

            memo_block = await agent.analyze_and_verify_figure(
                image_input=img_path,
                text_context=text_context if text_context else None,
                query_context=query,
                page_number=img_page,
            )
            memo_blocks.append(memo_block)
            citations.append({
                "source": str(img_path) if isinstance(img_path, str) else f"figure_{idx+1}",
                "citation_tag": memo_block.citation_tag,
                "grounding_score": memo_block.verification_report.grounding_score if memo_block.verification_report else 1.0,
            })
    else:
        logger.info("Vision node called without referenced image paths.")

    # Format synthesized message for LangGraph supervisor
    formatted_sections = "\n\n---\n\n".join([b.markdown_formatted_block for b in memo_blocks]) if memo_blocks else "No visual figures processed."
    ai_message = AIMessage(
        content=f"**[Visual Analytics & Multi-Modal Integration Report]**\n\n{formatted_sections}",
        name="VisualAnalyticsIntegratorAgent"
    )

    # Assess overall grounding
    avg_grounding = 1.0
    if memo_blocks and any(b.verification_report for b in memo_blocks):
        valid_scores = [b.verification_report.grounding_score for b in memo_blocks if b.verification_report]
        avg_grounding = sum(valid_scores) / len(valid_scores)

    elapsed = time.time() - start_time
    trace_id = state.get("trace_id")
    if trace_id:
        get_telemetry_manager().log_agent_step(
            trace_id=trace_id,
            agent_name="VisionAgent",
            action="MultiModalAnalysis",
            model="gemini-1.5-pro",
            input_data=f"Query: {query}, Images: {referenced_images}",
            output_data=formatted_sections,
            prompt_tokens=800 * len(memo_blocks) if memo_blocks else 800,
            completion_tokens=len(formatted_sections.split()),
            latency_seconds=elapsed
        )

    return {
        "messages": [ai_message],
        "visual_evidence": current_evidence + [b.model_dump() for b in memo_blocks],
        "citations": state.get("citations", []) + citations,
        "is_grounded": avg_grounding >= 0.75,
        "next_agent": "Supervisor",
    }


class VisionAgent(VisualAnalyticsIntegratorAgent):
    """Subclass/alias for VisionAgent for backward compatibility."""

    def __init__(self, vision_service: Optional[Any] = None, vector_store: Optional[Any] = None):
        super().__init__(vector_store=vector_store)
        if vision_service:
            self.service = vision_service

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




