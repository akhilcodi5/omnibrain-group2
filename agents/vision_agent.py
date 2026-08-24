"""Vision-Language Model (VLM) Agent for interpreting charts, graphs, and financial tables."""

import logging
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import AIMessage

from agents.state import AgentState
from app.models.vision_schemas import (
    ChartType,
    VisionExtractionRequest,
    VisionExtractionResponse,
)
from app.services.vision_service import VisionService

logger = logging.getLogger(__name__)


class VisionAgent:
    """Agent specialized in Multi-Modal VLM extraction, chart reasoning, and numerical validation."""

    def __init__(self, vision_service: Optional[VisionService] = None):
        self.service = vision_service or VisionService()

    async def analyze_visual_asset(
        self,
        image_input: Union[str, bytes],
        query_context: Optional[str] = None,
        expected_type: Optional[ChartType] = None,
    ) -> VisionExtractionResponse:
        """Analyze a visual asset (chart, table, diagram) with contextual reasoning."""
        if isinstance(image_input, str) and image_input.startswith(("data:image", "http://", "https://")):
            request = VisionExtractionRequest(
                image_base64=image_input,
                query_context=query_context,
                expected_type=expected_type,
            )
        elif isinstance(image_input, str):
            request = VisionExtractionRequest(
                image_path=image_input,
                query_context=query_context,
                expected_type=expected_type,
            )
        else:
            base64_str = self.service.encode_image_to_base64(image_input)
            request = VisionExtractionRequest(
                image_base64=base64_str,
                query_context=query_context,
                expected_type=expected_type,
            )

        return await self.service.analyze_figure(request)

    async def extract_chart_metrics(
        self, 
        image_path: str, 
        query_context: Optional[str] = None
    ) -> VisionExtractionResponse:
        """Specialized extraction for quantitative bar/line/pie charts."""
        return await self.analyze_visual_asset(
            image_input=image_path,
            query_context=query_context,
            expected_type=ChartType.BAR,
        )

    async def extract_table_matrix(
        self, 
        image_path: str, 
        query_context: Optional[str] = None
    ) -> VisionExtractionResponse:
        """Specialized extraction for balance sheet and tabular financial figures."""
        return await self.analyze_visual_asset(
            image_input=image_path,
            query_context=query_context,
            expected_type=ChartType.TABLE,
        )


async def vision_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node handler for Vision Agent.
    
    Executes when the Supervisor routes a visual or chart-related task to the Vision Specialist.
    """
    logger.info("Executing Vision Agent Node in LangGraph workflow...")
    
    agent = VisionAgent()
    query = state.get("query", "")
    referenced_images: List[str] = state.get("referenced_images", [])
    current_evidence: List[Dict[str, Any]] = state.get("visual_evidence", [])

    extracted_results = []
    
    if referenced_images:
        for img_path in referenced_images:
            response = await agent.analyze_visual_asset(
                image_input=img_path,
                query_context=query,
            )
            extracted_results.append({
                "image_path": img_path,
                "analysis": response.raw_markdown,
                "chart_data": response.chart_data.model_dump() if response.chart_data else None,
                "table_data": response.table_data.model_dump() if response.table_data else None,
            })
    else:
        # Fallback if no specific image path was attached
        logger.warning("Vision node invoked without explicit referenced_images.")
        response = await agent.analyze_visual_asset(
            image_input="",
            query_context=query,
        )
        extracted_results.append({
            "image_path": "unspecified",
            "analysis": response.raw_markdown,
        })

    # Format synthesized message for conversation state
    summary_text = "\n\n".join([r["analysis"] for r in extracted_results])
    ai_message = AIMessage(
        content=f"**[Vision Specialist Analysis]**\n\n{summary_text}",
        name="VisionAgent"
    )

    return {
        "messages": [ai_message],
        "visual_evidence": current_evidence + extracted_results,
        "next_agent": "Supervisor",
    }
