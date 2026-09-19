"""Vision Service handling multimodal preprocessing, VLM inference, and structured parsing."""

import base64
import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image

from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    ExtractedTableData,
    VisionExtractionRequest,
    VisionExtractionResponse,
)
from agents.vision_prompts import (
    CHART_EXTRACTION_PROMPT,
    FIGURE_REASONING_PROMPT,
    TABLE_EXTRACTION_PROMPT,
    VISION_SYSTEM_PROMPT,
)
from app.services.image_preprocessor import ImagePreprocessor
from app.services.vlm_engine import BaseVisionEngine, get_vision_engine

logger = logging.getLogger(__name__)


class VisionService:
    """Service to handle Vision-Language Model inference for charts, tables, and visual figures."""

    def __init__(
        self, 
        engine: Optional[BaseVisionEngine] = None,
        provider: Optional[str] = None,
    ):
        self.engine = engine or get_vision_engine(provider)
        self.preprocessor = ImagePreprocessor()

    @staticmethod
    def encode_image_to_base64(image_input: Union[str, bytes, Image.Image]) -> str:
        """Encode image from file path, raw bytes, or PIL Image to a base64 string."""
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                with open(image_input, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            # If string is already base64
            return image_input

        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")

        elif isinstance(image_input, Image.Image):
            buffered = BytesIO()
            if image_input.mode in ("RGBA", "P"):
                image_input = image_input.convert("RGB")
            image_input.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def get_image_media_type(image_path: Optional[str] = None) -> str:
        """Determine image media type from file extension."""
        if image_path:
            ext = os.path.splitext(image_path)[1].lower()
            if ext in (".png", ".webp", ".gif"):
                return f"image/{ext[1:]}"
        return "image/jpeg"

    async def analyze_figure(
        self, 
        request: VisionExtractionRequest
    ) -> VisionExtractionResponse:
        """Perform multimodal extraction and reasoning on a given figure with optional preprocessing."""
        try:
            # 1. Load image as PIL for optional cropping and preprocessing
            pil_image = None
            if request.image_path and os.path.exists(request.image_path):
                pil_image = Image.open(request.image_path)
            elif request.image_base64:
                raw_bytes = base64.b64decode(request.image_base64)
                pil_image = Image.open(BytesIO(raw_bytes))

            # Apply sub-bounding box crop if specified
            if pil_image and request.crop_box:
                pil_image = self.preprocessor.crop_bounding_box(
                    pil_image,
                    box=(
                        int(request.crop_box.xmin),
                        int(request.crop_box.ymin),
                        int(request.crop_box.xmax),
                        int(request.crop_box.ymax),
                    ),
                    normalized=request.crop_box.is_normalized,
                )

            # Preprocess image for VLM readability & token scaling
            if pil_image:
                pil_image = self.preprocessor.resize_for_vlm(pil_image)
                b64_img = self.encode_image_to_base64(pil_image)
            elif request.image_base64:
                b64_img = request.image_base64
            else:
                raise ValueError("No valid image input provided in request.")

            media_type = self.get_image_media_type(request.image_path)
            context = request.query_context or "General quantitative extraction and financial analysis."

            # 2. Select prompt template
            if request.expected_type == ChartType.TABLE:
                prompt_text = TABLE_EXTRACTION_PROMPT.format(query_context=context)
            elif request.expected_type in (ChartType.BAR, ChartType.LINE, ChartType.PIE, ChartType.AREA):
                prompt_text = CHART_EXTRACTION_PROMPT.format(query_context=context)
            else:
                prompt_text = FIGURE_REASONING_PROMPT.format(query_context=context)

            # 3. Call modular VLM Engine
            response = await self.engine.generate_response(
                base64_image=b64_img,
                prompt=prompt_text,
                system_prompt=VISION_SYSTEM_PROMPT,
                media_type=media_type,
            )

            # 4. If a structured chart was requested, also attempt structured schema extraction
            chart_data = None
            if request.expected_type in (ChartType.BAR, ChartType.LINE, ChartType.PIE, ChartType.AREA):
                try:
                    structured_json = await self.engine.extract_structured_json(
                        base64_image=b64_img,
                        prompt=CHART_EXTRACTION_PROMPT.format(query_context=context),
                        system_prompt=VISION_SYSTEM_PROMPT,
                        media_type=media_type,
                    )
                    if structured_json and "series" in structured_json:
                        chart_data = ExtractedChartData.model_validate(structured_json)
                except Exception as ex:
                    logger.debug(f"Structured chart extraction fallback: {ex}")

            return VisionExtractionResponse(
                image_id=os.path.basename(request.image_path) if request.image_path else "figure_asset",
                chart_data=chart_data,
                raw_markdown=response.get("content", ""),
                metadata={
                    "model": response.get("model", "unknown"),
                    "usage": response.get("usage", {}),
                }
            )

        except Exception as e:
            logger.error(f"Error during vision extraction: {str(e)}", exc_info=True)
            return VisionExtractionResponse(
                raw_markdown=f"### Vision Extraction Error\n\nFailed to process image: {str(e)}",
                metadata={"error": str(e)}
            )

    async def extract_structured_chart(
        self,
        image_input: Union[str, bytes, Image.Image],
        query_context: Optional[str] = None,
    ) -> ExtractedChartData:
        """Dedicated method returning guaranteed ExtractedChartData schema."""
        b64_img = self.encode_image_to_base64(image_input)
        context = query_context or "Extract all series, labels, units, and values from this chart."
        prompt = CHART_EXTRACTION_PROMPT.format(query_context=context)

        raw_json = await self.engine.extract_structured_json(
            base64_image=b64_img,
            prompt=prompt,
            system_prompt=VISION_SYSTEM_PROMPT,
        )
        try:
            if not isinstance(raw_json, dict):
                raw_json = {}
            if "summary" not in raw_json:
                raw_json["summary"] = raw_json.get("error") or raw_json.get("raw_text") or "No chart summary provided."
            if "chart_type" not in raw_json or not raw_json["chart_type"]:
                raw_json["chart_type"] = "unknown"
            return ExtractedChartData.model_validate(raw_json)
        except Exception as e:
            logger.warning(f"Chart schema validation fallback: {e}")
            return ExtractedChartData(
                title=raw_json.get("title", "Visual Figure"),
                chart_type=ChartType.UNKNOWN,
                summary=raw_json.get("summary") or raw_json.get("error") or str(raw_json.get("raw_text", "Could not parse valid chart data.")),
                key_insights=[str(e)] if "error" in raw_json else [],
            )

    async def extract_structured_table(
        self,
        image_input: Union[str, bytes, Image.Image],
        query_context: Optional[str] = None,
    ) -> ExtractedTableData:
        """Dedicated method returning guaranteed ExtractedTableData schema."""
        b64_img = self.encode_image_to_base64(image_input)
        context = query_context or "Extract all columns, headers, rows, and key financial figures."
        prompt = TABLE_EXTRACTION_PROMPT.format(query_context=context)

        raw_json = await self.engine.extract_structured_json(
            base64_image=b64_img,
            prompt=prompt,
            system_prompt=VISION_SYSTEM_PROMPT,
        )
        try:
            if not isinstance(raw_json, dict):
                raw_json = {}
            if "summary" not in raw_json:
                raw_json["summary"] = raw_json.get("error") or raw_json.get("raw_text") or "No table summary provided."
            return ExtractedTableData.model_validate(raw_json)
        except Exception as e:
            logger.warning(f"Table schema validation fallback: {e}")
            return ExtractedTableData(
                title=raw_json.get("title", "Visual Table"),
                headers=raw_json.get("headers", []),
                rows=raw_json.get("rows", []),
                summary=raw_json.get("summary") or raw_json.get("error") or str(raw_json.get("raw_text", "Could not parse valid table data.")),
            )

