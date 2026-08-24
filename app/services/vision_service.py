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

logger = logging.getLogger(__name__)


class VisionService:
    """Service to handle Vision-Language Model inference for charts, tables, and visual figures."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")

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
            # Convert RGBA to RGB if needed before saving as JPEG
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

    def build_multimodal_message(
        self, 
        base64_image: str, 
        prompt_text: str, 
        media_type: str = "image/jpeg"
    ) -> list:
        """Build standard multimodal message payload for OpenAI API / LangChain."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

    async def analyze_figure(
        self, 
        request: VisionExtractionRequest
    ) -> VisionExtractionResponse:
        """Perform multimodal extraction and reasoning on a given figure."""
        try:
            # 1. Resolve image to base64
            if request.image_base64:
                b64_img = request.image_base64
            elif request.image_path:
                b64_img = self.encode_image_to_base64(request.image_path)
            else:
                raise ValueError("No valid image input (base64 or path) provided in request.")

            media_type = self.get_image_media_type(request.image_path)
            context = request.query_context or "General quantitative extraction and financial analysis."

            # 2. Select prompt template based on expected type
            if request.expected_type == ChartType.TABLE:
                prompt_text = TABLE_EXTRACTION_PROMPT.format(query_context=context)
            elif request.expected_type in (ChartType.BAR, ChartType.LINE, ChartType.PIE, ChartType.AREA):
                prompt_text = CHART_EXTRACTION_PROMPT.format(query_context=context)
            else:
                prompt_text = FIGURE_REASONING_PROMPT.format(query_context=context)

            # 3. Call VLM if API key is present, otherwise provide fallback mock for offline dev
            if not self.api_key or self.api_key.startswith("your_"):
                logger.warning("OPENAI_API_KEY not configured. Returning structured placeholder result.")
                return self._create_mock_response(request, context)

            # Execute real VLM inference with OpenAI client
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)

            messages = [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                *self.build_multimodal_message(b64_img, prompt_text, media_type),
            ]

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
            )

            raw_content = response.choices[0].message.content or ""

            return VisionExtractionResponse(
                image_id=os.path.basename(request.image_path) if request.image_path else "memory_figure",
                raw_markdown=raw_content,
                metadata={
                    "model": self.model_name,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
            )

        except Exception as e:
            logger.error(f"Error during vision extraction: {str(e)}", exc_info=True)
            return VisionExtractionResponse(
                raw_markdown=f"### Vision Extraction Error\n\nFailed to process image: {str(e)}",
                metadata={"error": str(e)}
            )

    def _create_mock_response(
        self, 
        request: VisionExtractionRequest, 
        context: str
    ) -> VisionExtractionResponse:
        """Create structured placeholder response when running locally without API keys."""
        mock_chart = ExtractedChartData(
            title="Quarterly Revenue & Margin Breakdown",
            chart_type=ChartType.BAR,
            x_axis_label="Fiscal Quarter",
            y_axis_label="Revenue (USD Millions)",
            series=[
                ChartSeries(
                    series_name="Total Revenue",
                    data_points=[
                        DataPoint(label="Q1 FY24", value=120.5, raw_value="$120.5M", unit="USD Millions"),
                        DataPoint(label="Q2 FY24", value=135.2, raw_value="$135.2M", unit="USD Millions"),
                        DataPoint(label="Q3 FY24", value=148.8, raw_value="$148.8M", unit="USD Millions"),
                        DataPoint(label="Q4 FY24", value=162.0, raw_value="$162.0M", unit="USD Millions"),
                    ]
                )
            ],
            summary="Revenue demonstrated consistent sequential quarter-over-quarter growth throughout FY24.",
            key_insights=[
                "FY24 total revenue reached $566.5M, a +21% YoY increase.",
                "Q4 was the strongest quarter with $162.0M in revenue."
            ],
            notable_anomalies=[],
            confidence_score=0.95
        )

        return VisionExtractionResponse(
            image_id=os.path.basename(request.image_path) if request.image_path else "sample_figure",
            chart_data=mock_chart,
            raw_markdown=f"### Visual Analysis: Quarterly Revenue Trend\n\n- **Context**: {context}\n- **Summary**: Total Revenue expanded from $120.5M in Q1 to $162.0M in Q4 FY24.\n- **Trend**: Positive linear trajectory with strong Q4 acceleration.",
            metadata={"status": "mock_mode", "note": "Set OPENAI_API_KEY for live VLM responses"}
        )
