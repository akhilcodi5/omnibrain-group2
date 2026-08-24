"""Unit and integration tests for Multi-Modal Vision Specialist modules."""

import pytest
from PIL import Image

from agents.vision_agent import VisionAgent, vision_node
from agents.vision_prompts import CHART_EXTRACTION_PROMPT, VISION_SYSTEM_PROMPT
from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    VisionExtractionRequest,
)
from app.services.vision_service import VisionService


def test_vision_schemas():
    """Test instantiation and validation of vision Pydantic schemas."""
    dp1 = DataPoint(label="Q1", value=100.5, raw_value="$100.5M", unit="USD Millions")
    dp2 = DataPoint(label="Q2", value=120.0, raw_value="$120.0M", unit="USD Millions")
    
    series = ChartSeries(series_name="Operating Revenue", data_points=[dp1, dp2])
    
    chart_data = ExtractedChartData(
        title="Quarterly Growth",
        chart_type=ChartType.BAR,
        x_axis_label="Quarter",
        y_axis_label="USD (M)",
        series=[series],
        summary="Positive quarterly trajectory.",
        key_insights=["19.4% QoQ growth"],
        confidence_score=0.98
    )
    
    assert chart_data.chart_type == ChartType.BAR
    assert len(chart_data.series[0].data_points) == 2
    assert chart_data.series[0].data_points[0].value == 100.5


def test_vision_service_base64_encoding():
    """Test image encoding helper with PIL image."""
    img = Image.new("RGB", (64, 64), color="blue")
    b64_str = VisionService.encode_image_to_base64(img)
    
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0


def test_prompt_formatting():
    """Test that vision prompt templates format cleanly."""
    formatted = CHART_EXTRACTION_PROMPT.format(query_context="Extract 2024 EBITDA margins")
    assert "EBITDA margins" in formatted
    assert "X-Axis" in formatted
    assert len(VISION_SYSTEM_PROMPT) > 50


@pytest.mark.asyncio
async def test_vision_agent_mock_inference():
    """Test VisionAgent offline mock reasoning when API key is unconfigured."""
    agent = VisionAgent()
    img = Image.new("RGB", (32, 32), color="red")
    
    response = await agent.analyze_visual_asset(
        image_input=img,
        query_context="Analyze revenue trends",
        expected_type=ChartType.BAR,
    )
    
    assert response is not None
    assert response.raw_markdown != ""
    assert response.chart_data is not None
    assert response.chart_data.chart_type == ChartType.BAR
    assert len(response.chart_data.series[0].data_points) == 4


@pytest.mark.asyncio
async def test_langgraph_vision_node():
    """Test the LangGraph vision node execution with dummy state."""
    state = {
        "messages": [],
        "query": "What is the Q4 revenue in the bar chart?",
        "referenced_images": ["tests/sample_chart.png"],
        "visual_evidence": [],
        "retrieved_docs": [],
        "next_agent": None,
        "sql_query": None,
        "sql_results": None,
        "is_grounded": False,
        "retrieval_confidence": 0.0,
        "iteration_count": 0,
        "final_response": None,
        "citations": [],
    }
    
    result = await vision_node(state)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["next_agent"] == "Supervisor"
    assert len(result["visual_evidence"]) == 1
