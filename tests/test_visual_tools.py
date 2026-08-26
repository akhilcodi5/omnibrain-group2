"""Unit tests for Visual Tools and Memo Formatter (Task 2B)."""

import json
import pytest
from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
)
from agents.visual_analytics import VisualAnalyticsEngine
from agents.cross_modal_verifier import CrossModalVerifier
from agents.visual_tools import (
    VisualMemoFormatter,
    compute_visual_figure_trends,
    format_visual_memo_section_tool,
    verify_visual_numbers_against_text,
)


def test_visual_memo_formatter():
    """Test creating structured investment memo block with citations."""
    chart = ExtractedChartData(
        title="Revenue Trajectory",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Revenue",
                data_points=[
                    DataPoint(label="2023", value=100.0, raw_value="$100M"),
                    DataPoint(label="2024", value=130.0, raw_value="$130M"),
                ]
            )
        ],
        summary="Positive revenue trend."
    )

    trends = VisualAnalyticsEngine.analyze_chart_dataset(chart)
    verification = CrossModalVerifier.verify_chart_against_text(chart, "2023 was $100M, 2024 was $130M.")

    block = VisualMemoFormatter.format_memo_block(
        figure_id="rev_traj",
        figure_title="Revenue Trajectory",
        chart_data=chart,
        trends=trends,
        verification=verification,
        page_number=14,
    )

    assert block.figure_id == "rev_traj"
    assert block.page_number == 14
    assert "[Revenue Trajectory - Page 14]" in block.citation_tag
    assert "### 📊 Visual Analytical Evidence" in block.markdown_formatted_block
    assert "🟢 VERIFIED" in block.markdown_formatted_block


def test_langgraph_tool_compute_trends():
    """Test compute_visual_figure_trends LangGraph tool invocation."""
    chart = ExtractedChartData(
        title="Margin Expansion",
        chart_type=ChartType.LINE,
        series=[
            ChartSeries(
                series_name="Gross Margin",
                data_points=[
                    DataPoint(label="Q1", value=40.0),
                    DataPoint(label="Q2", value=45.0),
                ]
            )
        ],
        summary="Gross margin expanded by 500 bps."
    )

    json_str = chart.model_dump_json()
    tool_output = compute_visual_figure_trends.invoke({"chart_json_str": json_str})
    parsed = json.loads(tool_output)

    assert isinstance(parsed, list)
    assert parsed[0]["series_name"] == "Gross Margin"
    assert parsed[0]["total_percentage_change"] == 12.5


def test_langgraph_tool_verify_metrics():
    """Test verify_visual_numbers_against_text LangGraph tool invocation."""
    chart = ExtractedChartData(
        title="EPS Growth",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="EPS",
                data_points=[DataPoint(label="FY24", value=3.50, raw_value="$3.50")]
            )
        ],
        summary="FY24 EPS reached $3.50."
    )

    json_str = chart.model_dump_json()
    text = "Full year diluted earnings per share (EPS) was $3.50."
    
    tool_output = verify_visual_numbers_against_text.invoke({
        "chart_json_str": json_str,
        "text_context": text,
    })
    
    parsed = json.loads(tool_output)
    assert parsed["matched_metrics_count"] == 1
    assert parsed["grounding_score"] == 1.0
