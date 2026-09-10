"""Integration and Unit Tests for Day 4 Visual Analytics & Tool Integration (Task 2B)."""

import json
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from agents.multi_modal_bridge import (
    compare_two_visual_figures,
    generate_visual_citation_overlay_tool,
    get_visual_analytics_tools,
)
from agents.visual_comparator import VisualComparator
from app.main import app
from app.models.vision_schemas import (
    BoundingBox,
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    VerificationStatus,
)
from app.services.citation_renderer import CitationRenderer


def test_visual_comparator_two_charts():
    """Test comparative variance analysis between two financial charts."""
    chart_a = ExtractedChartData(
        title="2023 Segment Revenue",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Segment Revenue",
                data_points=[
                    DataPoint(label="North America", value=100.0),
                    DataPoint(label="Europe", value=80.0),
                ]
            )
        ],
        summary="2023 Segment breakdown."
    )

    chart_b = ExtractedChartData(
        title="2024 Segment Revenue",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Segment Revenue",
                data_points=[
                    DataPoint(label="North America", value=125.0), # +25%
                    DataPoint(label="Europe", value=76.0),         # -5%
                ]
            )
        ],
        summary="2024 Segment breakdown."
    )

    report = VisualComparator.compare_two_charts(chart_a, chart_b)

    assert report.primary_figure_title == "2023 Segment Revenue"
    assert report.secondary_figure_title == "2024 Segment Revenue"
    assert len(report.metric_deltas) == 2

    na_delta = next(d for d in report.metric_deltas if "North America" in d.label)
    assert na_delta.variance_percentage == 25.0
    assert na_delta.variance_absolute == 25.0


def test_citation_renderer_overlay():
    """Test generating bounding box highlight overlays and thumbnail snippets."""
    canvas = Image.new("RGB", (600, 800), color="white")
    box = BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.9, is_normalized=True)

    payload = CitationRenderer.render_citation_overlay(
        image=canvas,
        bounding_box=box,
        figure_title="Operating Revenue Chart",
        page_number=12,
        status=VerificationStatus.VERIFIED_MATCH,
        grounding_confidence=0.95,
    )

    assert payload.figure_id == "operating_revenue_chart"
    assert payload.page_number == 12
    assert payload.status == VerificationStatus.VERIFIED_MATCH
    assert payload.highlighted_page_base64 is not None
    assert payload.thumbnail_snippet_base64 is not None
    assert "[Operating Revenue Chart - Page 12]" in payload.citation_tag


def test_multi_modal_bridge_tools():
    """Test tool registry count and tool execution."""
    tools = get_visual_analytics_tools()
    assert len(tools) == 5

    tool_names = [t.name for t in tools]
    assert "verify_visual_numbers_against_text" in tool_names
    assert "compute_visual_figure_trends" in tool_names
    assert "compare_two_visual_figures" in tool_names
    assert "generate_visual_citation_overlay_tool" in tool_names


def test_fastapi_visual_routes():
    """Test FastAPI visual analytics endpoints with TestClient."""
    client = TestClient(app)

    # 1. Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Visual Analytics" in res_root.json().get("role_2b", "")
    assert res_root.json().get("workspace") == "/workspace"

    # Test Workspace UI Serving
    res_ws = client.get("/workspace")
    assert res_ws.status_code == 200
    assert "text/html" in res_ws.headers.get("content-type", "")
    assert "OmniBrain Quant Workspace" in res_ws.text

    # 2. Test Tools List Endpoint
    res_tools = client.get("/api/v1/visual/tools")
    assert res_tools.status_code == 200
    assert len(res_tools.json()) == 5

    # 3. Test Compute Trends Endpoint
    chart_payload = {
        "chart_data": {
            "title": "Margin Growth",
            "chart_type": "line",
            "series": [
                {
                    "series_name": "EBITDA Margin",
                    "data_points": [
                        {"label": "Q1", "value": 20.0},
                        {"label": "Q2", "value": 24.0},
                    ]
                }
            ],
            "summary": "Margin expanded 400 bps."
        }
    }
    res_trends = client.post("/api/v1/visual/compute-trends", json=chart_payload)
    assert res_trends.status_code == 200
    trends_json = res_trends.json()
    assert len(trends_json) == 1
    assert trends_json[0]["total_percentage_change"] == 20.0

    # 4. Test Extract Chart Endpoint
    res_extract_chart = client.post(
        "/api/v1/visual/extract-chart",
        json={"image_base64": "dummy_b64_string", "query_context": "Operating Margin"},
    )
    assert res_extract_chart.status_code == 200
    assert res_extract_chart.json()["chart_type"] == "bar"
    assert len(res_extract_chart.json()["series"][0]["data_points"]) == 4

    # 5. Test Extract Table Endpoint
    res_extract_table = client.post(
        "/api/v1/visual/extract-table",
        json={"image_base64": "dummy_b64_string", "query_context": "Financial Income Statement Table"},
    )
    assert res_extract_table.status_code == 200
    assert res_extract_table.json()["title"] == "Consolidated Statement of Income"
    assert len(res_extract_table.json()["headers"]) == 5

    # 6. Test Save & List Visual Assets Endpoint
    res_save_asset = client.post(
        "/api/v1/visual/assets",
        json={
            "image_base64": "dummy_b64_string",
            "doc_id": "doc_test_101",
            "page_number": 3,
            "figure_name": "Revenue Bar Chart",
            "figure_type": "bar",
        },
    )
    assert res_save_asset.status_code == 200
    asset_meta = res_save_asset.json()
    assert asset_meta["doc_id"] == "doc_test_101"

    res_list_assets = client.get("/api/v1/visual/assets/doc_test_101")
    assert res_list_assets.status_code == 200
    assert len(res_list_assets.json()) >= 1
