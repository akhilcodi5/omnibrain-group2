"""Unit tests for Day 5 Multi-Modal Synthesizer and Query Intent Evaluator (Task 2B)."""

import pytest
from agents.memo_synthesizer import MemoSynthesizer
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_routing_evaluator import VisualIntentType, VisualRoutingEvaluator
from agents.visual_tools import VisualMemoFormatter
from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
)


def test_visual_routing_evaluator():
    """Test classification of user queries into appropriate multi-modal routing intents."""
    # 1. Verification query
    q1 = "Cross reference the revenue claim in the report against the visual quarterly chart on page 14."
    score1 = VisualRoutingEvaluator.evaluate_query(q1)
    assert score1.primary_intent == VisualIntentType.CROSS_MODAL_VERIFY
    assert score1.requires_cross_modal_check is True
    assert score1.requires_visual_agent is True

    # 2. Multi-figure comparison query
    q2 = "Compare the regional revenue breakdown chart against the product line segment figure."
    score2 = VisualRoutingEvaluator.evaluate_query(q2)
    assert score2.primary_intent == VisualIntentType.MULTI_FIGURE_COMPARE
    assert score2.requires_multi_figure_compare is True

    # 3. Investment memo query
    q3 = "Generate an executive investment memo synthesized from all quarterly financial charts."
    score3 = VisualRoutingEvaluator.evaluate_query(q3)
    assert score3.primary_intent == VisualIntentType.SYNTHESIZE_MEMO

    # 4. Direct visual extraction
    q4 = "What is the EBITDA margin shown in the line chart?"
    score4 = VisualRoutingEvaluator.evaluate_query(q4)
    assert score4.primary_intent == VisualIntentType.VISUAL_EXTRACTION


def test_memo_synthesizer_output():
    """Test full multi-modal investment memo synthesis."""
    chart = ExtractedChartData(
        title="Revenue Growth",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Revenue",
                data_points=[
                    DataPoint(label="Q1", value=100.0, raw_value="$100M"),
                    DataPoint(label="Q2", value=120.0, raw_value="$120M"),
                ]
            )
        ],
        summary="Consistent 20% quarterly revenue expansion."
    )

    trends = VisualAnalyticsEngine.analyze_chart_dataset(chart)
    memo_block = VisualMemoFormatter.format_memo_block(
        figure_id="rev_growth",
        figure_title="Revenue Growth",
        chart_data=chart,
        trends=trends,
        page_number=14,
    )

    memo_md = MemoSynthesizer.synthesize_investment_memo(
        company_name="Apex Holdings Corp",
        analyst_query="Analyze quarterly revenue trends and margins.",
        visual_blocks=[memo_block],
        sql_metrics={"Ticker": "APEX", "Stock Price": "$184.50"}
    )

    assert "# 📑 Investment Research Memorandum: Apex Holdings Corp" in memo_md
    assert "## 1. Executive Summary & Quantitative Findings" in memo_md
    assert "## 2. Multi-Modal Visual Analytics & Trend Analysis" in memo_md
    assert "## 3. Cross-Modal Grounding & Verification Audit" in memo_md
    assert "[Revenue Growth - Page 14]" in memo_md
    assert "Ticker" in memo_md
