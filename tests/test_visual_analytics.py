"""Unit tests for VisualAnalyticsEngine (Task 2B: Visual Analytics & Multi-Modal Tool Integrator)."""

import pytest
from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    TrendDirection,
)
from agents.visual_analytics import VisualAnalyticsEngine


def test_calculate_cagr():
    """Test Compound Annual Growth Rate computation."""
    # $100M growing to $144M over 2 years = 20% CAGR
    cagr = VisualAnalyticsEngine.calculate_cagr(start_val=100.0, end_val=144.0, periods=2)
    assert cagr == 20.0

    # Invalid cases
    assert VisualAnalyticsEngine.calculate_cagr(start_val=0, end_val=100, periods=2) is None
    assert VisualAnalyticsEngine.calculate_cagr(start_val=100, end_val=200, periods=0) is None


def test_series_trend_upward_growth():
    """Test upward trend classification and period-over-period percentage computation."""
    series = ChartSeries(
        series_name="Total Revenue",
        data_points=[
            DataPoint(label="Q1 FY24", value=100.0, raw_value="$100M", unit="USD Millions"),
            DataPoint(label="Q2 FY24", value=115.0, raw_value="$115M", unit="USD Millions"),
            DataPoint(label="Q3 FY24", value=130.0, raw_value="$130M", unit="USD Millions"),
            DataPoint(label="Q4 FY24", value=150.0, raw_value="$150M", unit="USD Millions"),
        ]
    )

    trend = VisualAnalyticsEngine.analyze_series_trend(series)

    assert trend.trend_direction == TrendDirection.UPWARD
    assert trend.total_percentage_change == 50.0
    assert trend.absolute_delta == 50.0
    assert len(trend.period_over_period_growths) == 3
    assert trend.period_over_period_growths[0]["growth_percentage"] == 15.0
    assert trend.min_point.value == 100.0
    assert trend.max_point.value == 150.0
    assert "CAGR" in trend.analytical_takeaway


def test_anomaly_detection_sharp_drop():
    """Test anomaly detection flagging sharp contraction (>25% drop)."""
    series = ChartSeries(
        series_name="Net Operating Income",
        data_points=[
            DataPoint(label="Q1", value=100.0),
            DataPoint(label="Q2", value=105.0),
            DataPoint(label="Q3", value=60.0),  # > 40% drop
            DataPoint(label="Q4", value=65.0),
        ]
    )

    trend = VisualAnalyticsEngine.analyze_series_trend(series)

    assert len(trend.detected_anomalies) > 0
    assert any("contraction" in a.lower() or "decline" in a.lower() for a in trend.detected_anomalies)


def test_analyze_chart_dataset():
    """Test full chart dataset multi-series analytics."""
    chart = ExtractedChartData(
        title="Revenue vs Operating Expenses",
        chart_type=ChartType.LINE,
        series=[
            ChartSeries(
                series_name="Revenue",
                data_points=[DataPoint(label="2022", value=500.0), DataPoint(label="2023", value=600.0)]
            ),
            ChartSeries(
                series_name="OpEx",
                data_points=[DataPoint(label="2022", value=400.0), DataPoint(label="2023", value=420.0)]
            )
        ],
        summary="Revenue outpaced OpEx growth in 2023."
    )

    trends = VisualAnalyticsEngine.analyze_chart_dataset(chart)

    assert len(trends) == 2
    assert trends[0].series_name == "Revenue"
    assert trends[0].total_percentage_change == 20.0
    assert trends[1].series_name == "OpEx"
    assert trends[1].total_percentage_change == 5.0
