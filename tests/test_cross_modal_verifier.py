"""Unit tests for CrossModalVerifier (Task 2B: Cross-Modal Verification & Discrepancy Detection)."""

import pytest
from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    DiscrepancySeverity,
    ExtractedChartData,
    VerificationStatus,
)
from agents.cross_modal_verifier import CrossModalVerifier


def test_parse_numeric_tokens():
    """Test extracting numbers, currencies, and scale tokens from text."""
    sample_text = "In Q3 2024, our revenue climbed to $148.8M, representing a 14.2% YoY increase compared to $130.3M."
    tokens = CrossModalVerifier.parse_numeric_tokens(sample_text)
    
    values = [t[0] for t in tokens]
    assert 148.8 in values
    assert 14.2 in values
    assert 130.3 in values


def test_verify_chart_against_text_exact_match():
    """Test cross-referencing where visual data points match text claims."""
    chart = ExtractedChartData(
        title="Quarterly Performance",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Revenue",
                data_points=[
                    DataPoint(label="Q1", value=120.5, raw_value="$120.5M"),
                    DataPoint(label="Q2", value=135.2, raw_value="$135.2M"),
                ]
            )
        ],
        summary="Quarterly revenue progression."
    )

    text_context = "The company recorded Q1 revenues of $120.5 million and expanded to $135.2M in the second quarter."

    report = CrossModalVerifier.verify_chart_against_text(chart, text_context, tolerance_percentage=1.0)

    assert report.total_metrics_evaluated == 2
    assert report.matched_metrics_count == 2
    assert report.discrepancy_count == 0
    assert report.grounding_score == 1.0
    assert report.cross_references[0].status == VerificationStatus.VERIFIED_MATCH


def test_verify_chart_against_text_discrepancy_detection():
    """Test detecting contradiction between visual chart and written text."""
    chart = ExtractedChartData(
        title="Quarterly Performance",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Revenue",
                data_points=[
                    DataPoint(label="Q3", value=148.8, raw_value="$148.8M"),
                ]
            )
        ],
        summary="Q3 revenue recorded at $148.8M."
    )

    # Text claims $120M instead of $148.8M (~24% variance)
    text_context = "During the third quarter, total recognized revenue stood at only $120.0 million."

    report = CrossModalVerifier.verify_chart_against_text(chart, text_context)

    assert report.discrepancy_count == 1
    assert len(report.critical_discrepancies) == 1
    assert report.cross_references[0].status == VerificationStatus.DISCREPANCY_DETECTED
    assert report.cross_references[0].severity == DiscrepancySeverity.HIGH
    assert report.grounding_score < 1.0
