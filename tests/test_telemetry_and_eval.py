"""Unit and integration tests for TelemetryManager and MultiModalRAGEvaluator (Week 4)."""

import pytest
from app.core.telemetry import TelemetryManager, get_telemetry_manager
from eval.evaluator import MultiModalEvalReport, MultiModalRAGEvaluator


def test_telemetry_manager_in_memory_tracing():
    """Test telemetry manager trace lifecycle, token aggregation, and cost estimation."""
    telemetry = TelemetryManager()
    trace_id = telemetry.create_trace(name="test_query_trace", user_id="analyst_1")
    assert "trace_" in trace_id

    telemetry.log_agent_step(
        trace_id=trace_id,
        agent_name="SearchAgent",
        action="VectorSearch",
        prompt_tokens=100,
        completion_tokens=50,
        latency_seconds=0.25,
    )

    telemetry.log_agent_step(
        trace_id=trace_id,
        agent_name="VisualAnalyticsIntegratorAgent",
        action="ExtractChart",
        prompt_tokens=200,
        completion_tokens=100,
        latency_seconds=0.45,
    )

    telemetry.log_evaluation_score(trace_id=trace_id, metric_name="faithfulness", score=0.98)

    summary = telemetry.get_trace_summary(trace_id)
    assert summary["step_count"] == 2
    assert summary["total_tokens"] == 450
    assert summary["prompt_tokens"] == 300
    assert summary["completion_tokens"] == 150
    assert summary["total_latency_seconds"] == 0.7
    assert summary["estimated_cost_usd"] > 0.0
    assert len(summary["scores"]) == 1


def test_multimodal_rag_evaluator_faithfulness():
    """Test evaluation of claim faithfulness against evidence text."""
    evaluator = MultiModalRAGEvaluator()
    claims = [
        "In Q3 FY24, revenue was $148.8 million.",
        "Operating margin reached 27.2%.",
        "Company declared bankruptcy in 2020.",  # Unfounded claim
    ]

    evidence = "Apex reported Q3 FY24 revenue of $148.8 million and expanding operating margin to 27.2%."

    result = evaluator.evaluate_grounding_faithfulness(claims, evidence)
    assert result["total"] == 3
    assert result["grounded_count"] == 2
    assert result["score"] == 0.67


def test_full_pipeline_evaluation():
    """Test MultiModalRAGEvaluator full pipeline output scorecard."""
    evaluator = MultiModalRAGEvaluator()
    
    memo = "# Memo\n\nRevenue for Q1 was $120.5 million and operating margin stood at 22.5%."
    retrieved_docs = [{"text": "Apex recorded Q1 revenue of $120.5 million."}]
    visual_evidence = [{"analysis": "Visual bar chart confirms operating margin at 22.5%."}]
    citations = [{"page_number": 14}]

    report = evaluator.evaluate_pipeline_output(
        generated_memo=memo,
        retrieved_docs=retrieved_docs,
        visual_evidence=visual_evidence,
        citations=citations,
    )

    assert isinstance(report, MultiModalEvalReport)
    assert report.faithfulness_score == 1.0
    assert report.citation_precision == 1.0
    assert report.hallucination_index == 0.0
    assert report.passed_all_thresholds is True
