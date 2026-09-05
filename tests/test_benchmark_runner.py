"""Unit tests for BenchmarkRunner and dynamic citation overlay endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from eval.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_execution():
    """Test BenchmarkRunner scorecard generation and 100% pass rate."""
    runner = BenchmarkRunner()
    card = runner.run_all_benchmarks()

    assert card["total_test_cases"] == 2
    assert card["passed_test_cases"] == 2
    assert card["pass_rate_percentage"] == 100.0
    assert card["benchmark_status"] == "PASSED"
    assert len(card["test_case_results"]) == 2


def test_fastapi_benchmark_endpoint():
    """Test GET /api/v1/visual/benchmark endpoint."""
    client = TestClient(app)
    res = client.get("/api/v1/visual/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["benchmark_status"] == "PASSED"
    assert data["total_test_cases"] == 2


def test_fastapi_render_overlay_endpoint():
    """Test POST /api/v1/visual/render-overlay endpoint."""
    client = TestClient(app)
    payload = {
        "figure_title": "Quarterly Revenue",
        "page_number": 14,
        "bounding_box": {
            "ymin": 0.1,
            "xmin": 0.1,
            "ymax": 0.5,
            "xmax": 0.9,
            "is_normalized": True,
        },
        "status": "verified_match",
        "grounding_confidence": 0.98,
    }

    res = client.post("/api/v1/visual/render-overlay", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["figure_id"] == "quarterly_revenue"
    assert data["highlighted_page_base64"] is not None
    assert data["thumbnail_snippet_base64"] is not None
