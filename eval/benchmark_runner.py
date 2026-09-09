"""Multi-Modal Benchmark Runner: Executes standardized test cases over visual figures, verification loops, and agent pipelines."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is in Python path for standalone script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from agents.cross_modal_verifier import CrossModalVerifier
from agents.memo_synthesizer import MemoSynthesizer
from agents.supervisor import create_supervisor_graph
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_comparator import VisualComparator
from app.models.vision_schemas import (
    BoundingBox,
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    VerificationStatus,
)
from eval.evaluator import MultiModalRAGEvaluator
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs automated benchmarks evaluating OmniBrain's multi-modal accuracy, grounding, and latency."""

    @staticmethod
    def get_benchmark_fixtures() -> List[Dict[str, Any]]:
        """Return standardized test cases for benchmark evaluation."""
        return [
            {
                "test_name": "Quarterly Revenue Bar Chart (Clean Grounding)",
                "chart": ExtractedChartData(
                    title="FY24 Quarterly Revenue",
                    chart_type=ChartType.BAR,
                    series=[
                        ChartSeries(
                            series_name="Revenue",
                            data_points=[
                                DataPoint(label="Q1", value=120.5, raw_value="$120.5M"),
                                DataPoint(label="Q2", value=135.2, raw_value="$135.2M"),
                                DataPoint(label="Q3", value=148.8, raw_value="$148.8M"),
                                DataPoint(label="Q4", value=162.0, raw_value="$162.0M"),
                            ]
                        )
                    ],
                    summary="Revenue expanded consistently across FY24.",
                ),
                "text_context": "Apex recorded Q1 revenue of $120.5M, Q2 of $135.2M, Q3 of $148.8M, and Q4 of $162.0M.",
                "expected_grounding": 1.0,
                "expected_trend": "upward",
                "expected_cagr": 10.4,
            },
            {
                "test_name": "Operating Margin Expansion with Discrepancy Detection",
                "chart": ExtractedChartData(
                    title="Operating Margins",
                    chart_type=ChartType.LINE,
                    series=[
                        ChartSeries(
                            series_name="Margin",
                            data_points=[
                                DataPoint(label="2022", value=20.0, raw_value="20.0%"),
                                DataPoint(label="2023", value=25.0, raw_value="25.0%"),
                            ]
                        )
                    ],
                    summary="Margin expanded 500 bps.",
                ),
                "text_context": "Operating margin in 2022 was 20.0%, but dropped to 15.0% in 2023.",  # Deliberate contradiction
                "expected_grounding": 0.25,
                "expected_discrepancy_count": 1,
            }
        ]

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Execute all benchmark test cases and return a unified scorecard."""
        fixtures = self.get_benchmark_fixtures()
        evaluator = MultiModalRAGEvaluator()
        
        results = []
        start_time = time.time()
        passed_count = 0

        for case in fixtures:
            case_start = time.time()
            chart: ExtractedChartData = case["chart"]
            text_context: str = case["text_context"]

            # 1. Trend Analytics
            trends = VisualAnalyticsEngine.analyze_chart_dataset(chart)
            
            # 2. Cross-Modal Verification
            verification = CrossModalVerifier.verify_chart_against_text(chart, text_context)

            # 3. Assertions
            passed = True
            reasons = []

            if "expected_grounding" in case:
                if abs(verification.grounding_score - case["expected_grounding"]) > 0.05:
                    passed = False
                    reasons.append(f"Grounding score mismatch: got {verification.grounding_score}, expected {case['expected_grounding']}")

            if "expected_discrepancy_count" in case:
                if verification.discrepancy_count != case["expected_discrepancy_count"]:
                    passed = False
                    reasons.append(f"Discrepancy count mismatch: got {verification.discrepancy_count}, expected {case['expected_discrepancy_count']}")

            if passed:
                passed_count += 1

            case_elapsed = round(time.time() - case_start, 3)

            results.append({
                "test_name": case["test_name"],
                "passed": passed,
                "grounding_score": verification.grounding_score,
                "discrepancies_flagged": verification.discrepancy_count,
                "derived_trends": [t.trend_direction.value for t in trends],
                "latency_seconds": case_elapsed,
                "failure_reasons": reasons,
            })

        total_elapsed = round(time.time() - start_time, 3)
        pass_rate = round((passed_count / len(fixtures)) * 100.0, 1)

        scorecard = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(fixtures),
            "passed_test_cases": passed_count,
            "pass_rate_percentage": pass_rate,
            "total_benchmark_duration_seconds": total_elapsed,
            "benchmark_status": "PASSED" if pass_rate == 100.0 else "REVIEW_REQUIRED",
            "test_case_results": results,
        }

        return scorecard


if __name__ == "__main__":
    runner = BenchmarkRunner()
    card = runner.run_all_benchmarks()
    print(json.dumps(card, indent=2))
