"""Cross-Modal Verifier: Cross-references visual numbers against text context to detect discrepancies and compute grounding scores."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.models.vision_schemas import (
    CrossModalVerificationReport,
    DiscrepancySeverity,
    ExtractedChartData,
    ExtractedTableData,
    MetricCrossReference,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class CrossModalVerifier:
    """Verifies that numerical data extracted from visual figures matches textual claims in the document."""

    # Regex patterns for extracting dollar amounts, percentages, and floats with metric contexts
    NUMBER_PATTERN = re.compile(
        r"(?:\$|€|£)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:M|B|K|%|million|billion|thousand)?",
        re.IGNORECASE
    )

    @classmethod
    def parse_numeric_tokens(cls, text: str) -> List[Tuple[float, str]]:
        """Extract all candidate numerical values along with their surrounding snippet."""
        results = []
        # Find sentences or chunks
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        for sentence in sentences:
            matches = list(re.finditer(r"(?:\$|€|£)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(M|B|K|%|million|billion|thousand)?", sentence, re.IGNORECASE))
            for m in matches:
                raw_num_str = m.group(1).replace(",", "")
                scale_suffix = (m.group(2) or "").lower()
                try:
                    val = float(raw_num_str)
                    # Normalize scale if standard suffixes appear
                    if scale_suffix in ("m", "million"):
                        pass  # Keep in millions base or standard float
                    elif scale_suffix in ("b", "billion"):
                        val = val * 1000.0  # Scale to millions if comparison base is millions
                    elif scale_suffix in ("k", "thousand"):
                        val = val / 1000.0
                    results.append((val, sentence.strip()))
                except ValueError:
                    continue
        return results

    @classmethod
    def verify_chart_against_text(
        cls,
        chart_data: ExtractedChartData,
        text_context: str,
        tolerance_percentage: float = 2.0,
    ) -> CrossModalVerificationReport:
        """Cross-reference extracted chart series data points against document text context."""
        cross_refs: List[MetricCrossReference] = []
        text_numbers = cls.parse_numeric_tokens(text_context)
        
        total_metrics = 0
        matched_count = 0
        discrepancy_count = 0
        critical_issues = []

        for series in chart_data.series:
            for pt in series.data_points:
                if pt.value is None:
                    continue
                
                total_metrics += 1
                metric_name = f"{series.series_name} ({pt.label})"
                target_val = pt.value
                
                # Check if exact or close number appears in text context
                best_match = None
                min_diff = float("inf")
                matched_sentence = ""

                for num_val, sentence in text_numbers:
                    # Check relative percentage difference
                    diff_pct = abs(num_val - target_val) / max(abs(target_val), 1e-5) * 100.0
                    if diff_pct < min_diff:
                        min_diff = diff_pct
                        best_match = num_val
                        matched_sentence = sentence

                # Classify based on tolerance
                if best_match is not None and min_diff <= tolerance_percentage:
                    status = VerificationStatus.VERIFIED_MATCH
                    severity = DiscrepancySeverity.LOW
                    matched_count += 1
                    explanation = f"Exact match verified: visual ({target_val}) matches text figure ({best_match}) within {round(min_diff, 2)}%."
                elif best_match is not None and min_diff <= 15.0:
                    status = VerificationStatus.DISCREPANCY_DETECTED
                    severity = DiscrepancySeverity.MEDIUM
                    discrepancy_count += 1
                    explanation = f"Minor variance detected: visual states {target_val}, whereas text reports {best_match} ({round(min_diff, 1)}% delta)."
                    critical_issues.append(f"Metric '{metric_name}': {explanation}")
                elif best_match is not None and min_diff <= 50.0:
                    status = VerificationStatus.DISCREPANCY_DETECTED
                    severity = DiscrepancySeverity.HIGH
                    discrepancy_count += 1
                    explanation = f"HIGH DISCREPANCY: Visual shows {target_val}, but closest text figure is {best_match} ({round(min_diff, 1)}% delta)."
                    critical_issues.append(f"Metric '{metric_name}': {explanation}")
                else:
                    status = VerificationStatus.VISUAL_MISSING_IN_TEXT
                    severity = DiscrepancySeverity.LOW
                    explanation = f"Visual data point '{target_val}' was not explicitly cited in the retrieved text context."

                cross_refs.append(MetricCrossReference(
                    metric_name=metric_name,
                    visual_value=target_val,
                    text_value=best_match if status != VerificationStatus.VISUAL_MISSING_IN_TEXT else None,
                    visual_raw_string=pt.raw_value or str(target_val),
                    text_raw_string=matched_sentence if status != VerificationStatus.VISUAL_MISSING_IN_TEXT else None,
                    status=status,
                    variance_percentage=round(min_diff, 2) if best_match is not None else None,
                    severity=severity,
                    explanation=explanation,
                ))

        # Calculate grounding confidence score
        if total_metrics > 0:
            grounding_score = max(0.0, round((matched_count - (discrepancy_count * 0.5)) / total_metrics, 2))
        else:
            grounding_score = 1.0

        if discrepancy_count == 0:
            summary = f"All {matched_count} visual metrics cross-referenced were 100% consistent with the document text context."
        else:
            summary = (
                f"Cross-referencing verified {matched_count}/{total_metrics} data points, "
                f"flagging {discrepancy_count} potential discrepancies for supervisor review."
            )

        return CrossModalVerificationReport(
            total_metrics_evaluated=total_metrics,
            matched_metrics_count=matched_count,
            discrepancy_count=discrepancy_count,
            grounding_score=grounding_score,
            cross_references=cross_refs,
            critical_discrepancies=critical_issues,
            summary_assessment=summary,
        )
