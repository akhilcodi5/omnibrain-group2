"""Multi-Figure Comparative Analytics Engine (Task 2B: Visual Analytics & Multi-Modal Tool Integrator).

Compares, contrasts, and cross-analyzes multiple extracted visual assets (e.g., comparing
Segment Revenue vs. Consolidated Revenue, or Projected vs. Actual Quarterly Performance).
"""

import logging
from typing import Any, Dict, List, Optional

from app.models.vision_schemas import (
    ComparisonMetricDelta,
    ExtractedChartData,
    ExtractedTableData,
    MultiFigureComparisonReport,
)

logger = logging.getLogger(__name__)


class VisualComparator:
    """Performs comparative quantitative analysis between multiple extracted charts or tables."""

    @staticmethod
    def compare_two_charts(
        chart_a: ExtractedChartData,
        chart_b: ExtractedChartData,
        comparison_type: str = "Cross-Figure Performance Variance",
    ) -> MultiFigureComparisonReport:
        """Compare data points between two charts by matching category labels / time periods."""
        deltas: List[ComparisonMetricDelta] = []
        strategic_insights: List[str] = []

        title_a = chart_a.title or "Figure A"
        title_b = chart_b.title or "Figure B"

        # Build lookup table for Chart A data points (series 0 default)
        points_a: Dict[str, float] = {}
        if chart_a.series:
            for s in chart_a.series:
                for pt in s.data_points:
                    if pt.value is not None:
                        key = f"{s.series_name}: {pt.label}" if len(chart_a.series) > 1 else pt.label
                        points_a[key] = pt.value

        # Build lookup table for Chart B data points
        points_b: Dict[str, float] = {}
        if chart_b.series:
            for s in chart_b.series:
                for pt in s.data_points:
                    if pt.value is not None:
                        key = f"{s.series_name}: {pt.label}" if len(chart_b.series) > 1 else pt.label
                        points_b[key] = pt.value

        # Cross-match common labels
        all_keys = sorted(list(set(points_a.keys()).union(set(points_b.keys()))))
        divergence_flags = 0

        for key in all_keys:
            val_a = points_a.get(key)
            val_b = points_b.get(key)

            if val_a is not None and val_b is not None:
                abs_var = round(val_b - val_a, 2)
                pct_var = round(((val_b - val_a) / abs(val_a)) * 100.0, 2) if val_a != 0 else 0.0

                if abs(pct_var) > 15.0:
                    divergence_flags += 1
                    takeaway = f"Significant variance: '{title_b}' diverges from '{title_a}' by {pct_var}% ({abs_var:+})."
                elif abs(pct_var) == 0.0:
                    takeaway = "Complete consistency across both figures."
                else:
                    takeaway = f"Moderate variance of {pct_var}% ({abs_var:+})."

                deltas.append(ComparisonMetricDelta(
                    label=key,
                    figure_a_value=val_a,
                    figure_b_value=val_b,
                    variance_absolute=abs_var,
                    variance_percentage=pct_var,
                    takeaway=takeaway,
                ))
            elif val_a is not None:
                deltas.append(ComparisonMetricDelta(
                    label=key,
                    figure_a_value=val_a,
                    takeaway=f"Present in '{title_a}' ({val_a}) but absent in '{title_b}'.",
                ))
            elif val_b is not None:
                deltas.append(ComparisonMetricDelta(
                    label=key,
                    figure_b_value=val_b,
                    takeaway=f"Present in '{title_b}' ({val_b}) but absent in '{title_a}'.",
                ))

        # Generate Strategic Synthesis
        if divergence_flags > 0:
            divergence_summary = (
                f"Comparative analysis between '{title_a}' and '{title_b}' revealed {divergence_flags} "
                f"metrics with significant divergence (>15% variance), suggesting shift in reporting baseline or operational change."
            )
            strategic_insights.append(
                f"Flagged {divergence_flags} diverged periods requiring management discussion review."
            )
        else:
            divergence_summary = (
                f"Comparison between '{title_a}' and '{title_b}' demonstrated high directional alignment and baseline consistency."
            )

        return MultiFigureComparisonReport(
            primary_figure_title=title_a,
            secondary_figure_title=title_b,
            comparison_type=comparison_type,
            metric_deltas=deltas,
            divergence_summary=divergence_summary,
            strategic_insights=strategic_insights,
        )
