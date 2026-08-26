"""Visual Analytics Engine: Downstream quantitative reasoning, trend analysis, and anomaly detection over extracted visual figures."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    ExtractedTableData,
    TrendDirection,
    VisualTrendAnalysis,
)

logger = logging.getLogger(__name__)


class VisualAnalyticsEngine:
    """Performs mathematical and quantitative analysis on extracted visual chart and table data."""

    @staticmethod
    def calculate_cagr(start_val: float, end_val: float, periods: int) -> Optional[float]:
        """Calculate Compound Annual Growth Rate (CAGR)."""
        if periods <= 0 or start_val <= 0 or end_val <= 0:
            return None
        try:
            cagr = ((end_val / start_val) ** (1.0 / periods)) - 1.0
            return round(cagr * 100.0, 2)
        except Exception:
            return None

    @staticmethod
    def calculate_percentage_change(start_val: float, end_val: float) -> Optional[float]:
        """Calculate percentage change between two figures."""
        if start_val == 0:
            return None
        change = ((end_val - start_val) / abs(start_val)) * 100.0
        return round(change, 2)

    @classmethod
    def analyze_series_trend(cls, series: ChartSeries) -> VisualTrendAnalysis:
        """Perform comprehensive quantitative trend analysis on a single data series."""
        points = [p for p in series.data_points if p.value is not None]
        
        if not points:
            return VisualTrendAnalysis(
                series_name=series.series_name,
                trend_direction=TrendDirection.STABLE,
                analytical_takeaway="Insufficient numeric data points to establish quantitative trend."
            )

        values = [p.value for p in points]  # type: ignore
        first_pt = points[0]
        last_pt = points[-1]
        
        min_pt = min(points, key=lambda p: p.value or 0.0)
        max_pt = max(points, key=lambda p: p.value or 0.0)

        # 1. Absolute & Total Percentage Delta
        abs_delta = round(last_pt.value - first_pt.value, 2)  # type: ignore
        pct_change = cls.calculate_percentage_change(first_pt.value, last_pt.value)  # type: ignore

        # 2. Sequential Period-Over-Period Growth Rates
        pop_growths = []
        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            if prev.value is not None and curr.value is not None:
                step_pct = cls.calculate_percentage_change(prev.value, curr.value)
                pop_growths.append({
                    "from_period": prev.label,
                    "to_period": curr.label,
                    "from_value": prev.value,
                    "to_value": curr.value,
                    "growth_percentage": step_pct,
                })

        # 3. CAGR Calculation (if >= 3 points)
        cagr = None
        if len(points) >= 3 and first_pt.value > 0 and last_pt.value > 0:
            cagr = cls.calculate_cagr(first_pt.value, last_pt.value, len(points) - 1)

        # 4. Anomaly & Outlier Detection (Z-Score & Large Swings)
        anomalies = []
        if len(values) >= 3:
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_dev = math.sqrt(variance)

            for pt in points:
                if std_dev > 0 and pt.value is not None:
                    z_score = abs(pt.value - mean_val) / std_dev
                    if z_score >= 2.0:
                        anomalies.append(
                            f"Statistical outlier in {pt.label}: value {pt.raw_value or pt.value} is {round(z_score, 1)}σ from the mean."
                        )

        # Check for sharp inflection drops (> 25% single-period decline)
        for step in pop_growths:
            g = step.get("growth_percentage")
            if g is not None and g <= -25.0:
                anomalies.append(
                    f"Sharp contraction between {step['from_period']} and {step['to_period']}: {g}% decline."
                )

        # 5. Determine Trend Direction & Volatility
        direction = TrendDirection.STABLE
        if pct_change is not None:
            if pct_change > 5.0:
                direction = TrendDirection.UPWARD
            elif pct_change < -5.0:
                direction = TrendDirection.DOWNWARD

        # Check for high volatility (direction reversals)
        if len(pop_growths) >= 3:
            signs = [1 if (s.get("growth_percentage") or 0) > 0 else -1 for s in pop_growths]
            sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
            if sign_changes >= 2 and max(values) - min(values) > (sum(values)/len(values) * 0.3):
                direction = TrendDirection.VOLATILE

        # 6. Generate Analytical Narrative Takeaway
        unit_str = f" {first_pt.unit}" if first_pt.unit else ""
        cagr_text = f" at a CAGR of {cagr}%" if cagr else ""
        takeaway = (
            f"Series '{series.series_name}' demonstrated an overall {direction.value} trajectory, "
            f"moving from {first_pt.raw_value or first_pt.value} in {first_pt.label} to {last_pt.raw_value or last_pt.value} in {last_pt.label}"
            f"{unit_str} ({'+' if pct_change and pct_change > 0 else ''}{pct_change}% total change{cagr_text})."
        )

        return VisualTrendAnalysis(
            series_name=series.series_name,
            trend_direction=direction,
            cagr_percentage=cagr,
            total_percentage_change=pct_change,
            absolute_delta=abs_delta,
            min_point=min_pt,
            max_point=max_pt,
            period_over_period_growths=pop_growths,
            detected_anomalies=anomalies,
            analytical_takeaway=takeaway,
        )

    @classmethod
    def analyze_chart_dataset(cls, chart_data: ExtractedChartData) -> List[VisualTrendAnalysis]:
        """Analyze all series within an extracted chart payload."""
        results = []
        for series in chart_data.series:
            trend = cls.analyze_series_trend(series)
            results.append(trend)
        return results
