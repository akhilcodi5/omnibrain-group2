"""Pydantic schemas for Visual Analytics, Cross-Modal Verification, and Tool Integration."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# =====================================================================
# 1. Base Visual Data Models (Inputs from Upstream Extraction)
# =====================================================================

class ChartType(str, Enum):
    """Enumeration of supported visual figure and chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    CANDLESTICK = "candlestick"
    TABLE = "table"
    DIAGRAM = "diagram"
    INFOGRAPHIC = "infographic"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Normalized or pixel bounding box coordinates for visual regions in documents."""
    ymin: float = Field(..., description="Top coordinate (0.0 - 1.0 or pixel integer)")
    xmin: float = Field(..., description="Left coordinate (0.0 - 1.0 or pixel integer)")
    ymax: float = Field(..., description="Bottom coordinate (0.0 - 1.0 or pixel integer)")
    xmax: float = Field(..., description="Right coordinate (0.0 - 1.0 or pixel integer)")
    is_normalized: bool = Field(default=True, description="Whether coordinates are normalized between 0.0 and 1.0")

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return box as (xmin, ymin, xmax, ymax)."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class DataPoint(BaseModel):
    """Individual data point extracted from a chart or figure."""
    label: str = Field(..., description="Category label or timestamp on the X-axis/category axis")
    value: Optional[float] = Field(None, description="Numerical value parsed from the chart/axis")
    raw_value: Optional[str] = Field(None, description="Original raw text representation (e.g. '$12.5M', '45%')")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g. 'USD Millions', '%', 'Units')")


class ChartSeries(BaseModel):
    """A data series within a multi-series chart."""
    series_name: str = Field(default="Default", description="Name/legend label for the data series")
    data_points: List[DataPoint] = Field(default_factory=list, description="List of extracted data points")


class ExtractedChartData(BaseModel):
    """Structured data payload extracted from an embedded financial chart or diagram."""
    title: Optional[str] = Field(None, description="Chart title or header text")
    chart_type: ChartType = Field(default=ChartType.UNKNOWN, description="Identified type of visual figure")
    x_axis_label: Optional[str] = Field(None, description="Label for the horizontal axis")
    y_axis_label: Optional[str] = Field(None, description="Label for the vertical axis")
    series: List[ChartSeries] = Field(default_factory=list, description="All data series extracted from the figure")
    summary: str = Field(..., description="Executive summary and visual description of the chart")
    key_insights: List[str] = Field(default_factory=list, description="Key qualitative and quantitative takeaways")
    notable_anomalies: List[str] = Field(default_factory=list, description="Outliers, trend reversals, or unusual spikes")
    confidence_score: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0.0 to 1.0) of the visual extraction"
    )
    bounding_box: Optional[BoundingBox] = Field(None, description="Coordinates of chart on source page")


class ExtractedTableData(BaseModel):
    """Structured data payload extracted from an embedded financial table image."""
    title: Optional[str] = Field(None, description="Table title or caption")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[str]] = Field(default_factory=list, description="Table rows as string matrix")
    summary: str = Field(..., description="Narrative summary of tabular data")
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Key financial metrics parsed as key-value pairs (e.g. {'Revenue': '$5.2B'})"
    )
    currency: Optional[str] = Field(None, description="Currency symbol or code (e.g. 'USD', 'EUR')")
    scale: Optional[str] = Field(None, description="Scale of figures (e.g. 'Thousands', 'Millions', 'Billions')")
    bounding_box: Optional[BoundingBox] = Field(None, description="Coordinates of table on source page")


# =====================================================================
# 2. Downstream Visual Analytics Models (Task 2B)
# =====================================================================

class TrendDirection(str, Enum):
    """Directional trend categorization."""
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    VOLATILE = "volatile"


class VisualTrendAnalysis(BaseModel):
    """Analytical metrics and quantitative derivations computed from visual figures."""
    series_name: str = Field(..., description="Name of data series analyzed")
    trend_direction: TrendDirection = Field(..., description="Overall trend trajectory")
    cagr_percentage: Optional[float] = Field(None, description="Compound Annual Growth Rate if applicable")
    total_percentage_change: Optional[float] = Field(None, description="Percentage change from start to end of series")
    absolute_delta: Optional[float] = Field(None, description="Absolute difference between final and initial values")
    min_point: Optional[DataPoint] = Field(None, description="Minimum observed data point")
    max_point: Optional[DataPoint] = Field(None, description="Maximum observed data point")
    period_over_period_growths: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Sequential period-to-period growth rates (YoY / QoQ)"
    )
    detected_anomalies: List[str] = Field(default_factory=list, description="Statistically flagged unusual spikes or drops")
    analytical_takeaway: str = Field(..., description="Financial interpretation of the quantitative trend")


# =====================================================================
# 3. Cross-Modal Verification & Grounding Models (Task 2B)
# =====================================================================

class VerificationStatus(str, Enum):
    """Status of cross-modal cross-referencing between visual numbers and text context."""
    VERIFIED_MATCH = "verified_match"
    DISCREPANCY_DETECTED = "discrepancy_detected"
    TEXT_MISSING_IN_VISUAL = "text_missing_in_visual"
    VISUAL_MISSING_IN_TEXT = "visual_missing_in_text"
    UNVERIFIABLE = "unverifiable"


class DiscrepancySeverity(str, Enum):
    """Severity of variance between visual numbers and text claims."""
    LOW = "low"         # Rounding or minor scale difference (< 2%)
    MEDIUM = "medium"   # Noticeable discrepancy (2% - 10%)
    HIGH = "high"       # Severe contradiction or reporting error (> 10%)


class MetricCrossReference(BaseModel):
    """Cross-reference record comparing a visual metric to textual mentions."""
    metric_name: str = Field(..., description="Name of the financial metric (e.g. 'Operating Revenue Q3')")
    visual_value: Optional[float] = Field(None, description="Numeric value extracted from chart/table")
    text_value: Optional[float] = Field(None, description="Numeric value reported in text context")
    visual_raw_string: Optional[str] = Field(None, description="Raw visual display string (e.g. '$148.8M')")
    text_raw_string: Optional[str] = Field(None, description="Raw text sentence snippet")
    status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIABLE)
    variance_percentage: Optional[float] = Field(None, description="Calculated percentage delta between sources")
    severity: DiscrepancySeverity = Field(default=DiscrepancySeverity.LOW)
    explanation: str = Field(..., description="Verification rationale or discrepancy analysis")


class CrossModalVerificationReport(BaseModel):
    """Comprehensive verification report synthesized for the Supervisor Agent."""
    total_metrics_evaluated: int = Field(default=0)
    matched_metrics_count: int = Field(default=0)
    discrepancy_count: int = Field(default=0)
    grounding_score: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Overall cross-modal grounding confidence (0.0 to 1.0)"
    )
    cross_references: List[MetricCrossReference] = Field(default_factory=list)
    critical_discrepancies: List[str] = Field(default_factory=list, description="High-severity contradictions flagged")
    summary_assessment: str = Field(..., description="Executive verification summary for the memo")


# =====================================================================
# 4. Multi-Modal Memo Formatting (Supervisor Synthesis Payload)
# =====================================================================

class VisualAnalyticalMemoBlock(BaseModel):
    """Formatted multi-modal intelligence block ready for the Supervisor to synthesize into the final investment memo."""
    figure_id: str = Field(..., description="Unique ID or file path of the visual figure")
    figure_title: str = Field(..., description="Title of the visual asset")
    page_number: Optional[int] = Field(None, description="PDF page citation")
    executive_summary: str = Field(..., description="Concise visual analytical summary")
    trend_analytics: List[VisualTrendAnalysis] = Field(default_factory=list)
    verification_report: Optional[CrossModalVerificationReport] = Field(None)
    key_takeaways: List[str] = Field(default_factory=list)
    risk_warnings: List[str] = Field(default_factory=list)
    citation_tag: str = Field(..., description="Clickable citation anchor (e.g. '[Fig 3: Page 14 - Revenue Chart]')")
    markdown_formatted_block: str = Field(..., description="Full ready-to-embed markdown snippet for the memo")
