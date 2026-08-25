"""Pydantic schemas for Multi-Modal Vision extraction, bounding boxes, and table reasoning."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


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


class VLMProviderType(str, Enum):
    """Supported Vision-Language Model providers."""
    OPENAI = "openai"
    LLAVA_OLLAMA = "llava_ollama"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


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


class TableCell(BaseModel):
    """Individual cell in an extracted financial table."""
    row_idx: int = Field(..., description="Row index (0-based)")
    col_idx: int = Field(..., description="Column index (0-based)")
    value: str = Field(..., description="Text content of the cell")
    numeric_value: Optional[float] = Field(None, description="Parsed numeric value if cell is numeric")
    is_header: bool = Field(default=False, description="Whether this cell is part of the table header")


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


class VisionExtractionRequest(BaseModel):
    """Request payload for vision extraction."""
    image_path: Optional[str] = Field(None, description="Local path to the image file")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string")
    image_url: Optional[str] = Field(None, description="Public URL of the image")
    query_context: Optional[str] = Field(
        None, 
        description="Optional user query context to focus visual extraction on specific metrics"
    )
    expected_type: Optional[ChartType] = Field(
        None, 
        description="Hint for expected visual type (e.g. TABLE, BAR, LINE)"
    )
    crop_box: Optional[BoundingBox] = Field(
        None, 
        description="Optional sub-bounding box to crop before VLM inference"
    )


class VisionExtractionResponse(BaseModel):
    """Unified response container for multi-modal vision extraction."""
    image_id: Optional[str] = Field(None, description="Identifier of the source image/figure")
    page_number: Optional[int] = Field(None, description="Source PDF page number if applicable")
    chart_data: Optional[ExtractedChartData] = Field(None, description="Structured chart data if figure is a chart")
    table_data: Optional[ExtractedTableData] = Field(None, description="Structured table data if figure is a table")
    raw_markdown: str = Field(..., description="Markdown-formatted visual interpretation and analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata (model, tokens, latency)")
