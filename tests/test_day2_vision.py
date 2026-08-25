"""Unit tests for Day 2 Multi-Modal Vision Specialist modules (Preprocessing, Engines, Cropping)."""

import pytest
from PIL import Image

from agents.vision_agent import VisionAgent
from app.models.vision_schemas import (
    BoundingBox,
    ChartType,
    ExtractedChartData,
    ExtractedTableData,
    VLMProviderType,
)
from app.services.image_preprocessor import ImagePreprocessor
from app.services.vision_service import VisionService
from app.services.vlm_engine import (
    MockVisionEngine,
    OllamaLLaVAEngine,
    OpenAIVisionEngine,
    get_vision_engine,
)


def test_image_preprocessor_resizing():
    """Test that oversized images are scaled down while maintaining proportions."""
    large_img = Image.new("RGB", (4000, 3000), color="white")
    resized = ImagePreprocessor.resize_for_vlm(large_img, max_dimension=2048)
    
    assert max(resized.size) <= 2048
    # Aspect ratio check
    original_ratio = round(4000 / 3000, 2)
    new_ratio = round(resized.size[0] / resized.size[1], 2)
    assert original_ratio == new_ratio


def test_image_preprocessor_enhancement():
    """Test contrast and sharpness enhancement on sample figure."""
    img = Image.new("RGB", (200, 200), color="gray")
    enhanced = ImagePreprocessor.enhance_chart_readability(img, contrast_factor=1.5, sharpness_factor=1.3)
    
    assert enhanced is not None
    assert enhanced.size == (200, 200)


def test_image_preprocessor_cropping():
    """Test bounding box cropping with both pixel and normalized coordinates."""
    img = Image.new("RGB", (1000, 800), color="blue")
    
    # 1. Pixel crop
    pixel_crop = ImagePreprocessor.crop_bounding_box(img, box=(100, 100, 500, 400), normalized=False)
    assert pixel_crop.size == (400, 300)
    
    # 2. Normalized crop (0.0 to 1.0)
    norm_crop = ImagePreprocessor.crop_bounding_box(img, box=(0.1, 0.1, 0.6, 0.5), normalized=True)
    assert norm_crop.size == (500, 320)


def test_image_preprocessor_vlm_metadata():
    """Test token cost and tile calculation metadata."""
    img = Image.new("RGB", (1024, 768), color="white")
    meta = ImagePreprocessor.get_image_vlm_metadata(img)
    
    assert meta["width"] == 1024
    assert meta["height"] == 768
    assert meta["estimated_vlm_tiles"] > 0
    assert meta["estimated_token_cost"] > 0


def test_vlm_engine_factory():
    """Test engine factory instantiation for mock, ollama, and openai."""
    mock_engine = get_vision_engine("mock")
    assert isinstance(mock_engine, MockVisionEngine)

    ollama_engine = get_vision_engine("ollama")
    assert isinstance(ollama_engine, OllamaLLaVAEngine)


@pytest.mark.asyncio
async def test_structured_chart_extraction():
    """Test structured chart extraction returning valid Pydantic model."""
    service = VisionService(engine=MockVisionEngine())
    img = Image.new("RGB", (100, 100), color="green")
    
    chart_data = await service.extract_structured_chart(img, query_context="Operating Margin")
    
    assert isinstance(chart_data, ExtractedChartData)
    assert chart_data.chart_type == ChartType.BAR
    assert len(chart_data.series) == 1
    assert len(chart_data.series[0].data_points) == 4
    assert chart_data.series[0].data_points[0].value == 110.0


@pytest.mark.asyncio
async def test_vision_agent_with_crop_box():
    """Test VisionAgent analyzing a cropped subregion."""
    agent = VisionAgent(vision_service=VisionService(engine=MockVisionEngine()))
    img = Image.new("RGB", (800, 600), color="yellow")
    
    bbox = BoundingBox(ymin=0.2, xmin=0.1, ymax=0.8, xmax=0.9, is_normalized=True)
    
    response = await agent.analyze_visual_asset(
        image_input=img,
        query_context="Check revenue growth",
        expected_type=ChartType.BAR,
        crop_box=bbox,
    )
    
    assert response is not None
    assert response.raw_markdown != ""
