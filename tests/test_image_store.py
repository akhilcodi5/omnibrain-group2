"""Unit tests for ImageStore figure persistence and thumbnail caching."""

import os
import shutil
import tempfile
import pytest
from PIL import Image

from app.models.vision_schemas import BoundingBox
from storage.image_store import ImageStore


@pytest.fixture
def temp_image_store():
    """Create a temporary ImageStore directory for testing."""
    temp_dir = tempfile.mkdtemp()
    store = ImageStore(base_dir=temp_dir)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_figure_asset(temp_image_store):
    """Test saving a visual figure asset and generating thumbnail."""
    img = Image.new("RGB", (800, 600), color="blue")
    bbox = BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.9, is_normalized=True)

    meta = temp_image_store.save_figure_asset(
        image_input=img,
        doc_id="doc_apex_2024",
        page_number=14,
        figure_name="revenue_chart",
        figure_type="bar",
        bounding_box=bbox,
    )

    assert meta["doc_id"] == "doc_apex_2024"
    assert meta["page_number"] == 14
    assert meta["width"] == 800
    assert meta["height"] == 600
    assert os.path.exists(meta["file_path"])
    assert os.path.exists(meta["thumbnail_path"])


def test_get_and_list_assets(temp_image_store):
    """Test retrieving asset by ID and listing by document ID."""
    img = Image.new("RGB", (400, 300), color="green")
    
    meta = temp_image_store.save_figure_asset(
        image_input=img,
        doc_id="doc_123",
        page_number=2,
        figure_name="margin_table",
    )

    asset_id = meta["asset_id"]
    retrieved = temp_image_store.get_asset_by_id(asset_id)
    assert retrieved is not None
    assert retrieved["figure_name"] == "margin_table"

    doc_assets = temp_image_store.list_document_assets("doc_123")
    assert len(doc_assets) == 1
    assert doc_assets[0]["asset_id"] == asset_id

    # Load PIL image
    loaded_img = temp_image_store.load_pil_image(asset_id)
    assert loaded_img is not None
    assert loaded_img.size == (400, 300)
