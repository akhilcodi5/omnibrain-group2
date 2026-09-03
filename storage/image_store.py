"""Image persistence, caching, and visual asset retrieval manager for OmniBrain."""

import hashlib
import io
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from app.models.vision_schemas import BoundingBox, VerificationStatus

logger = logging.getLogger(__name__)


class ImageStore:
    """Manages disk storage, thumbnail caching, and metadata indexing for extracted document figures."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "storage", "images")
        self.thumbnails_dir = os.path.join(self.base_dir, "thumbnails")
        self.citations_dir = os.path.join(self.base_dir, "citations")
        self.metadata_index_path = os.path.join(self.base_dir, "image_index.json")

        # Ensure directory tree exists
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        os.makedirs(self.citations_dir, exist_ok=True)

        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load asset index metadata from disk."""
        if os.path.exists(self.metadata_index_path):
            try:
                with open(self.metadata_index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load image index JSON: {e}")
        return {}

    def _save_index(self):
        """Persist metadata index to disk."""
        try:
            with open(self.metadata_index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save image index JSON: {e}")

    @staticmethod
    def _compute_sha256(image_bytes: bytes) -> str:
        """Compute SHA-256 hash for deduplication and unique identification."""
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def save_figure_asset(
        self,
        image_input: Union[bytes, Image.Image, str],
        doc_id: str,
        page_number: int,
        figure_name: str = "figure",
        figure_type: str = "chart",
        bounding_box: Optional[BoundingBox] = None,
    ) -> Dict[str, Any]:
        """Save a visual chart/table image asset and index its metadata.
        
        Args:
            image_input: Raw image bytes, PIL Image, or path to file
            doc_id: Unique document identifier
            page_number: Source page number in PDF
            figure_name: Descriptive name/caption
            figure_type: 'bar', 'line', 'table', etc.
            bounding_box: Bounding box coordinates on page
            
        Returns:
            Metadata dictionary containing asset_id, file_path, relative_url, and dimensions.
        """
        # Resolve to PIL Image and bytes
        if isinstance(image_input, bytes):
            image_bytes = image_input
            pil_image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
            buf = io.BytesIO()
            if pil_image.mode in ("RGBA", "P"):
                pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
            else:
                pil_image.save(buf, format="JPEG", quality=90)
            image_bytes = buf.getvalue()
        elif isinstance(image_input, str) and os.path.exists(image_input):
            with open(image_input, "rb") as f:
                image_bytes = f.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
        else:
            raise ValueError("Unsupported or invalid image input.")

        asset_id = f"{doc_id}_p{page_number}_{self._compute_sha256(image_bytes)}"
        
        # Save main image asset
        doc_dir = os.path.join(self.base_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        file_path = os.path.join(doc_dir, f"{asset_id}.png")
        
        pil_image.save(file_path, format="PNG")

        # Generate and cache 300px thumbnail
        thumb_path = os.path.join(self.thumbnails_dir, f"{asset_id}_thumb.jpg")
        thumb_image = pil_image.copy()
        thumb_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
        if thumb_image.mode in ("RGBA", "P"):
            thumb_image = thumb_image.convert("RGB")
        thumb_image.save(thumb_path, format="JPEG", quality=80)

        metadata = {
            "asset_id": asset_id,
            "doc_id": doc_id,
            "page_number": page_number,
            "figure_name": figure_name,
            "figure_type": figure_type,
            "file_path": file_path,
            "thumbnail_path": thumb_path,
            "width": pil_image.width,
            "height": pil_image.height,
            "bounding_box": bounding_box.model_dump() if bounding_box else None,
        }

        self._index[asset_id] = metadata
        self._save_index()

        logger.info(f"Saved visual asset: {asset_id} to {file_path}")
        return metadata

    def get_asset_by_id(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve indexed asset metadata by ID."""
        return self._index.get(asset_id)

    def load_pil_image(self, asset_id: str) -> Optional[Image.Image]:
        """Load the PIL Image object for an asset."""
        meta = self.get_asset_by_id(asset_id)
        if meta and os.path.exists(meta["file_path"]):
            return Image.open(meta["file_path"])
        return None

    def list_document_assets(self, doc_id: str) -> List[Dict[str, Any]]:
        """List all extracted visual assets for a given document."""
        return [meta for meta in self._index.values() if meta.get("doc_id") == doc_id]

    def clear_document_assets(self, doc_id: str):
        """Remove all visual assets for a document."""
        doc_dir = os.path.join(self.base_dir, doc_id)
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir, ignore_errors=True)

        # Clean index
        to_delete = [k for k, v in self._index.items() if v.get("doc_id") == doc_id]
        for k in to_delete:
            del self._index[k]
        self._save_index()


_default_image_store = None


def get_image_store(base_dir: Optional[str] = None) -> ImageStore:
    """Singleton getter for ImageStore."""
    global _default_image_store
    if _default_image_store is None:
        _default_image_store = ImageStore(base_dir=base_dir)
    return _default_image_store
