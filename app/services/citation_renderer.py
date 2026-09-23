"""Visual Citation & Bounding Box Overlay Generator for Multi-Modal Grounding."""

import base64
import io
import logging
import os
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from app.models.vision_schemas import (
    BoundingBox,
    VerificationStatus,
    VisualCitationPayload,
)

logger = logging.getLogger(__name__)


class CitationRenderer:
    """Renders visual overlays, bounding-box highlights, and thumbnail snippets for citations."""

    COLOR_MAP = {
        VerificationStatus.VERIFIED_MATCH: (16, 185, 129, 255),      # Emerald Green
        VerificationStatus.DISCREPANCY_DETECTED: (239, 68, 68, 255), # Crimson Red
        VerificationStatus.VISUAL_MISSING_IN_TEXT: (245, 158, 11, 255), # Amber Yellow
        VerificationStatus.UNVERIFIABLE: (148, 163, 184, 255),       # Slate Gray
    }

    @classmethod
    def render_citation_overlay(
        cls,
        image: Image.Image,
        bounding_box: BoundingBox,
        figure_title: str,
        page_number: Optional[int] = None,
        status: VerificationStatus = VerificationStatus.VERIFIED_MATCH,
        grounding_confidence: float = 1.0,
    ) -> VisualCitationPayload:
        """Draw an illuminated bounding box with a status tag on the page image and create a thumbnail."""
        if image.mode != "RGBA":
            page_rgba = image.convert("RGBA")
        else:
            page_rgba = image.copy()

        width, height = page_rgba.size

        # Resolve box coordinates
        if bounding_box.is_normalized:
            left = int(bounding_box.xmin * width)
            top = int(bounding_box.ymin * height)
            right = int(bounding_box.xmax * width)
            bottom = int(bounding_box.ymax * height)
        else:
            left = int(bounding_box.xmin)
            top = int(bounding_box.ymin)
            right = int(bounding_box.xmax)
            bottom = int(bounding_box.ymax)

        # Ensure inside image boundaries
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        # 1. Create Crop Thumbnail Snippet
        cropped_snippet = page_rgba.crop((left, top, right, bottom))
        thumb_buffer = io.BytesIO()
        cropped_snippet.convert("RGB").save(thumb_buffer, format="JPEG", quality=85)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode("utf-8")

        # 2. Draw Highlight Overlay on Page
        overlay = Image.new("RGBA", page_rgba.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        stroke_color = cls.COLOR_MAP.get(status, (16, 185, 129, 255))
        # Draw translucent fill inside bounding box
        fill_color = (stroke_color[0], stroke_color[1], stroke_color[2], 35)
        draw.rectangle([left, top, right, bottom], fill=fill_color, outline=stroke_color, width=4)

        # Draw Tag Label Header
        tag_text = f" {figure_title} | Grounding: {int(grounding_confidence * 100)}% "
        badge_top = max(0, top - 24)
        badge_bottom = top
        draw.rectangle([left, badge_top, min(width, left + len(tag_text) * 8), badge_bottom], fill=stroke_color)
        draw.text((left + 4, badge_top + 4), tag_text, fill=(255, 255, 255, 255))

        highlighted_page = Image.alpha_composite(page_rgba, overlay)

        page_buffer = io.BytesIO()
        highlighted_page.convert("RGB").save(page_buffer, format="JPEG", quality=85)
        page_base64 = base64.b64encode(page_buffer.getvalue()).decode("utf-8")

        citation_tag = f"[{figure_title} - Page {page_number}]" if page_number else f"[{figure_title}]"

        return VisualCitationPayload(
            figure_id=figure_title.lower().replace(" ", "_"),
            figure_title=figure_title,
            page_number=page_number,
            citation_tag=citation_tag,
            status=status,
            highlighted_page_base64=page_base64,
            thumbnail_snippet_base64=thumb_base64,
            grounding_confidence=grounding_confidence,
        )
