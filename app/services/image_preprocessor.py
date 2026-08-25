"""Image preprocessing, enhancement, and tiling utilities for Multi-Modal Vision models."""

import io
import logging
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses document images, charts, and tables for optimal VLM token economy and readability."""

    @staticmethod
    def resize_for_vlm(
        image: Image.Image,
        max_dimension: int = 2048,
        min_dimension: int = 512,
    ) -> Image.Image:
        """Resize image to fit within VLM token/resolution sweet spots while preserving aspect ratio.
        
        Most VLMs (like GPT-4o) tile images in 512x512 patches. This utility ensures images
        are neither excessively large (causing high token latency) nor too small (losing small font details).
        """
        width, height = image.size
        
        # If image is larger than max dimension, scale down
        if max(width, height) > max_dimension:
            scaling_factor = max_dimension / float(max(width, height))
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Scaled down image from ({width}, {height}) to ({new_width}, {new_height})")
            
        # If image is smaller than min dimension on the short edge, upscale slightly for OCR clarity
        elif min(width, height) < min_dimension:
            scaling_factor = min_dimension / float(min(width, height))
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            logger.debug(f"Upscaled image from ({width}, {height}) to ({new_width}, {new_height})")

        return image

    @staticmethod
    def enhance_chart_readability(
        image: Image.Image,
        contrast_factor: float = 1.3,
        sharpness_factor: float = 1.2,
    ) -> Image.Image:
        """Enhance contrast and sharpness to make faded financial chart axis labels and grid lines legible."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Enhance Contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(contrast_factor)

        # 2. Enhance Sharpness for small table text and numbers
        sharpness = ImageEnhance.Sharpness(enhanced_image)
        sharpened_image = sharpness.enhance(sharpness_factor)

        return sharpened_image

    @staticmethod
    def crop_bounding_box(
        image: Image.Image,
        box: Tuple[int, int, int, int],
        normalized: bool = False,
    ) -> Image.Image:
        """Crop a sub-region (chart or table) from a full PDF page.
        
        Args:
            image: PIL Image of the page
            box: (left, top, right, bottom)
            normalized: If True, coordinates are 0.0 to 1.0 (or 0 to 1000)
        """
        width, height = image.size
        left, top, right, bottom = box

        if normalized:
            scale_x = width if max(left, right) <= 1.0 else (width / 1000.0)
            scale_y = height if max(top, bottom) <= 1.0 else (height / 1000.0)
            
            left = int(left * scale_x)
            top = int(top * scale_y)
            right = int(right * scale_x)
            bottom = int(bottom * scale_y)

        # Ensure bounds are valid
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        return image.crop((left, top, right, bottom))

    @staticmethod
    def get_image_vlm_metadata(image: Image.Image) -> Dict[str, Any]:
        """Compute image statistics including dimensions, aspect ratio, and estimated GPT-4o tile tokens."""
        width, height = image.size
        aspect_ratio = round(width / float(height), 3) if height > 0 else 1.0

        # GPT-4o high-detail calculation: scaled to max 2048, min 768, then counted in 512x512 tiles
        # Each tile is 170 tokens + 85 base tokens
        tiles_x = (width + 511) // 512
        tiles_y = (height + 511) // 512
        estimated_tiles = tiles_x * tiles_y
        estimated_tokens = (estimated_tiles * 170) + 85

        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "mode": image.mode,
            "estimated_vlm_tiles": estimated_tiles,
            "estimated_token_cost": estimated_tokens,
        }
