"""PDF image and embedded chart extraction service."""

import logging
import io
import os
import uuid
from typing import Any, Dict, List, Optional, Union
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from app.models.schemas import ExtractedImageSchema

logger = logging.getLogger(__name__)


class PDFImageExtractor:
    """Extract embedded figures, charts, and visual figures from PDF pages."""

    def __init__(self, default_output_dir: str = "storage/extracted_images"):
        self.default_output_dir = default_output_dir
        os.makedirs(self.default_output_dir, exist_ok=True)

    def extract_images(
        self,
        file_input: Union[str, bytes, io.BytesIO],
        pdf_name: str = "document.pdf",
        output_dir: Optional[str] = None,
        min_width: int = 60,
        min_height: int = 60,
    ) -> List[ExtractedImageSchema]:
        """Extract embedded images from a PDF and save them to the specified output directory.
        
        Args:
            file_input: File path (str), raw PDF bytes (bytes), or BytesIO stream
            pdf_name: Filename string for naming output images
            output_dir: Target folder path for saving extracted images
            min_width: Minimum width threshold to filter icons/bullets
            min_height: Minimum height threshold to filter icons/bullets

        Returns:
            List of ExtractedImageSchema instances containing image paths and dimensions.
        """
        target_dir = output_dir or self.default_output_dir
        os.makedirs(target_dir, exist_ok=True)

        extracted_images: List[ExtractedImageSchema] = []

        if isinstance(file_input, str):
            if not os.path.exists(file_input):
                logger.error(f"PDF file path does not exist: {file_input}")
                return []
            with open(file_input, "rb") as f:
                pdf_bytes = f.read()
        elif isinstance(file_input, io.BytesIO):
            pdf_bytes = file_input.getvalue()
        else:
            pdf_bytes = file_input

        if fitz is None or not pdf_bytes:
            logger.warning("PyMuPDF (fitz) unavailable or empty PDF bytes. Skipping image extraction.")
            return []

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            clean_pdf_name = os.path.splitext(os.path.basename(pdf_name))[0]

            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list, 1):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    image_ext = base_image.get("ext", "png")

                    if not image_bytes:
                        continue

                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_img.size

                        # Filter tiny images (e.g. logos, lines, bullet points)
                        if width < min_width or height < min_height:
                            continue

                        image_id = f"img_{clean_pdf_name}_p{page_num}_{img_idx}_{uuid.uuid4().hex[:6]}"
                        file_filename = f"{image_id}.{image_ext}"
                        image_path = os.path.join(target_dir, file_filename)

                        pil_img.save(image_path)

                        extracted_images.append(
                            ExtractedImageSchema(
                                image_id=image_id,
                                pdf_name=pdf_name,
                                page_number=page_num,
                                image_path=image_path,
                                width=width,
                                height=height,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Could not process image xref {xref} on page {page_num}: {e}")

            doc.close()
        except Exception as e:
            logger.error(f"Image extraction failed for PDF '{pdf_name}': {e}")

        logger.info(f"Extracted {len(extracted_images)} figures/charts from '{pdf_name}'")
        return extracted_images
