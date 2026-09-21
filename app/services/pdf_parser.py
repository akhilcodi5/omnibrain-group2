"""PDF parsing service for extracting text, layout metadata, and table structures."""

import logging
import io
import re
from typing import Any, Dict, List, Optional, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pypdf
except ImportError:
    pypdf = None

from app.models.schemas import PDFPageSchema

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF Document parsing engine utilizing PyMuPDF and pdfplumber."""

    def __init__(self):
        if fitz is None:
            logger.warning("PyMuPDF (fitz) is not installed. PDF text parsing will use fallback mode.")

    def _extract_section_heading(self, page_text: str) -> str:
        """Extract dominant section heading or title from page text lines using heuristics."""
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if not lines:
            return ""

        # Look for headers like "1. Executive Summary", "FINANCIAL HIGHLIGHTS", "Section A", etc.
        for line in lines[:5]:
            if len(line) < 80 and (
                line.isupper()
                or re.match(r"^(?:[0-9]+(?:\.[0-9]+)*|Section|CHAPTER|PART)\b", line, re.IGNORECASE)
                or line.endswith(":")
            ):
                return line

        # Return first non-empty line as fallback heading if short enough
        return lines[0] if len(lines[0]) < 60 else ""

    def parse_pdf(
        self,
        file_input: Union[str, bytes, io.BytesIO],
        pdf_name: str = "document.pdf",
    ) -> List[PDFPageSchema]:
        """Parse a PDF file from a file path, raw bytes, or BytesIO buffer into PDFPageSchema instances.
        
        Args:
            file_input: File path (str), raw PDF bytes (bytes), or BytesIO stream
            pdf_name: Filename string for reference

        Returns:
            List of PDFPageSchema objects, one per page.
        """
        pages: List[PDFPageSchema] = []

        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                pdf_bytes = f.read()
        elif isinstance(file_input, io.BytesIO):
            pdf_bytes = file_input.getvalue()
        else:
            pdf_bytes = file_input

        # 1. Parse page text & headings via PyMuPDF (fitz)
        if fitz is not None:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    page_num = page_idx + 1
                    text = page.get_text("text") or ""
                    heading = self._extract_section_heading(text)

                    pages.append(
                        PDFPageSchema(
                            page_number=page_num,
                            text=text,
                            tables=[],
                            section_title=heading,
                        )
                    )
                doc.close()
            except Exception as e:
                logger.error(f"PyMuPDF failed to parse PDF '{pdf_name}': {e}")

        # 2. Extract tables via pdfplumber if available
        if pdfplumber is not None and pdf_bytes:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    for page_idx, plumber_page in enumerate(pdf.pages):
                        if page_idx < len(pages):
                            extracted_tables = plumber_page.extract_tables() or []
                            pages[page_idx].tables = extracted_tables
            except Exception as e:
                logger.warning(f"pdfplumber table extraction warning for '{pdf_name}': {e}")

        # Fallback if fitz was unavailable or returned 0 pages
        if not pages and pdf_bytes:
            if pypdf is not None:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                    for idx, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        heading = self._extract_section_heading(page_text)
                        pages.append(
                            PDFPageSchema(
                                page_number=idx + 1,
                                text=page_text.strip(),
                                tables=[],
                                section_title=heading or f"Page {idx + 1}",
                            )
                        )
                except Exception as e:
                    logger.debug(f"pypdf fallback extraction warning: {e}")

            if not pages:
                logger.warning(f"Using fallback text decoder for '{pdf_name}'")
                try:
                    raw_str = pdf_bytes.decode("latin-1", errors="ignore")
                    # Basic string text extraction fallback
                    text_clean = re.sub(r"[^\x20-\x7E\n\t]", " ", raw_str)[:2000]
                    pages.append(
                        PDFPageSchema(
                            page_number=1,
                            text=text_clean,
                            tables=[],
                            section_title="Document Overview",
                        )
                    )
                except Exception as e:
                    logger.error(f"Fallback text decoder failed: {e}")

        logger.info(f"Successfully parsed {len(pages)} pages from '{pdf_name}'")
        return pages
