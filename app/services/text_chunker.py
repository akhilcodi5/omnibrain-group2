"""Text chunking service for semantic segmentation with page and layout metadata binding."""

import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from app.models.schemas import DocumentChunkSchema, PDFPageSchema

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits parsed PDF page text into semantically cohesive chunks for vector storage."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text_recursive(self, text: str) -> List[str]:
        """Recursively split a text string into chunks based on paragraph and sentence separators."""
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        chunks = []
        
        # Try separators in order
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                current_chunk = []
                current_len = 0
                
                for part in parts:
                    part_str = part + sep if sep != " " else part + " "
                    if current_len + len(part_str) > self.chunk_size and current_chunk:
                        chunk_text = "".join(current_chunk).strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                        
                        # Apply overlap
                        overlap_len = 0
                        overlap_parts = []
                        for prev in reversed(current_chunk):
                            if overlap_len + len(prev) <= self.chunk_overlap:
                                overlap_parts.insert(0, prev)
                                overlap_len += len(prev)
                            else:
                                break
                        
                        current_chunk = overlap_parts + [part_str]
                        current_len = sum(len(p) for p in current_chunk)
                    else:
                        current_chunk.append(part_str)
                        current_len += len(part_str)

                if current_chunk:
                    chunk_text = "".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                
                if chunks:
                    return chunks

        # Fallback character slicing
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i : i + self.chunk_size].strip() for i in range(0, len(text), step) if text[i : i + self.chunk_size].strip()]

    def chunk_pages(
        self,
        pages: List[PDFPageSchema],
        pdf_name: str = "document.pdf",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[DocumentChunkSchema]:
        """Process parsed PDF pages and create DocumentChunkSchema instances with metadata.
        
        Args:
            pages: List of PDFPageSchema objects
            pdf_name: Name of PDF file
            chunk_size: Overriding target max character count per chunk
            chunk_overlap: Overriding character overlap

        Returns:
            List of DocumentChunkSchema instances ready for vector indexing.
        """
        c_size = chunk_size or self.chunk_size
        c_overlap = chunk_overlap or self.chunk_overlap

        clean_pdf_name = os.path.splitext(os.path.basename(pdf_name))[0]
        chunks: List[DocumentChunkSchema] = []

        current_heading = "Document Overview"

        for page in pages:
            page_num = page.page_number
            if page.section_title:
                current_heading = page.section_title

            page_text = page.text or ""

            # Append formatted table representations if present
            if page.tables:
                table_strs = []
                for tbl in page.tables:
                    rows = [" | ".join([str(cell) if cell is not None else "" for cell in row]) for row in tbl]
                    table_strs.append("\n[TABLE DATA]\n" + "\n".join(rows) + "\n[/TABLE DATA]")
                page_text += "\n\n" + "\n".join(table_strs)

            if not page_text.strip():
                continue

            raw_chunks = self._split_text_recursive(page_text)

            for idx, raw_text in enumerate(raw_chunks, 1):
                chunk_id = f"chk_{clean_pdf_name}_p{page_num}_{idx}_{uuid.uuid4().hex[:6]}"

                chunks.append(
                    DocumentChunkSchema(
                        chunk_id=chunk_id,
                        text=raw_text,
                        pdf_name=pdf_name,
                        page_number=page_num,
                        section_title=current_heading,
                        metadata={
                            "pdf_name": pdf_name,
                            "page_number": page_num,
                            "section_title": current_heading,
                            "char_count": len(raw_text),
                            "has_table": bool(page.tables),
                        },
                    )
                )

        logger.info(f"Generated {len(chunks)} text chunks from {len(pages)} pages of '{pdf_name}'")
        return chunks
