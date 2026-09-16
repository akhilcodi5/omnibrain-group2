"""Unit and integration tests for Multi-Modal Document Ingestion Pipeline."""

import io
import os
import pytest
from fastapi.testclient import TestClient

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from app.main import app
from app.models.schemas import PDFPageSchema
from app.services.image_extractor import PDFImageExtractor
from app.services.pdf_parser import PDFParser
from app.services.text_chunker import TextChunker
from storage.vector_store import get_vector_store

client = TestClient(app)


def create_synthetic_pdf_bytes() -> bytes:
    """Generate a synthetic 2-page PDF in memory using PyMuPDF for testing."""
    if fitz is None:
        # Fallback simple binary PDF representation
        return b"%PDF-1.4 sample test pdf document bytes\n%%EOF"

    doc = fitz.open()

    # Page 1: Financial Overview
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "FINANCIAL OVERVIEW\n\n"
        "OmniBrain corporate performance for Q3 2026 demonstrated strong net income growth. "
        "Total quarterly revenue reached 84.5 million USD, representing an 18 percent increase year-over-year. "
        "Operating margins expanded across all core software licensing divisions.",
        fontsize=11,
    )

    # Add a small visual rectangle shape representing a chart/image
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 100, 100), False)
    pix.clear_with(255)  # white background
    page1.insert_image(fitz.Rect(50, 250, 200, 400), pixmap=pix)

    # Page 2: Operational Risks
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "OPERATIONAL RISKS\n\n"
        "Key risk factors include supply chain fluctuations in GPU hardware availability and market volatility. "
        "Management has mitigated these risks through long-term supplier commitments.",
        fontsize=11,
    )

    pdf_buffer = io.BytesIO()
    doc.save(pdf_buffer)
    doc.close()
    return pdf_buffer.getvalue()


def test_pdf_parser():
    """Test PDFParser text extraction and page parsing."""
    pdf_bytes = create_synthetic_pdf_bytes()
    parser = PDFParser()

    pages = parser.parse_pdf(pdf_bytes, pdf_name="test_financials.pdf")

    assert len(pages) >= 1
    p1 = pages[0]
    assert isinstance(p1, PDFPageSchema)
    assert p1.page_number == 1
    assert "OmniBrain" in p1.text or "FINANCIAL" in p1.text or len(p1.text) > 0


def test_pdf_image_extractor(tmp_path):
    """Test PDFImageExtractor figure extraction."""
    pdf_bytes = create_synthetic_pdf_bytes()
    extractor = PDFImageExtractor(default_output_dir=str(tmp_path))

    images = extractor.extract_images(pdf_bytes, pdf_name="test_doc.pdf", min_width=20, min_height=20)

    assert isinstance(images, list)
    if fitz is not None:
        assert len(images) >= 1
        img = images[0]
        assert img.page_number == 1
        assert os.path.exists(img.image_path)


def test_text_chunker():
    """Test TextChunker text splitting and metadata binding."""
    pages = [
        PDFPageSchema(
            page_number=1,
            text="Executive Summary: Q3 2026 performance exceeded market expectations. Revenue grew by 18 percent.",
            section_title="EXECUTIVE SUMMARY",
        ),
        PDFPageSchema(
            page_number=2,
            text="Risk Factors: Hardware supply constraints remain a key operational variable.",
            section_title="RISK FACTORS",
        ),
    ]

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_pages(pages, pdf_name="report.pdf")

    assert len(chunks) >= 2
    c1 = chunks[0]
    assert c1.pdf_name == "report.pdf"
    assert c1.page_number == 1
    assert c1.section_title == "EXECUTIVE SUMMARY"
    assert "Revenue" in c1.text or "Executive" in c1.text
    assert c1.metadata["page_number"] == 1


def test_ingest_api_endpoint():
    """Test POST /api/v1/ingest REST endpoint."""
    pdf_bytes = create_synthetic_pdf_bytes()

    response = client.post(
        "/api/v1/ingest",
        files={"file": ("q3_performance.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["pdf_name"] == "q3_performance.pdf"
    assert data["status"] == "success"
    assert data["total_pages"] >= 1
    assert data["chunks_indexed"] >= 1
    assert isinstance(data["chunk_ids"], list)

    # Verify chunks exist in VectorStore
    vs = get_vector_store(in_memory=True)
    search_results = vs.similarity_search("revenue growth", top_k=2)
    assert len(search_results) > 0


def test_ingest_pdf_alias_endpoint():
    """Test POST /api/v1/ingest/pdf alias REST endpoint."""
    pdf_bytes = create_synthetic_pdf_bytes()

    response = client.post(
        "/api/v1/ingest/pdf",
        files={"file": ("q3_performance_alias.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["pdf_name"] == "q3_performance_alias.pdf"
    assert data["status"] == "success"
    assert data["total_pages"] >= 1



def test_ingest_api_invalid_filetype():
    """Test uploading invalid file format returns 400 Bad Request."""
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("invalid_file.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
