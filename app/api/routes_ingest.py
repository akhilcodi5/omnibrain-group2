"""FastAPI endpoint for multi-modal document ingestion and vector indexing."""

import logging
from typing import Any, Dict
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.schemas import IngestResponse
from app.services.image_extractor import PDFImageExtractor
from app.services.pdf_parser import PDFParser
from app.services.text_chunker import TextChunker
from storage.vector_store import get_vector_store
from storage.sql_db import get_financial_db
from app.core.telemetry import get_telemetry_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a multi-modal PDF document",
    description="Parses PDF text, extracts embedded chart images, chunks text semantically, and indexes vectors into Qdrant.",
)
@router.post(
    "/ingest/pdf",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a multi-modal PDF document (alias)",
)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Handle asynchronous document upload, multi-modal parsing, and vector database indexing."""
    filename = file.filename or "uploaded_document.pdf"
    
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format for '{filename}'. Only PDF files are supported.",
        )

    telemetry = get_telemetry_manager()
    trace_id = telemetry.create_trace(name="Ingest_Document", user_id="system")
    
    try:
        logger.info(f"Receiving file upload for ingestion: '{filename}'")
        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # 1. Parse PDF pages and tables
        parser = PDFParser()
        pages = parser.parse_pdf(pdf_bytes, pdf_name=filename)

        # 2. Extract embedded images & figures
        image_extractor = PDFImageExtractor()
        extracted_images = image_extractor.extract_images(pdf_bytes, pdf_name=filename)
        
        telemetry.log_event(
            trace_id=trace_id,
            name="PDF_Metadata_Parsed",
            metadata={
                "pdf_name": filename,
                "size_bytes": len(pdf_bytes),
                "total_pages": len(pages),
                "images_found": len(extracted_images)
            }
        )

        image_metadata = [
            {"image_name": img.image_path.split('/')[-1], "path": img.image_path, "page": img.page_number}
            for img in extracted_images
        ]
        telemetry.log_event(
            trace_id=trace_id,
            name="Images_Extracted",
            metadata={"extracted_images": image_metadata}
        )

        # 3. Chunk text semantically
        chunker = TextChunker()
        chunks = chunker.chunk_pages(pages, pdf_name=filename)

        # 4. Bulk index chunks into Qdrant VectorStore
        vector_store = get_vector_store(in_memory=False)
        doc_payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "pdf_name": chunk.pdf_name,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        inserted_ids = vector_store.add_documents(doc_payloads)

        # 5. Ingest extracted 2D PDF tables into SQLite database for SQL queries
        try:
            db = get_financial_db()
            tables_by_page = [p.tables for p in pages]
            table_metadata = db.ingest_pdf_tables(pdf_name=filename, tables_by_page=tables_by_page)
            
            telemetry.log_event(
                trace_id=trace_id,
                name="Tables_Extracted_To_SQL",
                metadata={
                    "db_path": db.db_path,
                    "tables_ingested": table_metadata
                }
            )
        except Exception as e:
            logger.warning(f"Could not ingest tabular data into SQL database: {e}")

        logger.info(f"Successfully ingested '{filename}': {len(pages)} pages, {len(chunks)} chunks, {len(extracted_images)} images.")


        return IngestResponse(
            pdf_name=filename,
            total_pages=len(pages),
            chunks_indexed=len(inserted_ids),
            images_extracted=len(extracted_images),
            extracted_image_paths=[img.image_path for img in extracted_images],
            status="success",
            chunk_ids=inserted_ids,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during document ingestion for '{filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and ingest PDF document: {str(e)}",
        )
