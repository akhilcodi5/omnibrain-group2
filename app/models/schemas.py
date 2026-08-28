"""Pydantic schemas for API requests, responses, and ingestion data structures."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PDFPageSchema(BaseModel):
    """Parsed contents of a single PDF document page."""
    
    page_number: int = Field(..., description="1-based page index")
    text: str = Field("", description="Extracted textual content of the page")
    tables: List[List[List[Optional[str]]]] = Field(default_factory=list, description="Extracted table grids")
    section_title: str = Field("", description="Detected section heading or header")


class ExtractedImageSchema(BaseModel):
    """Metadata and storage path for an extracted figure or chart image."""
    
    image_id: str = Field(..., description="Unique image identifier")
    pdf_name: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number where image was located")
    image_path: str = Field(..., description="Local filesystem path to extracted image file")
    width: int = Field(0, description="Image pixel width")
    height: int = Field(0, description="Image pixel height")


class DocumentChunkSchema(BaseModel):
    """Single semantically chunked text segment ready for vector indexing."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Textual content of chunk")
    pdf_name: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number of origin")
    section_title: str = Field("", description="Section title or heading context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class IngestResponse(BaseModel):
    """API response model for document ingestion endpoint."""
    
    pdf_name: str = Field(..., description="Name of ingested PDF file")
    total_pages: int = Field(..., description="Total pages parsed")
    chunks_indexed: int = Field(..., description="Number of text chunks indexed in vector store")
    images_extracted: int = Field(..., description="Number of figure/chart images extracted")
    status: str = Field("success", description="Status message")
    chunk_ids: List[str] = Field(default_factory=list, description="List of generated chunk IDs")


class ChatRequest(BaseModel):
    """API request model for user chat query."""
    
    query: str = Field(..., description="User query prompt")
    pdf_name_filter: Optional[str] = Field(None, description="Optional PDF document filter")
    top_k: int = Field(5, description="Max document chunks to retrieve")


class ChatResponse(BaseModel):
    """API response model for user chat query."""
    
    query: str = Field(..., description="Original user prompt")
    response: str = Field(..., description="Synthesized response string")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Reference citations")
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved raw context chunks")
    is_grounded: bool = Field(False, description="Grounding verification status")
    confidence: float = Field(0.0, description="Retrieval / synthesis confidence score")
