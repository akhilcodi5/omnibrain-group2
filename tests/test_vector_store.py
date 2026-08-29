"""Unit tests for Qdrant VectorStore, document chunk indexing, and similarity search."""

import pytest
import uuid
from storage.vector_store import VectorStore, DefaultEmbedder, get_vector_store


@pytest.fixture
def memory_vector_store():
    """Fixture to provide a clean in-memory VectorStore for tests."""
    embedder = DefaultEmbedder()
    store = VectorStore(
        collection_name=f"test_collection_{uuid.uuid4().hex[:8]}",
        in_memory=True,
        embedder=embedder,
    )
    yield store
    # Teardown
    try:
        store.client.delete_collection(store.collection_name)
    except Exception:
        pass


def test_default_embedder_dimension_and_embedding():
    """Test that DefaultEmbedder generates non-empty vector embeddings of expected dimension."""
    embedder = DefaultEmbedder()
    assert embedder.vector_size > 0
    
    vec = embedder.embed_text("OmniBrain quantitative financial report analysis")
    assert isinstance(vec, list)
    assert len(vec) == embedder.vector_size
    assert all(isinstance(val, float) for val in vec)


def test_default_embedder_batch():
    """Test batch document embedding generation."""
    embedder = DefaultEmbedder()
    texts = [
        "Q3 2026 revenue increased by 14.5% year-over-year.",
        "Operating expenses decreased due to cloud infrastructure optimizations.",
    ]
    embeddings = embedder.embed_documents(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedder.vector_size
    assert len(embeddings[1]) == embedder.vector_size


def test_vector_store_add_documents_and_search(memory_vector_store):
    """Test document chunk indexing and semantic search retrieval."""
    docs = [
        {
            "text": "OmniBrain uses a LangGraph supervisor agent for multi-agent dynamic query routing.",
            "metadata": {
                "pdf_name": "omnibrain_whitepaper.pdf",
                "page_number": 1,
                "section_title": "Architecture Overview",
            },
            "chunk_id": "chunk_001",
        },
        {
            "text": "The Qdrant vector database stores multi-modal text and image embeddings for fast similarity search.",
            "metadata": {
                "pdf_name": "omnibrain_whitepaper.pdf",
                "page_number": 3,
                "section_title": "Vector Storage",
            },
            "chunk_id": "chunk_002",
        },
        {
            "text": "Net income for Q2 2026 reached 42.5 million USD, representing a 12 percent growth.",
            "metadata": {
                "pdf_name": "q2_financials.pdf",
                "page_number": 12,
                "section_title": "Financial Highlights",
            },
            "chunk_id": "chunk_003",
        },
    ]

    inserted_ids = memory_vector_store.add_documents(docs)
    assert len(inserted_ids) == 3
    assert "chunk_001" in inserted_ids

    # Query for vector database storage
    results = memory_vector_store.similarity_search("Tell me about Qdrant vector storage and embeddings", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "text" in top_hit
    assert "score" in top_hit
    assert top_hit["pdf_name"] in ["omnibrain_whitepaper.pdf", "q2_financials.pdf"]
    assert isinstance(top_hit["page_number"], int)


def test_vector_store_metadata_filtering(memory_vector_store):
    """Test filtering search results by metadata (pdf_name and page_number)."""
    docs = [
        {
            "text": "Financial risk disclosure on market volatility.",
            "metadata": {"pdf_name": "annual_report.pdf", "page_number": 5},
            "chunk_id": "c1",
        },
        {
            "text": "Executive summary of quarterly earnings.",
            "metadata": {"pdf_name": "q3_report.pdf", "page_number": 1},
            "chunk_id": "c2",
        },
    ]
    memory_vector_store.add_documents(docs)

    # Search filtered strictly to annual_report.pdf
    filtered_results = memory_vector_store.similarity_search(
        query="report details",
        top_k=5,
        filter_metadata={"pdf_name": "annual_report.pdf"},
    )
    assert len(filtered_results) == 1
    assert filtered_results[0]["pdf_name"] == "annual_report.pdf"
    assert filtered_results[0]["chunk_id"] == "c1"


def test_vector_store_clear_collection(memory_vector_store):
    """Test clearing all vectors from a collection."""
    docs = [
        {
            "text": "Temporary document data for clearing test.",
            "metadata": {"pdf_name": "temp.pdf", "page_number": 1},
        }
    ]
    memory_vector_store.add_documents(docs)
    assert len(memory_vector_store.similarity_search("Temporary", top_k=5)) == 1

    memory_vector_store.clear_collection()
    assert len(memory_vector_store.similarity_search("Temporary", top_k=5)) == 0


def test_get_vector_store_singleton():
    """Test singleton vector store retrieval helper."""
    vs1 = get_vector_store(in_memory=True)
    vs2 = get_vector_store(in_memory=True)
    assert vs1 is vs2
