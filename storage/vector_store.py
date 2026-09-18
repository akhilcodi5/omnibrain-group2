"""Qdrant vector database client, collection lifecycle, and hybrid search methods."""

import logging
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Union

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    QdrantClient = None
    models = None

from app.core.config import settings

logger = logging.getLogger(__name__)


class DefaultEmbedder:
    """Fallback embedding generator using SentenceTransformers or deterministic hashing when offline."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._vector_size = 384
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self._st_model = SentenceTransformer(model_name)
            get_dim_fn = getattr(self._st_model, "get_embedding_dimension", None) or getattr(self._st_model, "get_sentence_embedding_dimension", None)
            self._vector_size = get_dim_fn() if get_dim_fn else 384
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic fallback embeddings.")

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text query or document chunk into a float vector."""
        if self._st_model is not None:
            embedding = self._st_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        
        # Deterministic hashing fallback for offline testing
        vec = []
        for i in range(self._vector_size):
            hash_val = hashlib.sha256(f"{text}_{i}".encode('utf-8')).hexdigest()
            val = (int(hash_val[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)
        
        # Normalize vector
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text document chunks."""
        if self._st_model is not None:
            embeddings = self._st_model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        return [self.embed_text(t) for t in texts]


class VectorStore:
    """Qdrant vector database wrapper for document chunk indexing and semantic search."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        in_memory: bool = False,
        embedder: Optional[Any] = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedder = embedder or DefaultEmbedder()
        self.vector_size = self.embedder.vector_size
        
        if QdrantClient is None:
            raise ImportError("qdrant-client package is required for VectorStore.")

        if in_memory:
            logger.info("Initializing QdrantClient in-memory mode.")
            self.client = QdrantClient(location=":memory:")
        else:
            host_val = host or settings.QDRANT_HOST
            port_val = port or settings.QDRANT_PORT
            api_key_val = api_key or settings.QDRANT_API_KEY
            try:
                logger.info(f"Connecting to Qdrant server at {host_val}:{port_val}")
                self.client = QdrantClient(host=host_val, port=port_val, api_key=api_key_val, timeout=3.0)
                # Test connection
                self.client.get_collections()
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {host_val}:{port_val} ({e}). Falling back to persistent disk mode.")
                self.client = QdrantClient(path="storage/qdrant_data")

        self.ensure_collection_exists()

    def ensure_collection_exists(self, vector_size: Optional[int] = None, distance_metric: str = "COSINE") -> None:
        """Create the Qdrant collection if it does not already exist."""
        size = vector_size or self.vector_size
        metric = models.Distance.COSINE if distance_metric.upper() == "COSINE" else models.Distance.DOT
        
        try:
            collections_res = self.client.get_collections()
            existing_names = [c.name for c in collections_res.collections]
            
            if self.collection_name not in existing_names:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (vector size: {size})")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=size, distance=metric),
                )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection exists: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Index a batch of text document chunks into Qdrant.
        
        Args:
            documents: List of dicts, each containing:
                - 'text' (str, required): Document chunk text
                - 'metadata' (dict, optional): Chunk metadata (pdf_name, page_number, section_title, etc.)
                - 'chunk_id' (str, optional): Unique chunk ID

        Returns:
            List of generated/assigned point string IDs.
        """
        if not documents:
            return []

        texts = [doc.get("text", "") for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
        
        points = []
        inserted_ids = []

        for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
            chunk_id = str(doc.get("chunk_id") or doc.get("id") or uuid.uuid4())
            inserted_ids.append(chunk_id)
            
            # Ensure Qdrant point ID is a valid UUID
            try:
                point_id = str(uuid.UUID(chunk_id))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            metadata = doc.get("metadata", {})
            payload = {
                "text": doc.get("text", ""),
                "chunk_id": chunk_id,
                "pdf_name": metadata.get("pdf_name", doc.get("pdf_name", "unknown.pdf")),
                "page_number": metadata.get("page_number", doc.get("page_number", 1)),
                "section_title": metadata.get("section_title", doc.get("section_title", "")),
                "metadata": metadata,
            }
            
            # Additional root-level items in payload if present
            if "table_data" in doc:
                payload["table_data"] = doc["table_data"]
            if "image_id" in doc:
                payload["image_id"] = doc["image_id"]

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Indexed {len(points)} document chunks into '{self.collection_name}'")
        return inserted_ids

    def add_chunks(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Convenience method to index raw texts and metadatas into the vector store.
        
        Args:
            texts: List of text chunk strings.
            metadatas: Optional list of metadata dicts corresponding to texts.
            
        Returns:
            List of generated/assigned chunk IDs.
        """
        docs = []
        for idx, t in enumerate(texts):
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            chunk_id = meta.get("chunk_id", str(uuid.uuid4()))
            docs.append({
                "text": t,
                "metadata": meta,
                "pdf_name": meta.get("pdf_name", "unknown.pdf"),
                "page_number": meta.get("page_number", 1),
                "section_title": meta.get("section_title", ""),
                "chunk_id": chunk_id,
            })
        return self.add_documents(docs)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search over stored document chunks.
        
        Args:
            query: User text query
            top_k: Number of top relevant results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Dictionary of metadata key-value filters (e.g. {"pdf_name": "report.pdf", "page_number": 2})

        Returns:
            List of dictionaries containing matching chunks, scores, text, and metadata.
        """
        if not query or not query.strip():
            return []

        query_vector = self.embedder.embed_text(query)
        
        # Build filter if requested
        query_filter = None
        if filter_metadata:
            must_conditions = []
            for key, val in filter_metadata.items():
                # Check root fields or metadata nested fields
                field_key = f"metadata.{key}" if key not in ["pdf_name", "page_number", "section_title", "chunk_id"] else key
                must_conditions.append(
                    models.FieldCondition(
                        key=field_key,
                        match=models.MatchValue(value=val),
                    )
                )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        # Support client.search or client.query_points API
        results = []
        try:
            if hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
                hits = response.points

            for hit in hits:
                payload = hit.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                    "pdf_name": payload.get("pdf_name", "unknown.pdf"),
                    "page_number": payload.get("page_number", 1),
                    "section_title": payload.get("section_title", ""),
                    "metadata": payload.get("metadata", {}),
                    "table_data": payload.get("table_data"),
                    "image_id": payload.get("image_id"),
                })
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

        return results

    def clear_collection(self) -> None:
        """Clear all points from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.ensure_collection_exists()
            logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}")


# Singleton instance manager
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store(in_memory: bool = False) -> VectorStore:
    """Factory function for retrieving or initializing the singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(in_memory=in_memory)
    return _vector_store_instance
