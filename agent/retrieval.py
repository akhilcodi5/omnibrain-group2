"""Grounded hybrid retrieval and bounded Self-RAG orchestration.

The concrete vector store, BM25 index and LLM are injected by the application.
Keeping those edges as small callables makes the safety-critical metadata and
citation handling deterministic and independently testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional


def _configured_attempts() -> int:
    try:
        return max(1, int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3")))
    except ValueError:
        return 3


MAX_RETRIEVAL_ATTEMPTS = _configured_attempts()


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    page_number: Optional[int] = None
    source: Optional[str] = None
    score: float = 0.0
    retrieval_method: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any, method: str) -> "RetrievedChunk":
        if isinstance(value, cls):
            return cls(**{**value.__dict__, "retrieval_method": method})
        if not isinstance(value, Mapping):
            raise TypeError("retrieval results must be mappings or RetrievedChunk instances")
        metadata = dict(value.get("metadata") or {})
        def get(key: str, default: Any = None) -> Any:
            return value.get(key, metadata.get(key, default))
        required = ("document_id", "document_name", "chunk_id")
        missing = [key for key in required if get(key) is None]
        if missing:
            raise ValueError("retrieval result missing required metadata: " + ", ".join(missing))
        return cls(
            document_id=str(get("document_id")), document_name=str(get("document_name")),
            chunk_id=str(get("chunk_id")), text=str(get("text", "")),
            page_number=get("page_number"), source=get("source"),
            score=float(get("score", 0.0)), retrieval_method=method, metadata=metadata,
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.document_id, self.chunk_id

    def as_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "document_name": self.document_name,
                "chunk_id": self.chunk_id, "text": self.text, "page_number": self.page_number,
                "source": self.source, "score": self.score,
                "retrieval_method": self.retrieval_method, "metadata": dict(self.metadata)}


class HybridRetriever:
    """Fuses independently-failable semantic and keyword retrieval with RRF."""
    def __init__(self, semantic_search: Optional[Callable[..., Iterable[Any]]] = None,
                 keyword_search: Optional[Callable[..., Iterable[Any]]] = None,
                 reranker: Optional[Callable[[str, list[RetrievedChunk]], Iterable[Any]]] = None,
                 rrf_k: int = 60):
        self.semantic_search, self.keyword_search = semantic_search, keyword_search
        self.reranker, self.rrf_k = reranker, rrf_k

    def _run(self, search: Optional[Callable[..., Iterable[Any]]], query: str, method: str,
             metadata_filter: Optional[Mapping[str, Any]]) -> list[RetrievedChunk]:
        if search is None:
            return []
        try:
            values = search(query, metadata_filter=metadata_filter)
            if metadata_filter is not None:
                return []
            values = search(query)
        except Exception:
            return []
        chunks = []
        try:
            for value in values or []:
                try:
                    chunks.append(RetrievedChunk.from_value(value, method))
                except (TypeError, ValueError):
                    continue
        except Exception:
            return chunks
        return chunks

    def retrieve(self, query: str, top_k: int = 8,
                 metadata_filter: Optional[Mapping[str, Any]] = None) -> list[RetrievedChunk]:
        semantic = self._run(self.semantic_search, query, "semantic", metadata_filter)
        keyword = self._run(self.keyword_search, query, "keyword", metadata_filter)
        fused: dict[tuple[str, str], RetrievedChunk] = {}
        scores: dict[tuple[str, str], float] = {}
        methods: dict[tuple[str, str], set[str]] = {}
        for results in (semantic, keyword):
            for rank, chunk in enumerate(results, start=1):
                key = chunk.key
                fused.setdefault(key, chunk)  # Keeps page/source from an actual retrieved chunk.
                scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_k + rank)
                methods.setdefault(key, set()).add(chunk.retrieval_method)
        ordered = []
        for key, chunk in fused.items():
            ordered.append(RetrievedChunk(**{**chunk.__dict__, "score": scores[key],
                "retrieval_method": "+".join(sorted(methods[key]))}))
        ordered.sort(key=lambda item: item.score, reverse=True)
        if self.reranker and ordered:
            try:
                reranked = list(self.reranker(query, ordered))
                ordered = [RetrievedChunk.from_value(item, "reranked") for item in reranked]
            except Exception:
                pass
        return ordered[:top_k]


@dataclass(frozen=True)
class RelevanceEvaluation:
    relevant: bool
    confidence: float
    reason: str = ""
    missing_information: str = ""
    recommended_action: str = "answer"


def parse_evaluation(value: Any, has_context: bool) -> RelevanceEvaluation:
    """Safely normalise untrusted LLM output without exposing reasoning."""
    if not isinstance(value, Mapping):
        return RelevanceEvaluation(has_context, 0.5 if has_context else 0.0,
                                   "Evaluator response unavailable", "", "answer" if has_context else "retrieve_more")
    try:
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    action = value.get("recommended_action", "answer")
    if action not in {"answer", "rewrite_query", "retrieve_more"}:
        action = "answer" if bool(value.get("relevant")) else "retrieve_more"
    return RelevanceEvaluation(bool(value.get("relevant")) and has_context, confidence,
        str(value.get("reason", "")), str(value.get("missing_information", "")), action)


def validate_citations(candidates: Iterable[Mapping[str, Any]],
                       context: Iterable[RetrievedChunk]) -> list[dict[str, Any]]:
    """Only return citations backed by the current retrieved evidence."""
    evidence = {chunk.key: chunk for chunk in context}
    valid, seen = [], set()
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        key = (str(candidate.get("document_id", "")), str(candidate.get("chunk_id", "")))
        if not chunk:
            continue
        page = candidate.get("page_number")
        # A model can name evidence, but metadata remains the source of truth.
        if page is not None and page != chunk.page_number:
            continue
        citation = {"document_id": chunk.document_id, "document_name": chunk.document_name,
                    "chunk_id": chunk.chunk_id, "page_number": chunk.page_number,
                    "source": chunk.source}
        fingerprint = (citation["document_id"], citation["chunk_id"], citation["page_number"])
        if fingerprint not in seen:
            valid.append(citation); seen.add(fingerprint)
    return valid


class SelfRAGPipeline:
    def __init__(self, retriever: HybridRetriever, evaluator: Optional[Callable[..., Any]] = None,
                 rewriter: Optional[Callable[..., str]] = None,
                 generator: Optional[Callable[..., Any]] = None,
                 max_attempts: int = MAX_RETRIEVAL_ATTEMPTS):
        self.retriever, self.evaluator = retriever, evaluator
        self.rewriter, self.generator = rewriter, generator
        self.max_attempts = max(1, max_attempts)

    def run(self, original_query: str, metadata_filter: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        query, trace, best = original_query, [], []
        for attempt in range(1, self.max_attempts + 1):
            chunks = self.retriever.retrieve(query, metadata_filter=metadata_filter)
            if chunks: best = chunks
            try:
                raw = self.evaluator(query, chunks) if self.evaluator else {"relevant": bool(chunks), "confidence": .5}
            except Exception:
                raw = None
            evaluation = parse_evaluation(raw, bool(chunks))
            trace.append({"attempt_number": attempt, "query": query, "retrieved_chunks": len(chunks),
                          "relevant": evaluation.relevant, "confidence": evaluation.confidence})
            if evaluation.relevant:
                return self._answer(original_query, query, chunks, trace, False)
            if attempt == self.max_attempts:
                break
            rewritten = original_query
            if self.rewriter:
                try:
                    candidate = self.rewriter(query, evaluation.missing_information)
                    if isinstance(candidate, str) and candidate.strip(): rewritten = candidate.strip()
                except Exception:
                    pass
            trace[-1]["rewritten_query"] = rewritten
            query = rewritten
        return self._answer(original_query, query, best, trace, True)

    def _answer(self, original_query: str, query: str, chunks: list[RetrievedChunk], trace: list[dict[str, Any]], exhausted: bool) -> dict[str, Any]:
        if not chunks:
            return {"answer": "The available documents do not contain enough information to answer this question.",
                    "citations": [], "attempts": trace, "original_query": original_query, "rewritten_query": query}
        generated: Any = None
        if self.generator:
            try: generated = self.generator(query, chunks)
            except Exception: generated = None
        answer = generated.get("answer") if isinstance(generated, Mapping) else generated
        if not isinstance(answer, str) or not answer.strip():
            answer = "The retrieved documents provide relevant context, but no grounded answer could be generated."
        candidates = generated.get("citations", []) if isinstance(generated, Mapping) else []
        citations = validate_citations(candidates, chunks)
        if exhausted and not citations:
            answer = "The available documents do not contain enough verified information to answer this question."
        return {"answer": answer, "citations": citations, "attempts": trace,
                "original_query": original_query, "rewritten_query": query,
                "context": [chunk.as_dict() for chunk in chunks]}
