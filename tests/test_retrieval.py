import unittest

from agent.retrieval import HybridRetriever, SelfRAGPipeline, validate_citations
from agent.rag import rag_node


def chunk(doc, chunk_id, page, text="evidence", score=1):
    return {"document_id": doc, "document_name": doc + ".pdf", "chunk_id": chunk_id,
            "page_number": page, "source": "upload", "text": text, "score": score,
            "metadata": {"tenant": "t1"}}


class RetrievalTests(unittest.TestCase):
    def test_hybrid_fusion_deduplicates_and_preserves_page_metadata(self):
        retriever = HybridRetriever(lambda q, **_: [chunk("a", "1", 4), chunk("b", "2", 17)],
                                    lambda q, **_: [chunk("a", "1", 4), chunk("a", "3", 8)])
        results = retriever.retrieve("risk")
        self.assertEqual(len(results), 3)
        duplicate = next(result for result in results if result.chunk_id == "1")
        self.assertEqual(duplicate.page_number, 4)
        self.assertEqual(duplicate.retrieval_method, "keyword+semantic")
        self.assertEqual(duplicate.metadata["tenant"], "t1")

    def test_one_retrieval_failure_does_not_hide_other_method(self):
        def broken(*_args, **_kwargs): raise RuntimeError("offline")
        results = HybridRetriever(broken, lambda q, **_: [chunk("a", "1", 4)]).retrieve("risk")
        self.assertEqual([result.chunk_id for result in results], ["1"])

    def test_successful_rewrite_retry_tracks_original_and_rewrite(self):
        calls = []
        def semantic(query, **_):
            calls.append(query)
            return [] if query == "ambiguous" else [chunk("a", "1", 4)]
        pipeline = SelfRAGPipeline(HybridRetriever(semantic),
            evaluator=lambda q, c: {"relevant": bool(c), "confidence": .9, "missing_information": "topic"},
            rewriter=lambda q, missing: "specific topic",
            generator=lambda q, c: {"answer": "Grounded answer", "citations": [{"document_id": "a", "chunk_id": "1", "page_number": 4}]})
        result = pipeline.run("ambiguous")
        self.assertEqual(calls, ["ambiguous", "specific topic"])
        self.assertEqual(result["citations"][0]["page_number"], 4)
        self.assertEqual(result["attempts"][0]["rewritten_query"], "specific topic")

    def test_maximum_retry_empty_retrieval_and_rewrite_failure(self):
        pipeline = SelfRAGPipeline(HybridRetriever(lambda q, **_: []), evaluator=lambda q, c: "bad",
            rewriter=lambda q, m: (_ for _ in ()).throw(RuntimeError("bad")), max_attempts=3)
        result = pipeline.run("unknown")
        self.assertEqual(len(result["attempts"]), 3)
        self.assertIn("do not contain enough", result["answer"])

    def test_invalid_citations_are_dropped_and_multiple_pages_remain(self):
        context = HybridRetriever(lambda q, **_: [chunk("a", "1", 4), chunk("a", "2", 8), chunk("b", "3", 17)]).retrieve("q")
        citations = validate_citations([
            {"document_id": "a", "chunk_id": "1", "page_number": 4},
            {"document_id": "a", "chunk_id": "2", "page_number": 99},
            {"document_id": "b", "chunk_id": "3", "page_number": 17},
            {"document_id": "missing", "chunk_id": "x", "page_number": 1},
        ], context)
        self.assertEqual([(c["document_id"], c["page_number"]) for c in citations], [("a", 4), ("b", 17)])

    def test_missing_page_is_not_invented(self):
        context = HybridRetriever(lambda q, **_: [chunk("a", "1", None)]).retrieve("q")
        citations = validate_citations([{"document_id": "a", "chunk_id": "1"}], context)
        self.assertEqual(citations[0]["page_number"], None)

    def test_malformed_evaluator_is_safe_and_rag_node_is_connected(self):
        pipeline = SelfRAGPipeline(HybridRetriever(lambda q, **_: [chunk("a", "1", 4)]),
            evaluator=lambda q, c: ["not", "an", "object"])
        state = rag_node({"user_query": "q", "rag_pipeline": pipeline})
        self.assertEqual(state["retrieved_context"][0]["page_number"], 4)
        self.assertEqual(state["citations"], [])


if __name__ == "__main__":
    unittest.main()
