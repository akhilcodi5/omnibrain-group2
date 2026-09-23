from storage.vector_store import get_vector_store
from agents.search_agent import SearchAgent

def test_search_agent_execution():
    """Verify search agent retrieval execution."""
    store = get_vector_store(in_memory=True)
    agent = SearchAgent(vector_store=store)
    res = agent.execute_search("Management claims in the MD&A section that Services net sales increased in 2023.", top_k=5)
    assert "retrieved_docs" in res
    assert "citations" in res

