from storage.vector_store import get_vector_store

def test_qdrant_collection_status():
    """Verify Qdrant collection accessibility."""
    store = get_vector_store(in_memory=True)
    assert store.collection_name is not None
    info = store.client.get_collection(store.collection_name)
    assert info is not None

