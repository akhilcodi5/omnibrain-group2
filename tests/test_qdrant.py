import asyncio
from storage.vector_store import get_vector_store

store = get_vector_store(in_memory=False)
info = store.client.get_collection(store.collection_name)
print(f"Points count: {info.points_count}")
