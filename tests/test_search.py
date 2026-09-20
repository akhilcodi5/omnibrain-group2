import asyncio
from storage.vector_store import get_vector_store
from agents.search_agent import SearchAgent

async def main():
    store = get_vector_store(in_memory=False)
    agent = SearchAgent(vector_store=store)
    res = agent.execute_search("Management claims in the MD&A section that Services net sales increased in 2023.", top_k=5)
    
    print(f"Retrieved docs count: {len(res['retrieved_docs'])}")
    for d in res['retrieved_docs']:
        print(f"Page: {d.get('page_number')} | Metadata: {d.get('metadata')}")

asyncio.run(main())
