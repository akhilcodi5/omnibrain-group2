import pytest
import os
import json
from unittest.mock import patch, MagicMock

# Local imports
from agents.supervisor import create_supervisor_graph

@pytest.mark.asyncio
async def test_apple_10k_math_anomalies_and_sql():
    """Test that the end-to-end multi-modal RAG does not hallucinate YoY percentages."""
    
    from dotenv import load_dotenv
    load_dotenv()

    # Check if API key is present
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set. Skipping real API evaluation.")
        
    initial_state = {
        "messages": [],
        "query": "Extract the 'Net sales by reportable segment' table. Compute the year-over-year percentage change for iPhone and Services revenue between 2022 and 2023. Did Services growth offset the decline in iPhone sales?",
        "next_agent": None,
        "retrieved_docs": [],
        "visual_evidence": [],
        "referenced_images": ["storage/extracted_images/table_Apple 10-K PDF 2023_p25_1_e53c12.png"],
        "sql_query": None,
        "sql_results": None,
        "is_grounded": True,
        "retrieval_confidence": 1.0,
        "iteration_count": 0,
        "final_response": None,
        "citations": [],
    }

    from langchain_core.messages import HumanMessage
    initial_state["messages"] = [HumanMessage(content=initial_state["query"])]

    # Run the graph
    try:
        graph = create_supervisor_graph()
        final_state = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "test_thread_10k"}}
        )
        
        memo = final_state.get("final_response", "")
        
        # Verify the pipeline ran and generated output
        assert len(memo) > 100, "Memo should be generated with substantial content"
        
        # Look for crazy anomalies that we saw earlier (like -85.36% or -1250%)
        # A smart AI math engine shouldn't be comparing iPhone vs Mac on the same row.
        assert "-85.36%" not in memo, "The pipeline still hallucinates iPhone vs Mac math anomalies!"
        assert "-1250" not in memo, "The pipeline still hallucinates math anomalies!"
        
        # Verify Text-to-SQL logic execution
        sql_query = final_state.get("sql_query")
        # Note: Depending on routing, the SQL agent might not trigger for this exact query, 
        # but if it does, it should be a SELECT statement.
        if sql_query:
            assert "SELECT" in sql_query.upper(), "Generated SQL must contain a SELECT statement"
            
    except Exception as e:
        pytest.fail(f"Graph invocation failed: {e}")
