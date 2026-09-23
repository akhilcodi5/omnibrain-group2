import pytest
from app.api.routes_chat import query_agent_orchestrator, ChatQueryRequest

@pytest.mark.asyncio
async def test_mock_pipeline_query():
    """Verify end-to-end query orchestrator execution with visual references."""
    req = ChatQueryRequest(
        query="Extract the 'Net sales by reportable segment' table. Compute the year-over-year percentage change for iPhone and Services revenue between 2022 and 2023.",
        pdf_name="Apple 10-K PDF 2023",
        referenced_images=[
            "chart_Apple 10-K PDF 2023_p24_1_9f14e8.png",
            "table_Apple 10-K PDF 2023_p24_1_03efef.png"
        ]
    )
    resp = await query_agent_orchestrator(req)
    assert resp is not None
    assert resp.final_response is not None

