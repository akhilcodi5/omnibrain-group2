from fastapi.testclient import TestClient
from app.main import app

def test_optimization_chat_query():
    """Test optimized multi-modal chat query route."""
    client = TestClient(app)
    payload = {
        "query": "Net sales by reportable segment for iPhone and Services revenue",
        "pdf_name": "Apple 10-K PDF 2023",
        "referenced_images": [
            "chart_Apple 10-K PDF 2023_p24_1_8071c5.png",
            "chart_Apple 10-K PDF 2023_p30_1_d05266.png",
            "chart_Apple 10-K PDF 2023_p1_1_xxx.png"
        ]
    }
    response = client.post("/api/v1/chat/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "final_response" in data
    assert "visual_evidence" in data

