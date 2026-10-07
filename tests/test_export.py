"""Tests for Chat and Memorandum Export endpoints (PDF and Markdown)."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_export_chat_markdown():
    """Verify chat export to Markdown returns 200 with text/markdown."""
    client = TestClient(app)
    payload = {
        "format": "md",
        "chat_history": [
            {
                "query": "Extract Q3 revenue trajectory",
                "response": "Revenue increased by 15% YoY.",
                "citations": [
                    {
                        "figure_title": "Revenue Chart P1",
                        "relevance_score": 0.95,
                        "text": "Total GAAP revenue $35.1B",
                    }
                ],
            }
        ],
    }

    res = client.post("/api/v1/export/chat", json=payload)
    assert res.status_code == 200
    assert "text/markdown" in res.headers.get("content-type", "")
    assert b"Extract Q3 revenue trajectory" in res.content
    assert b"Revenue increased by 15% YoY" in res.content
    assert b"Revenue Chart P1" in res.content


def test_export_chat_pdf():
    """Verify chat export to PDF generates valid PDF binary without 501 Not Implemented."""
    client = TestClient(app)
    payload = {
        "format": "pdf",
        "chat_history": [
            {
                "query": "Analyze quarterly revenue bar chart",
                "response": "Fiscal year revenue growth demonstrated strong momentum.",
                "citations": [
                    {
                        "figure_title": "Quarterly Revenue Bar Chart (Page 1)",
                        "relevance_score": 0.98,
                        "text": "Total revenue reported across four fiscal quarters.",
                    }
                ],
            }
        ],
    }

    res = client.post("/api/v1/export/chat", json=payload)
    assert res.status_code == 200
    assert "application/pdf" in res.headers.get("content-type", "")
    assert res.content.startswith(b"%PDF")
    assert len(res.content) > 500


def test_export_invalid_format():
    """Verify invalid export format returns 400 Bad Request."""
    client = TestClient(app)
    res = client.post("/api/v1/export/chat", json={"format": "docx", "chat_history": []})
    assert res.status_code == 400
