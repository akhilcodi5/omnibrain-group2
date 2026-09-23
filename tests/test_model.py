import os
import pytest

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Manual test requiring GEMINI_API_KEY")
def test_gemini_model_connectivity():
    """Verify Gemini API connectivity for flash models."""
    import requests
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
    response = requests.post(url, json=payload)
    assert response.status_code in (200, 429, 503)

