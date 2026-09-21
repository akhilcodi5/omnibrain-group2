import os
import pytest

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY") or not os.path.exists("test_chart.png"), reason="Manual test requiring test_chart.png and GEMINI_API_KEY")
def test_gemini_api_direct():
    """Manual direct API verification with sample chart."""
    import base64
    import requests
    api_key = os.getenv("GEMINI_API_KEY")
    with open("test_chart.png", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": "Extract chart data"},
                {"inline_data": {"mime_type": "image/png", "data": base64_image}}
            ]
        }]
    }
    response = requests.post(url, json=payload)
    if response.status_code in (429, 503):
        pytest.skip(f"Gemini API temporarily unavailable: HTTP {response.status_code}")
    assert response.status_code == 200

