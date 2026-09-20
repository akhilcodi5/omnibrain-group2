import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key.startswith("your_"):
    print("Please set a valid GEMINI_API_KEY in your .env file.")
    exit(1)

# Read and encode the image
image_path = "test_chart.png"
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found.")
    exit(1)

with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

question = "What data is shown in this chart? Please extract the key numbers."

for model in ["gemini-3.5-flash-lite"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": question},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1
        }
    }
    
    print(f"Sending request to {model}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("SUCCESS (200 OK)\n")
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                print("Response:")
                print(parts[0].get("text", ""))
        else:
            print("No content returned.")
    else:
        print(f"FAILED (Status Code: {response.status_code})")
        print(response.text)
