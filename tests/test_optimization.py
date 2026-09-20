import requests
import json

url = "http://localhost:8000/api/v1/chat/query"
payload = {
    "query": "Net sales by reportable segment for iPhone and Services revenue",
    "pdf_name": "Apple 10-K PDF 2023",
    "referenced_images": [
        "chart_Apple 10-K PDF 2023_p24_1_8071c5.png",
        "chart_Apple 10-K PDF 2023_p30_1_d05266.png",
        "chart_Apple 10-K PDF 2023_p1_1_xxx.png"
    ]
}

print("Sending request...")
response = requests.post(url, json=payload)
if response.status_code == 200:
    print("Success!")
    data = response.json()
    print("Execution time:", data.get("execution_time_seconds"))
    # Check visual evidence
    print("Visual Evidence count:", len(data.get("visual_evidence", [])))
else:
    print("Failed:", response.status_code, response.text)
