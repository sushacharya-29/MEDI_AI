import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GROK_API_KEY")

url = "http://127.0.0.1:8000/api/v2/diagnose"

payload = {
    "patient_data": {
        "patient_id": "test123",
        "name": "John Doe",
        "age": 30,
        "gender": "male",
        "symptoms": "fever, headache, nausea"
    }
}


headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
