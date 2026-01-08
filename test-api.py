import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Get prompt from user instead of hardcoding
prompt = input("Enter your prompt: ")

# API configuration
url = "http://127.0.0.1:8000/generate"

headers = {
    "x-api-key": os.getenv("API_KEY"), 
    "Content-Type": "application/json"
}

# Send request with JSON body (matches the Pydantic model)
data = {
    "prompt": prompt,
    "model": "mistral",
    "temperature": 0.7
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    # Handle response
    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*60)
        print("Response:", result.get("response"))
        print("Model:", result.get("model"))
        print("Credits remaining:", result.get("credits_remaining"))
        print("="*60)
    else:
        print(f"Error {response.status_code}: {response.json()}")
        
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to server. Is it running?")
except requests.exceptions.Timeout:
    print("❌ Error: Request timed out")
except Exception as e:
    print(f"❌ Error: {e}")
