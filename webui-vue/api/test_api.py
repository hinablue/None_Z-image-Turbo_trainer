import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_system_info():
    print(f"Testing {BASE_URL}/api/system/info ...")
    try:
        resp = requests.get(f"{BASE_URL}/api/system/info")
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print("Response:", resp.json())
            # Check if pytorch is N/A
            data = resp.json()
            if data['pytorch'] == 'N/A' and data['cuda'] == 'N/A':
                print("[PASS] System info correctly reports N/A for missing torch")
            else:
                print(f"[WARN] System info reports: {data}")
        else:
            print("[FAIL] Failed to get system info")
            print(resp.text)
    except Exception as e:
        print(f"[FAIL] Connection error: {e}")

def test_generation_endpoint():
    print(f"\nTesting {BASE_URL}/api/generation/generate (Expect 503)...")
    try:
        resp = requests.post(f"{BASE_URL}/api/generation/generate", json={
            "prompt": "test",
            "model_type": "zimage"
        })
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 503:
            print("[PASS] Correctly returned 503 Service Unavailable")
            print("Response:", resp.json())
        else:
            print(f"[FAIL] Expected 503, got {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"[FAIL] Connection error: {e}")

if __name__ == "__main__":
    test_system_info()
    test_generation_endpoint()
