#!/usr/bin/env python3
"""Quick test to verify the API endpoints respond correctly."""
import requests
import time

print("Starting test...")
print("Make sure the server is running: python -m uvicorn backend.main:app --reload --port 8000")
print()

# Wait a moment for server to be ready
time.sleep(1)

BASE_URL = "http://127.0.0.1:8000"

tests = [
    ("Health Check", f"{BASE_URL}/healthz"),
    ("Readiness Check", f"{BASE_URL}/readyz"),
    ("List Sessions", f"{BASE_URL}/api/sessions"),
    ("Frontend (index)", f"{BASE_URL}/"),
]

print("Running API tests...")
print("-" * 60)

for name, url in tests:
    try:
        response = requests.get(url, timeout=5)
        status = "✓ PASS" if response.status_code < 400 else f"✗ FAIL ({response.status_code})"
        print(f"{status} | {name}")
        print(f"      URL: {url}")
        if response.status_code >= 400:
            print(f"      Response: {response.text[:200]}")
    except requests.exceptions.ConnectionError:
        print(f"✗ FAIL | {name}")
        print(f"      URL: {url}")
        print(f"      Error: Cannot connect to server - is it running?")
    except Exception as e:
        print(f"✗ FAIL | {name}")
        print(f"      URL: {url}")
        print(f"      Error: {e}")
    print()

print("-" * 60)
print("Test complete!")
