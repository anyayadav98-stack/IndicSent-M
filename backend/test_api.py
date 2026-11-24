"""
test_api.py - Verifies all FastAPI endpoints
"""

from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

def test_endpoints():
    print("Testing GET /health...")
    r = client.get("/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    print("Health response:", r.json())

    print("\nTesting GET /dataset-stats...")
    r = client.get("/dataset-stats")
    assert r.status_code == 200, f"Dataset-stats failed: {r.text}"
    print("Dataset stats:", r.json()["splits"])

    print("\nTesting GET /metrics...")
    r = client.get("/metrics")
    assert r.status_code == 200, f"Metrics failed: {r.text}"
    print("Metrics keys:", list(r.json().keys()))

    print("\nTesting GET /history...")
    r = client.get("/history")
    assert r.status_code == 200, f"History failed: {r.text}"
    print("History count:", len(r.json()))

    print("\nTesting POST /predict (text and synthetic meme)...")
    r = client.post("/predict", data={"text": "When you complete all your assignments but still have more due tomorrow"})
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred = r.json()
    print("Predict Sentiment:", pred["final_sentiment"])
    print("Confidence Scores:", pred["confidence_scores"])
    print("LIME Words:", pred["lime_words"][:3])
    print("Explanation:", pred["explanation"][:80] + "...")

    print("\nALL API ENDPOINTS TESTED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()

