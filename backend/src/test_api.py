import os
import sys

# Ensure backend/src is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_text_search_removed():
    # Verify that the text search route has been removed and returns 404
    response = client.post("/api/search/text", json={"query": "test shoe"})
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    print("[OK] Verified that text search endpoint is removed.")

def test_catalog_stats():
    # Verify stats endpoint returns correct keys and is_reindexing state
    response = client.get("/api/catalog/stats")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "is_reindexing" in data
    assert "catalog_size" in data
    assert "is_out_of_sync" in data
    print("[OK] Verified catalog stats endpoint is operational.")

def test_reindex_background():
    # Verify trigger reindexing initiates successfully in background
    response = client.post("/api/reindex")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] in ("success", "warning")
    print("[OK] Verified reindexing endpoint returns background task confirmation.")

if __name__ == "__main__":
    try:
        test_text_search_removed()
        test_catalog_stats()
        test_reindex_background()
        print("\nAll API verification tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
