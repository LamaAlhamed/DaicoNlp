"""Lab 7/capstone API contract smoke tests."""
from fastapi.testclient import TestClient
from bayan.serving.api import app

client = TestClient(app)


def test_health_endpoint_exists():
    response = client.get("/health")
    assert response.status_code == 200


def test_classify_endpoint_exists():
    response = client.post("/v1/classify", json={"text": "الخدمة ممتازة"})
    # During the starter phase this may fail until Lab 7 is implemented;
    # after Lab 7 it must become a successful contract.
    assert response.status_code == 200
