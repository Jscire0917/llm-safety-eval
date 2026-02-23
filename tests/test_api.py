from fastapi.testclient import TestClient
from ai_eval.service.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.post("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    