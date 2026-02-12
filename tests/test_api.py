from fastapi.testclient import TestClient
from ai_eval.service.api import app

client = TestClient(app)

def test_health_endpoint():
    r = client.post("/health")
    assert r.status_code == 200
