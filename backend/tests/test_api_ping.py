from fastapi.testclient import TestClient
from api.server import app


def test_ping():
    client = TestClient(app)
    resp = client.get("/api/ping")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    assert "cwd" in data
