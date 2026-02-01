from fastapi.testclient import TestClient
from api.server import app
import json


def test_ping():
    client = TestClient(app)
    resp = client.get("/api/ping")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    assert "cwd" in data


def test_post_telemetry_creates_log(tmp_path, monkeypatch):
    # Run the server inside a temp working directory so logs are written to tmp_path / logs
    monkeypatch.chdir(tmp_path)

    client = TestClient(app)

    payload = {"dx": 0.1, "dy": -0.02, "note": "test"}
    resp = client.post("/api/telemetry", json=payload)
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"

    log = tmp_path / "logs" / "telemetry.log"
    assert log.exists()
    text = log.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["payload"] == payload
    assert entry["ts"].endswith("Z")

    # Test the GET helper that returns last N lines
    resp2 = client.get("/api/telemetry/last?n=10")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["count"] == 1
    assert data["entries"][0]["payload"] == payload
