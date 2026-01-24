import io
import zipfile
import json
from pathlib import Path
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


def make_sample_run(tmp_path, run_id="sample_run"):
    run_dir = tmp_path / run_id
    proxy_dir = run_dir / "proxy_frames"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    # records.ndjson with metadata
    with open(run_dir / "records.ndjson", "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": True, "metadata": {"test": True}}) + "\n")
        f.write(json.dumps({"t": 0, "c": [-0.7, 0.27], "notes": "ok"}) + "\n")
    # proxy frame
    with open(proxy_dir / "000000.png", "wb") as f:
        f.write(b"PNGDATA")
    # Create zip
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as zf:
        for p in run_dir.rglob("*"):
            zf.write(p, arcname=str(p.relative_to(run_dir)))
    zip_bytes.seek(0)
    return zip_bytes


def test_flight_recorder_upload_download(tmp_path):
    zip_bytes = make_sample_run(tmp_path)

    files = {"file": ("sample_run.zip", zip_bytes, "application/zip")}
    resp = client.post("/api/flight_recorder/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "sample_run"

    # List should include the run
    resp = client.get("/api/flight_recorder/list")
    assert resp.status_code == 200
    runs = resp.json().get("runs", [])
    assert any(r["run_id"] == "sample_run" for r in runs)

    # Download
    resp = client.get("/api/flight_recorder/sample_run/download")
    assert resp.status_code == 200
    content = resp.content
    z = zipfile.ZipFile(io.BytesIO(content))
    names = z.namelist()
    assert "records.ndjson" in names
    assert "proxy_frames/000000.png" in names
