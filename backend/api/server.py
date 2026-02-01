"""
FastAPI server for model serving.
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import datetime

# Configure basic logging for server diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FractalSync Training API")

# Log current working directory on import/startup to help diagnose which process is running
logger.info("Server imported. CWD=%s", Path.cwd())


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/model/latest")
async def get_latest_model():
    """Download the latest ONNX model."""
    # Prioritize checkpoints (latest trained models)
    models_dir = Path("checkpoints")
    if not models_dir.exists():
        models_dir = Path("models")  # Fallback to legacy location

    onnx_files = list(models_dir.glob("*.onnx"))

    if not onnx_files:
        raise HTTPException(status_code=404, detail="No models found")

    latest_model = max(onnx_files, key=lambda p: p.stat().st_mtime)

    return FileResponse(
        str(latest_model),
        media_type="application/octet-stream",
        filename=latest_model.name,
    )


@app.get("/api/model/metadata")
async def get_model_metadata():
    """Get metadata for the latest model."""
    # Prioritize checkpoints (latest trained models)
    models_dir = Path("checkpoints")
    if not models_dir.exists():
        models_dir = Path("models")  # Fallback to legacy location

    metadata_files = list(models_dir.glob("*_metadata.json"))

    if not metadata_files:
        raise HTTPException(status_code=404, detail="No model metadata found")

    latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)

    with open(latest_metadata, "r") as f:
        metadata = json.load(f)

    return metadata


@app.get("/api/ping")
async def get_ping():
    """Diagnostic ping that returns server readiness and current working directory."""
    return {"ok": True, "cwd": str(Path.cwd())}


@app.post("/api/telemetry")
async def post_telemetry(request: Request):
    """Append telemetry JSON payload to logs/telemetry.log with timestamp and client info.

    Reads the JSON body robustly and logs an INFO record on receipt. This makes it easier
    to debug when the frontend isn't sending telemetry or when writes fail.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.warning("Telemetry: failed to decode JSON body: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "telemetry.log"

    entry = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "client": request.client.host if request.client else None,
        "payload": payload,
    }

    logger.info(
        "Telemetry received: client=%s size=%d",
        entry["client"],
        len(json.dumps(payload)),
    )

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.exception("Failed to write telemetry to %s: %s", log_file, e)
        raise HTTPException(status_code=500, detail=f"Failed to write telemetry: {e}")

    return {"status": "ok"}


@app.get("/api/telemetry/last")
async def get_last_telemetry(n: int = 50):
    """Return the last `n` JSON telemetry lines as a JSON list (debug helper)."""
    log_file = Path("logs") / "telemetry.log"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="No telemetry log found")

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.exception("Failed to read telemetry log: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to read telemetry: {e}")

    # Return up to n last entries parsed as JSON
    last_lines = lines[-n:]
    entries = []
    for line in last_lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            # skip malformed lines
            continue

    return {"count": len(entries), "entries": entries}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
