"""
FastAPI server for model serving.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="FractalSync Training API")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
