"""
FastAPI server for model serving.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure /backend/ is the cwd
import os

os.chdir(Path(__file__).parent.parent)

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


# Serve shared shaders to clients/backend
@app.get("/api/shader/{name}")
async def get_shared_shader(name: str):
    """Return a shared shader by name (whitelisted)."""
    from src.shaders import get_shader_path

    try:
        shader_path = get_shader_path(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Shader not found: {name}")

    return FileResponse(
        str(shader_path), media_type="text/plain", filename=shader_path.name
    )


# --- Mip pyramid artifacts (issue #88) ---
# The baked minimap pyramid lives at the repo root (written by
# scripts/bake_mandel_maps_gl.py). Serve it so the browser Player can load
# the minimaps and use contour_biased_step.
_MIP_PYRAMID_ROOT = Path(__file__).parent.parent.parent  # repo root


@app.get("/api/minimap/meta")
async def get_minimap_meta():
    """Return the mip pyramid metadata JSON."""
    meta = _MIP_PYRAMID_ROOT / "mandel_mips_meta.json"
    if not meta.exists():
        raise HTTPException(status_code=404, detail="Mip pyramid metadata not found")
    return FileResponse(str(meta), media_type="application/json")


@app.get("/api/minimap/field/{field}")
async def get_minimap_field(field: str):
    """Return a mip pyramid field binary (F = escape, S = shore proximity)."""
    if field not in ("F", "S"):
        raise HTTPException(status_code=404, detail="Unknown field")
    name = f"mandel_{field}_mips_f32.bin"
    path = _MIP_PYRAMID_ROOT / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Mip pyramid field not found: {name}")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename=name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
