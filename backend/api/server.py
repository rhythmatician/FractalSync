"""
FastAPI server for training management and model serving.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import logging

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import zipfile
import tempfile
import shutil
from pathlib import Path as _Path  # local alias to avoid shadowing earlier Path import
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.data_loader import AudioDataset  # noqa: E402
from src.control_model import AudioToControlModel  # noqa: E402
from src.control_trainer import ControlTrainer  # noqa: E402
from src.visual_metrics import VisualMetrics  # noqa: E402
from src.runtime_core_bridge import make_feature_extractor  # noqa: E402

# GPU rendering optimization imports
try:
    from src.julia_gpu import GPUJuliaRenderer

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

app = FastAPI(title="FractalSync Training API")


class TrainingRequest(BaseModel):
    data_dir: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    window_frames: int = 10
    use_curriculum: bool = True
    curriculum_weight: float = 1.0
    curriculum_decay: float = 0.95
    damping_factor: float = 0.95
    speed_scale: float = 0.1
    use_gpu_rendering: bool = True
    julia_resolution: int = 64
    julia_max_iter: int = 50
    num_workers: int = 4


class TrainingStatus(BaseModel):
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss_history: List[dict]
    error: Optional[str] = None


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = TrainingStatus(
    status="idle",  # idle, training, completed, error
    progress=0,
    current_epoch=0,
    total_epochs=0,
    loss_history=[],
    error=None,
)

training_task: Optional[asyncio.Task] = None


@app.get("/")
async def root():
    return {"message": "FractalSync Training API"}


@app.post("/api/train/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training job."""
    global training_state, training_task

    if training_state.status == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")

    # Reset state
    training_state = TrainingStatus(
        status="training",
        progress=0,
        current_epoch=0,
        total_epochs=request.epochs,
        loss_history=[],
        error=None,
    )

    # Start training in background
    training_task = asyncio.create_task(train_model_async(request))

    return {"message": "Training started", "status": "training"}


async def train_model_async(request: TrainingRequest):
    """Async wrapper for training."""
    global training_state

    try:
        # Initialize components
        feature_extractor = make_feature_extractor()
        visual_metrics = VisualMetrics()

        # Initialize GPU renderer if requested and available
        julia_renderer = None
        if request.use_gpu_rendering and GPU_AVAILABLE:
            try:
                julia_renderer = GPUJuliaRenderer(
                    width=request.julia_resolution,
                    height=request.julia_resolution,
                )
                logging.info("GPU renderer initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU renderer: {e}")

        # Initialize control model
        model = AudioToControlModel(
            window_frames=request.window_frames,
            k_bands=6,  # Default to 6 band gates
        )

        # Initialize control trainer
        trainer = ControlTrainer(
            model=model,
            feature_extractor=feature_extractor,
            visual_metrics=visual_metrics,
            learning_rate=request.learning_rate,
            use_curriculum=request.use_curriculum,
            curriculum_weight=request.curriculum_weight,
            julia_renderer=julia_renderer,
            julia_resolution=request.julia_resolution,
            julia_max_iter=request.julia_max_iter,
            num_workers=request.num_workers,
        )

        # Load dataset
        dataset = AudioDataset(
            data_dir=request.data_dir,
            feature_extractor=feature_extractor,
            window_frames=request.window_frames,
        )

        # Custom training loop with progress updates
        all_features = dataset.load_all_features()
        feature_extractor.compute_normalization_stats(all_features)

        normalized_features = [
            feature_extractor.normalize_features(f) for f in all_features
        ]

        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        all_features_tensor = torch.tensor(
            np.concatenate(normalized_features, axis=0), dtype=torch.float32
        )

        tensor_dataset = TensorDataset(all_features_tensor)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=request.batch_size,
            shuffle=True,
            num_workers=(
                request.num_workers if not request.use_gpu_rendering else 0
            ),  # GPU rendering conflicts with workers
        )

        # Training loop
        for epoch in range(request.epochs):
            avg_losses = trainer.train_epoch(
                dataloader,
                epoch,
                curriculum_decay=request.curriculum_decay,
            )

            # Update state
            training_state.current_epoch = epoch + 1
            training_state.progress = (epoch + 1) / request.epochs
            training_state.loss_history.append({"epoch": epoch + 1, **avg_losses})

            # Update history
            for key, value in avg_losses.items():
                trainer.history[key].append(value)

            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0 or (epoch + 1) == request.epochs:
                save_dir = "checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                trainer.save_checkpoint(save_dir, epoch + 1)

            await asyncio.sleep(0)  # Yield to event loop

        # Export to ONNX
        # TEMP: Disabled because load_checkpoint_and_export doesn't exist in current export_model.py
        # checkpoint_files = list(Path("checkpoints").glob("checkpoint_epoch_*.pt"))
        # if checkpoint_files:
        #     latest_checkpoint = max(
        #         checkpoint_files,
        #         key=lambda p: p.stat().st_mtime,
        #     )
        #     load_checkpoint_and_export(
        #         str(latest_checkpoint),
        #         output_dir="models",
        #         window_frames=request.window_frames,
        #     )
        # else:
        #     logging.warning("No checkpoints found to export")
        logging.warning("ONNX export after training is temporarily disabled")

        training_state.status = "completed"
        training_state.progress = 1.0

    except Exception as e:
        training_state.status = "error"
        training_state.error = str(e)
        logging.info(f"Training error: {e}")


@app.get("/api/train/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return training_state


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


# Flight recorder API
@app.get("/api/flight_recorder/list")
async def list_flight_runs():
    """List available flight recorder runs with summary metadata."""
    runs_dir = Path("logs/flight_recorder")
    if not runs_dir.exists():
        return {"runs": []}

    runs = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        record_file = run_dir / "records.ndjson"
        meta = None
        try:
            if record_file.exists():
                with open(record_file, "r", encoding="utf-8") as f:
                    first = f.readline().strip()
                    if first:
                        parsed = json.loads(first)
                        if parsed.get("_meta"):
                            meta = parsed.get("metadata")
        except Exception:
            meta = None

        runs.append({"run_id": run_dir.name, "metadata": meta})

    return {"runs": runs}


@app.get("/api/flight_recorder/{run_id}/download")
async def download_flight_run(run_id: str):
    """Download a flight recorder run as a ZIP archive."""
    runs_dir = Path("logs/flight_recorder")
    run_dir = runs_dir / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    # Create a temporary zip file
    tmp_dir = tempfile.mkdtemp()
    zip_path = Path(tmp_dir) / f"{run_id}.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in run_dir.rglob("*"):
                rel = p.relative_to(run_dir)
                zf.write(p, arcname=str(rel))

        return FileResponse(
            str(zip_path), media_type="application/zip", filename=f"{run_id}.zip"
        )
    finally:
        # Clean up the tmp_dir after the response has been sent by the server
        pass


@app.post("/api/flight_recorder/upload")
async def upload_flight_run(file: UploadFile = File(...)):
    """Upload a flight run ZIP archive. Extracts into logs/flight_recorder/<run_id>.

    The ZIP is expected to contain `records.ndjson` at the root and optional `proxy_frames/`.
    """
    runs_dir = Path("logs/flight_recorder")
    runs_dir.mkdir(parents=True, exist_ok=True)

    assert file.filename
    # Determine run id from filename (strip suffix) or use timestamp
    run_id = Path(file.filename).stem
    dest_dir = runs_dir / run_id
    if dest_dir.exists():
        raise HTTPException(status_code=400, detail="Run with that id already exists")

    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_zip = Path(tmp_dir) / file.filename
        with open(tmp_zip, "wb") as f:
            content = await file.read()
            f.write(content)

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(dest_dir)

        # Basic validation
        record_file = dest_dir / "records.ndjson"
        if not record_file.exists():
            raise HTTPException(
                status_code=400, detail="Missing records.ndjson in uploaded run"
            )

        return {"message": "Run uploaded", "run_id": run_id}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a trained model file."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    assert file.filename
    file_path = models_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"message": f"Model uploaded: {file.filename}"}


@app.post("/api/audio/upload")
async def upload_audio_files(files: List[UploadFile] = File(...)):
    """Upload audio files for training."""
    upload_dir = Path("data/audio")
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    for file in files:
        assert file.filename
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded_files.append(file.filename)

    return {"message": f"Uploaded {len(uploaded_files)} files", "files": uploaded_files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
