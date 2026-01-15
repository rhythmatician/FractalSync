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
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.audio_features import AudioFeatureExtractor  # noqa: E402
from src.data_loader import AudioDataset  # noqa: E402
from src.export_model import load_checkpoint_and_export  # noqa: E402
from src.model import AudioToVisualModel  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.visual_metrics import VisualMetrics  # noqa: E402

app = FastAPI(title="FractalSync Training API")


class TrainingRequest(BaseModel):
    data_dir: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    window_frames: int = 10


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
        feature_extractor = AudioFeatureExtractor()
        visual_metrics = VisualMetrics()
        model = AudioToVisualModel(window_frames=request.window_frames)

        trainer = Trainer(
            model=model,
            feature_extractor=feature_extractor,
            visual_metrics=visual_metrics,
            learning_rate=request.learning_rate,
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
            tensor_dataset, batch_size=request.batch_size, shuffle=True
        )

        # Training loop
        for epoch in range(request.epochs):
            training_state.current_epoch = epoch + 1
            training_state.progress = (epoch + 1) / request.epochs

            avg_losses = trainer.train_epoch(dataloader, epoch)

            # Update history
            for key, value in avg_losses.items():
                trainer.history[key].append(value)

            training_state.loss_history.append({"epoch": epoch + 1, **avg_losses})

            # Save checkpoint periodically (and always on last epoch)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == request.epochs:
                save_dir = "checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                trainer.save_checkpoint(save_dir, epoch + 1)

            # Yield control to event loop
            await asyncio.sleep(0)

        # Export to ONNX
        checkpoint_files = list(Path("checkpoints").glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda p: p.stat().st_mtime,
            )
            load_checkpoint_and_export(
                str(latest_checkpoint),
                output_dir="models",
                window_frames=request.window_frames,
            )
        else:
            logging.warning("No checkpoints found to export")

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
    models_dir = Path("models")
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
    models_dir = Path("models")
    metadata_files = list(models_dir.glob("*_metadata.json"))

    if not metadata_files:
        raise HTTPException(status_code=404, detail="No model metadata found")

    latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)

    with open(latest_metadata, "r") as f:
        metadata = json.load(f)

    return metadata


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
