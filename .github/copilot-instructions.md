# AI Coding Agent Instructions (FractalSync)

These instructions make AI agents immediately productive in this repo.

## Big Picture
- Two-tier app: **backend** (Python/PyTorch + FastAPI) and **frontend** (React + Vite + ONNX.js).
- Goal: learn correlations from audio features → visual parameters to render **Julia sets** in real time.
- Training exports ONNX models for browser inference; API provides training lifecycle and model management.

## Repository Structure
- Backend core:
  - `backend/api/server.py`: FastAPI endpoints for training (`/api/train/start`, `/api/train/status`) and model export.
  - `backend/src/audio_features.py`: Librosa feature extraction and sliding-window flattening; 6 features × `window_frames` → input dim.
  - `backend/src/data_loader.py`: Audio dataset discovery + persistent `.npy` feature cache (`data/cache`).
  - `backend/src/model.py`: `AudioToVisualModel` MLP; default expects 60-dim (6 × 10) input.
  - `backend/src/trainer.py`: Training loop with correlation + smoothness losses; DataLoader batching; ONNX export helpers.
  - `backend/train.py`: CLI to run training without API.
- Frontend core:
  - `frontend/src/components/*`: audio capture, training panel, visualizer.
  - `frontend/src/lib/*`: ONNX inference, Julia renderer, feature helpers.

## Developer Workflows
- Backend setup (Windows):
  - `cd backend`
  - `pip install -r requirements.txt`
- Start API server (from `backend/`):
  - Preferred: `python api/server.py`
  - Alternative: `python -m api.server` (ensure CWD is `backend/`).
- Training via API:
  - `POST /api/train/start` JSON body: `{ "data_dir": "data/audio", "epochs": 1, "batch_size": 32, "learning_rate": 0.0001, "window_frames": 10, "include_delta": false, "include_delta_delta": false }`
  - Check status: `GET /api/train/status`
- CLI training:
  - `cd backend`
  - `python train.py --data-dir data/audio --epochs 100`
  - With velocity features: `python train.py --data-dir data/audio --epochs 100 --include-delta`
- Frontend:
  - `cd frontend && npm install && npm run dev`
  - Open `http://localhost:3000`

## Project Conventions
- Audio features are flattened windows: `[centroid, flux, rms, zcr, onset, rolloff] × window_frames` → `input_dim = 6 * window_frames`.
- **Velocity features**: Optional delta (first-order derivative) and delta-delta (second-order derivative) features can be enabled:
  - Base only: 6 features per frame → `input_dim = 6 * window_frames`
  - With delta: 12 features per frame → `input_dim = 12 * window_frames`
  - With delta-delta: 18 features per frame → `input_dim = 18 * window_frames`
- Feature caching: `.npy` files keyed by path + mtime + extractor config; auto-invalidates on config change.
- Tensors in trainer:
  - Handle DataLoader batches that return tuples/lists; extract the single tensor element to avoid extra dims.
  - Keep model outputs (`visual_params`) as tensors (do not `.item()`); stack metric lists and align lengths with batch.
- Error handling:
  - Model `forward()` validates input dim and raises with a clear message if mismatched.

## Integration Points
- Backend ↔ Frontend: exported ONNX model consumed by ONNX.js in `frontend/src/lib/modelInference.ts`.
- Visual metrics: `backend/src/visual_metrics.py` renders Julia sets and computes metrics used by correlation losses.
- Data: place audio under `backend/data/audio/`; cache in `backend/data/cache/` (ignored by git).

## Debugging Tips
- Common server run issues:
  - If `python -m api.server` fails, ensure you run it from `backend/`.
  - Use `python api/server.py` for simpler Windows path behavior.
- Shape mismatches usually originate from `window_frames`; confirm `extract_windowed_features()` returns `n × (6 * window_frames)`.
- Batch formatting: ensure DataLoader returns `(tensor,)`; extract `batch[0]` to keep shape `(batch_size, input_dim)`.
- Gradients: avoid `.item()` on tensors used in loss; stack and slice tensors to the minimal common length.

## Examples
- Start server then train:
  - Server: `cd backend && python api/server.py`
  - Train: `curl -X POST http://localhost:8000/api/train/start -H "Content-Type: application/json" -d '{"data_dir":"data/audio","epochs":1,"batch_size":32,"learning_rate":0.0001,"window_frames":10,"include_delta":false,"include_delta_delta":false}'`
  - Status: `curl http://localhost:8000/api/train/status`
- Train with velocity features:
  - `curl -X POST http://localhost:8000/api/train/start -H "Content-Type: application/json" -d '{"data_dir":"data/audio","epochs":100,"batch_size":32,"learning_rate":0.0001,"window_frames":10,"include_delta":true,"include_delta_delta":false}'`

If anything here seems off or incomplete (e.g., ports, paths, or training params), tell us and we’ll refine this doc.
