# Quick Start After Implementation

## ✅ All Acceptance Criteria Met

The refactored system is now fully functional. Here's how to use it:

---

## 1. Run Training

```bash
cd backend
python train.py --epochs 100 --no-gpu-rendering
```

**Expected Output:**
- Progress bars for epochs and batches
- Checkpoint saves every 10 epochs
- Final ONNX model export
- Files in `backend/checkpoints/`:
  - `model_orbit_control_YYYYMMDD_HHMMSS.onnx`
  - `model_orbit_control_YYYYMMDD_HHMMSS.onnx_metadata.json`
  - `checkpoint_epoch_*.pt`
  - `training_history.json`

---

## 2. Start Backend API

```bash
cd backend
python api/server.py
```

**Server will run at:** `http://localhost:8000`

**Test endpoint:**
```bash
curl http://localhost:8000/api/train/status
```

---

## 3. Start Frontend

```bash
cd frontend
npm install  # first time only
npm run dev
```

**Dev server will run at:** `http://localhost:3001` (or 3000 if available)

---

## 4. Load Model in Browser

1. Open http://localhost:3001
2. Find the model upload UI
3. Upload: `backend/checkpoints/model_orbit_control_*.onnx`
4. Model should load successfully with metadata

---

## Quick Validation Test

Run the automated test script:

```bash
.\test_acceptance.ps1
```

This will verify all three acceptance criteria.

---

## What Changed

### Fixed: Feature Extraction Hanging
The Rust-based feature extractor had a bug causing it to hang indefinitely. A Python fallback using librosa was implemented as a workaround.

**Files Added/Modified:**
- ✅ `backend/src/python_feature_extractor.py` (new)
- ✅ `backend/src/runtime_core_bridge.py` (modified to use Python fallback)

**Impact:** Training now works reliably. Performance is slightly slower than Rust would be, but acceptable for the current use case.

---

## Known Issues

### Rust Feature Extractor Hangs
**Status:** Known bug, workaround in place

**Symptom:** Calling `runtime_core.FeatureExtractor.extract_windowed_features()` hangs indefinitely

**Workaround:** Python fallback extractor automatically used

**Future Work:** Debug and fix Rust implementation

---

## Troubleshooting

### Training Fails
- Ensure audio files in `backend/data/audio/`
- Supported formats: WAV, MP3, FLAC, OGG, M4A
- Check logs in `backend/logs/train.log`

### API Won't Start
- Check port 8000 is not in use
- Verify dependencies: `pip install -r backend/requirements.txt`

### Frontend Won't Build
- Run `npm install` in frontend directory
- Check Node.js version (LTS recommended)

---

## Documentation

- **Full Details:** See `IMPLEMENTATION_SUMMARY.md`
- **Training Logs:** `backend/logs/train.log`
- **Training History:** `backend/checkpoints/training_history.json`
- **Model Metadata:** `backend/checkpoints/*.onnx_metadata.json`

---

**Last Updated:** January 20, 2026  
**Status:** ✅ Ready for use
