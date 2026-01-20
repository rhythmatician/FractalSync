# Quick Start: Testing the Orbit Synthesizer Implementation

## Prerequisites
- ✅ Backend: Python 3.8+, PyTorch, ONNX
- ✅ Frontend: Node.js 16+, npm
- ✅ Audio data in `backend/data/audio/`

## Step 1: Train an Orbit Model

```bash
cd backend

# Basic training (100 epochs)
python train_orbit.py --data-dir data/audio --epochs 100

# With curriculum learning (recommended)
python train_orbit.py \
    --data-dir data/audio \
    --epochs 100 \
    --batch-size 32 \
    --use-curriculum \
    --k-bands 6

# Quick test (1 epoch)
python train_orbit.py --data-dir data/audio --epochs 1
```

**Output:**
- Model: `backend/checkpoints/model_orbit_control_TIMESTAMP.onnx`
- Metadata: `backend/checkpoints/model_orbit_control_TIMESTAMP.onnx_metadata.json`

## Step 2: Copy Model to Frontend

```bash
# Copy both files to frontend public directory
cp backend/checkpoints/model_orbit_control_*.onnx frontend/public/model.onnx
cp backend/checkpoints/model_orbit_control_*.onnx_metadata.json frontend/public/model.onnx_metadata.json
```

## Step 3: Start Frontend

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Access:** Open browser to `http://localhost:3000`

## Step 4: Test in Browser

### Check Console
Look for:
```
[ModelInference] Loaded orbit-based control model
```

### Load Audio
1. Click "Audio Capture" or load file
2. Watch for visual updates
3. Check console logs every ~60 frames

### Verify Control Signals
Console should show:
```javascript
{
  julia: [0.123, -0.456],  // c(t) from synthesizer
  color: [0.789, 0.654, 0.543],
  zoom: 2.500,
  speed: 0.567,
  modelType: 'orbit_control'
}
```

## Expected Behavior

### Orbit Model (New)
- Julia parameter (c) evolves smoothly along orbit
- Deterministic: same audio → same trajectory
- Control signals visible in console
- Model type: `orbit_control`

### Legacy Model (Old)
- Julia parameter varies frame-by-frame
- Post-processing active
- Model type: `legacy`

## Troubleshooting

### Model Not Loading
```javascript
// Console error: "Model not found"
// Solution: Check model.onnx exists in frontend/public/
```

### Wrong Model Type
```javascript
// Console shows: "Loaded legacy visual parameter model"
// Check: metadata.model_type should be "orbit_control"
// Solution: Re-export with correct metadata
```

### TypeScript Errors
```bash
cd frontend
npm run build

# Should show: "✓ built in X.XXs"
# If errors: Check TypeScript version and deps
```

### ONNX Runtime Errors
```javascript
// Console error: "WASM paths not found"
// Solution: Ensure onnxruntime-web is installed
npm install onnxruntime-web@1.14.0
```

## Verification Checklist

### Backend ✅
- [ ] `train_orbit.py` runs without errors
- [ ] ONNX file generated in `checkpoints/`
- [ ] Metadata JSON has `model_type: "orbit_control"`
- [ ] Metadata includes `k_bands` field

### Frontend ✅
- [ ] `npm run build` succeeds
- [ ] Model files in `public/` directory
- [ ] Console shows "Loaded orbit-based control model"
- [ ] Visual output updates with audio

### Visual Quality ✅
- [ ] Julia set renders correctly
- [ ] Smooth transitions (no jitter)
- [ ] Color responds to audio
- [ ] Performance: 60 FPS maintained

## Performance Monitoring

### Chrome DevTools
```javascript
// Open DevTools → Performance
// Record 10 seconds
// Check:
// - Frame rate: Should be ~60 FPS
// - GPU usage: Main bottleneck
// - Synthesis: <1ms per frame
```

### Console Metrics
```javascript
// Check inference timings in console
lastInferenceTime: X.XX ms
averageInferenceTime: Y.YY ms
```

**Target:** <16ms total (for 60 FPS)

## Comparing Models

### Train Model
```bash
cd backend
python train_orbit.py --data-dir data/audio --epochs 100
```

### Test Side-by-Side
1. Load legacy model → observe behavior
2. Load orbit model → observe behavior
3. Compare smoothness, determinism, visual quality

## Next Steps After Testing

### If Tests Pass ✅
1. Fine-tune training hyperparameters
2. Experiment with `k_bands` (4, 6, 8)
3. Test with different audio genres
4. Implement section detection
5. Add impact envelope

### If Tests Fail ❌
1. Check console for errors
2. Verify metadata format
3. Re-export model with correct settings
4. Compare against Python synthesis output
5. File issue with error details

## Quick Commands Reference

```bash
# Backend
cd backend
python train_orbit.py --data-dir data/audio --epochs 1  # Quick test
python -m pytest tests/                                  # Run tests

# Frontend
cd frontend
npm run build                                            # Build check
npm run dev                                              # Start dev server

# Copy models
cp backend/checkpoints/*.onnx frontend/public/
cp backend/checkpoints/*.json frontend/public/

# Git status
git status --short
git add -A
git commit -m "Implement orbit synthesizer in frontend"
```

## Documentation

- **Implementation Details**: `ORBIT_SYNTHESIZER_IMPLEMENTATION.md`
- **Status & Results**: `IMPLEMENTATION_STATUS.md`
- **Quick Overview**: `README_ORBIT_IMPLEMENTATION.md`
- **This Guide**: `QUICKSTART_TESTING.md`

## Support

### Common Issues
1. **"Module not found"**: Check `sys.path` in backend
2. **"ONNX export failed"**: Update PyTorch and ONNX
3. **"Type error in TS"**: Run `npm install` again
4. **"Model outputs wrong shape"**: Check `k_bands` matches

### Debug Mode
```javascript
// In modelInference.ts, add:
console.log('[DEBUG] Control signals:', controlSignals);
console.log('[DEBUG] Orbit state:', this.orbitState);
console.log('[DEBUG] Synthesized c:', c);
```

---

**Ready to test!** 🚀 Follow the steps above and report any issues.
