# FractalSync - Real-Time Julia Set Music Visualizer with ML

A real-time music visualizer that renders morphing Julia sets, using machine learning to learn correlations between audio features and visual parameters.

## Architecture

- **Backend (Python/PyTorch)**: Training pipeline that learns audio-to-visual mappings, exports models to ONNX
- **Frontend (React)**: Real-time visualization with microphone input, ONNX.js inference, and WebGL Julia set rendering
- **API Server**: FastAPI for training monitoring and model management

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Start the backend API server:
```bash
cd backend
python -m api.server
```

2. Start the frontend dev server:
```bash
cd frontend
npm run dev
```

3. Open your browser to `http://localhost:3000`

4. Allow microphone access and start visualizing!

## Training

### Standard Model Training

To train a new model:

1. Place audio files in `backend/data/audio/` (or specify path)
2. Start training via the UI or command line:
```bash
cd backend
python train.py --data-dir data/audio --epochs 100
```

### Physics-Based Model Training (NEW)

To train with physics-based velocity prediction and curriculum learning:

```bash
cd backend
python train_physics.py --data-dir data/audio --epochs 100 --use-curriculum
```

**Physics Model Features:**
- Predicts velocity instead of position for Julia parameter `c`
- Speed is modulated by audio loudness (RMS energy)
- Uses preset Mandelbrot orbits for curriculum learning
- More physically interpretable and temporally consistent

See [backend/docs/PHYSICS_MODEL.md](backend/docs/PHYSICS_MODEL.md) for detailed documentation.

The trained model will be exported to ONNX format and can be used by the frontend.

## Features

- Real-time audio analysis from microphone input
- ML-learned mappings between audio features and visual parameters
- Smooth morphing Julia sets rendered with WebGL
- Training UI for model management and monitoring
- **NEW**: Physics-based model with velocity prediction (treats Julia parameter as physical object)
- **NEW**: Curriculum learning using Mandelbrot set orbital trajectories


## Training parameters:

**epochs** (e.g., 1, 5, 100)
- Number of times the model sees the entire dataset during training.
- Each epoch = one complete pass through all audio files.
- More epochs → model learns more patterns, but risks overfitting if too many.
- You're using 1 epoch for testing; typically 50-100+ for real training.

**batch_size** (e.g., 32)
- Number of audio feature samples processed together before updating model weights.
- Your audio files are sliced into many feature frames; 32 frames are fed to the model at once.
- Larger batches → faster training, more stable gradients, but higher memory usage.
- Smaller batches → noisier gradients, slower, but can generalize better.
- 32 is a common sweet spot; yours has ~162,464 samples (5,077 full batches + 1 partial of 30).

**learning_rate** (e.g., 0.0001)
- How aggressively the model adjusts weights after each batch.
- Too high → training unstable, loss oscillates or explodes.
- Too low → training very slow, may get stuck in local minima.
- 0.0001 is conservative but safe; 0.001 would be faster but riskier.

**window_frames** (e.g., 10)
- Number of consecutive audio frames (time steps) flattened into one input vector.
- Your extractor computes 6 features per frame (centroid, flux, rms, zcr, onset, rolloff).
- 10 frames × 6 features = 60-dimensional input vector.
- Larger windows → model sees more temporal context but input grows (5 frames → 30-dim, 20 frames → 120-dim).

## Troubleshooting

### Why is onnxruntime-web pinned to 1.14.0?

Version 1.16+ introduced dynamic ES module imports for different execution providers (JSEP, WebGPU), which conflicts with Vite's handling of files in the `public/` folder. When ONNX Runtime tries to dynamically import `.mjs` files, Vite intercepts them as source code rather than serving them as static assets, causing 404 errors.

**Solution**: Pin to 1.14.0, which uses a simpler WASM-only backend without dynamic imports.

**Alternative**: If you need 1.16+ features, configure Vite's `optimizeDeps` and `publicDir` to properly handle the `.mjs` files, or use a bundled version of onnxruntime-web.
