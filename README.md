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

To train a new model:

1. Place audio files in `backend/data/audio/` (or specify path)
2. Start training via the UI or command line:
```bash
cd backend
python train.py --data-dir data/audio --epochs 100
```

The trained model will be exported to ONNX format and can be used by the frontend.

## Features

- Real-time audio analysis from microphone input
- ML-learned mappings between audio features and visual parameters
- Smooth morphing Julia sets rendered with WebGL
- Training UI for model management and monitoring
