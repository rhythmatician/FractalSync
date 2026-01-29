# FractalSync Frontend

React + TypeScript + Vite frontend for real-time audio-reactive Julia set visualizations.

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:3000

## Key Dependencies

- **React 18** - UI framework
- **onnxruntime-web 1.14.0** - Browser-based ML inference (pinned to avoid dynamic import issues with 1.16+)
- **Vite 5** - Fast dev server and bundler

## Architecture

- `src/components/` - React components (AudioCapture, Visualizer)
- `src/lib/modelInference.ts` - ONNX model loading and inference with **audio-driven post-processing**
- `src/lib/audioFeatures.ts` - Browser-based audio feature extraction
- `src/lib/juliaRenderer.ts` - WebGL Julia set renderer
- `public/` - Static assets including ONNX WASM files (auto-copied from node_modules)

## Audio-Reactive Dynamics

Even with an undertrained model, the visualizer creates dynamic fractals by:
- **Extracting audio features** from the input: spectral centroid, flux, RMS (loudness), zero-crossing rate, onset strength, and spectral rolloff
- **Mixing model outputs with audio features**:
  - Julia parameters influenced by spectral centroid and flux
  - Color hue cycles with loudness (RMS)
  - Saturation boosts on transients (onset strength)
  - Brightness tracks loudness
  - Zoom varies with spectral rolloff and RMS
  - Animation speed increases with flux and onsets

This ensures the fractal responds to music in real-time, even before the model is fully trained.

## Notes

- **ONNX Runtime version**: Pinned to 1.14.0 because 1.16+ uses dynamic ES module imports that conflict with Vite's public folder handling
- **WASM files**: Vite copies the canonical single-thread non‑SIMD `ort-wasm.wasm` from `node_modules` to `public/` on startup. The build will warn if the preferred artifact is missing — there is no runtime automatic fallback.
- **CORS headers**: Required for SharedArrayBuffer (multi-threaded WASM)
