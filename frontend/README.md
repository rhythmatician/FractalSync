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
- **onnxruntime-web 1.18.0** - Browser-based ML inference (see `vite.config.ts` for WASM handling; if upgrading to versions that use dynamic imports, update Vite's `optimizeDeps`/`publicDir`)
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

### Renderer controls
- **Grad Mode** selects whether gradient-derived normals are computed: Off (disabled), Cheap (gated low-cost FD), Full (accurate FD). Default: **Full**.
- **Prefer Gradient Normals** is a preference toggle that controls which normal is used when both gradient-derived and analytic normals are available. It does not enable gradient computation; enable Grad Mode to allow gradient normals to be computed. Default: **enabled**.

- **Shader source**: The canonical fragment shader is located at `shared/shaders/julia.frag` and is served by the backend at `/api/shader/julia.frag`. The frontend fetches the shader at runtime to avoid keeping multiple copies.


- **ONNX Runtime**: Repo uses onnxruntime-web 1.18.0 (see `package.json`/`vite.config.ts`). If you encounter dynamic-import issues with newer runtime versions (1.16+), update Vite's `optimizeDeps`/`publicDir` to handle `.mjs` files or provide a bundled runtime.
- **WASM files**: Vite copies the canonical single-thread non‑SIMD `ort-wasm.wasm` from `node_modules` to `public/` on startup. The build will warn if the preferred artifact is missing — there is no runtime automatic fallback.
- **CORS headers**: Required for SharedArrayBuffer (multi-threaded WASM)
