# AI Coding Agent Instructions (FractalSync)

These instructions make AI agents immediately productive in this repo.

## Big picture<!-- TOC -->

- [AI Coding Agent Instructions FractalSync](#ai-coding-agent-instructions-fractalsync)
    - [Big Picture](#big-picture)
    - [Repository Structure](#repository-structure)
    - [Developer Workflows](#developer-workflows)
    - [Project Conventions](#project-conventions)
    - [Integration Points](#integration-points)
    - [Debugging Tips](#debugging-tips)
    - [Examples](#examples)
    - [Testing](#testing)
    - [Building](#building)

<!-- /TOC -->
- Two-tier app: **backend** (Python + PyTorch) and **frontend** (React + Vite + onnxruntime-web).
- **Runtime core is implemented in Rust (`runtime-core`) and is the single source of truth for geometry, orbit synthesis, feature extraction and visual metrics.** Use `runtime-core` code first — implement logic in Rust and expose it to Python and the browser via `runtime-core/src/pybindings.rs` and `runtime-core/src/wasm_bindings.rs`.
- Current model: **orbit-based control model** (predicts control signals that synthesize Julia seeds). Training exports an ONNX model plus a metadata JSON consumed by the browser.
- Feature extraction prefers the Rust `runtime_core` extractor for performance; a Python fallback (`backend/src/python_feature_extractor.py`) is used automatically when the Rust extractor fails a short subprocess sanity check (see `runtime_core_bridge._rust_extractor_sanity_check`).

## Key files & responsibilities
- `runtime-core/` — **Rust crate that implements the shared runtime**. It implements geometry, controller (orbit synthesis), feature extraction, and runtime visual metrics. Tests and bindings live here:
  - `runtime-core/src/pybindings.rs` — Python/PyO3 bindings used by the backend
  - `runtime-core/src/wasm_bindings.rs` — wasm-bindgen bindings used by the browser (via `wasm-orbit` / `wasm-pack`)
  - Run: `cargo test -q` (rust tests) and `maturin develop --release` to install Python wheel for local dev.
- `backend/train.py` — CLI entrypoint to train the orbit control model (primary way to train).
- `backend/api/server.py` — lightweight FastAPI server exposing model artifacts:
  - `/api/model/latest` — download latest `.onnx`
  - `/api/model/metadata` — latest `*_metadata.json`
- `backend/src/control_trainer.py`, `control_model.py` — training glue and high-level model wiring (prefer delegating math/geometry to `runtime-core`).
- `backend/src/export_model.py` — ONNX exporter (opset 18, prefers dynamo exporter, embeds external data, writes `*.onnx_metadata.json`).
- `backend/src/runtime_core_bridge.py` — thin adapter that calls into `runtime-core` (and runs the extractor sanity check). Keep this small — prefer moving logic into Rust.
- `backend/src/python_feature_extractor.py` — librosa-based extractor and normalization helpers used as fallback and for stats.
- `frontend/src/lib/modelInference.ts` — loads ONNX + metadata, supports `orbit_control` and legacy models; expects normalization stats in metadata when present.
- `checkpoints/` — where checkpoints, ONNX models and metadata are saved.

## Developer workflows (short)
- Backend setup (Windows):
  - `cd backend`
  - `pip install -r requirements.txt`
- Train locally (CLI):
  - `cd backend`
  - `python train.py --data-dir data/audio --epochs 100 [--include-delta] [--include-delta-delta] [--no-gpu-rendering]`
- Start model-serving API (serves artifacts, not training control):
  - `cd backend` then `python api/server.py` (serves `/api/model/*` endpoints)
- Frontend dev:
  - `cd frontend`
  - `npm install`
  - `npm run dev` (visit http://localhost:3000)
- Tests:
  - `pytest backend`
  - `npm test --prefix frontend`
  - `cargo test -q` (runtime-core)

## Project Conventions
- **Rust-first policy:** Implement core geometry, synth, feature extraction and visual metrics in `runtime-core`. Avoid duplicating algorithms in Python/TypeScript — add or adjust bindings instead.
- Audio features are flattened windows: `[centroid, flux, rms, zcr, onset, rolloff] × window_frames` → `input_dim = 6 * window_frames`.
- **Velocity features**: Optional delta (first-order derivative) and delta-delta (second-order derivative) features can be enabled:
  - Base only: 6 features per frame → `input_dim = 6 * window_frames`
  - With delta: 12 features per frame → `input_dim = 12 * window_frames`
  - With delta-delta: 18 features per frame → `input_dim = 18 * window_frames`
- Feature caching: `.npy` files keyed by path + mtime + extractor config; auto-invalidates on config change.
- Tensors in trainer:
  - Handle DataLoader batches that return tuples/lists; extract the single tensor element to avoid extra dims.
  - Keep model outputs (`visual_params`) as tensors (do not `.item()`); stack metric lists and align lengths with batch.
- Velocity-based prediction:
  - Always enabled; adds jerk penalty for smoother parameter transitions
  - Tracks velocity state across batches; handles partial batches correctly
  - Uses `VelocityLoss` from `velocity_predictor.py` to penalize rapid velocity changes
- Error handling:
  - Model `forward()` validates input dim and raises with a clear message if mismatched.

## Integration points
- Backend ↔ Frontend: exported ONNX model + `*.onnx_metadata.json` consumed by `frontend/src/lib/modelInference.ts`.
  - Metadata keys the frontend expects: `input_shape`, `output_dim`, `parameter_names`, `parameter_ranges`, optional `feature_mean` and `feature_std`, `model_type` (`orbit_control`), and `k_bands` (for control models).
  - `export_to_onnx()` uses opset 18, prefers the dynamo exporter, and attempts to inline external data sidecars so the browser gets a single binary.
- Frontend build/runtime: `vite.config.ts` copies a canonical `ort-wasm.wasm` into `public/` if present; dev server proxies `/api` → `http://localhost:8000`.
- Visual metrics and rendering hooks: `backend/src/visual_metrics.py` and `backend/src/julia_gpu.py` (GPU optional).

## Runtime-core (Rust) — guidance for changes and builds
- `runtime-core` is the canonical implementation of geometry, orbits, features and runtime visual metrics. When adding or changing core behavior:
  - Implement the logic in `runtime-core/src/*` and add unit tests in `runtime-core/tests` (run with `cargo test -q`).
  - Update the Python bindings in `runtime-core/src/pybindings.rs` and the wasm bindings in `runtime-core/src/wasm_bindings.rs` as needed.
  - Rebuild and validate the Python wheel: `cd runtime-core && maturin develop --release` (installs locally for `import runtime_core`).
  - Rebuild wasm bindings via `cd wasm-orbit && wasm-pack build --target web` and update frontend consumers.
  - Add or update parity tests that validate exported constants and function signatures (examples: `SAMPLE_RATE`, `WINDOW_FRAMES`, `FeatureExtractor.extract_windowed_features`, `OrbitState.step/synthesize`).
- Quick sanity checks:
  - Python: `python -c "import runtime_core as rc; fe=rc.FeatureExtractor(); print(fe.test_simple())"` — `test_simple` exists to confirm bindings load.
  - The backend bridge runs a short subprocess probe (`runtime_core_bridge._rust_extractor_sanity_check`); replicate that snippet when debugging.

## Debugging tips
- Rust feature extractor: `runtime_core` is preferred but may hang; the bridge runs a short subprocess sanity check and logs a warning on failure — the Python extractor is used automatically on failure.
- ONNX export: check opset (18) and the generated `*.onnx_metadata.json`; missing `feature_mean`/`feature_std` means the frontend will not auto-normalize inputs.
- Ports & proxies: frontend dev server (3000) proxies to backend (8000); failing fetches often mean the backend server is not running or metadata path is wrong.
- Windows specifics: `train.py` forces `mp.set_start_method('spawn', force=True)` to avoid worker handle duplication issues.
- GPU renderer: `julia_gpu` uses ModernGL/GLFW headless mode and falls back to CPU rendering; use `--no-gpu-rendering` for CI/quick runs.

## Examples
- Start server then train:
  - Server: `cd backend && python api/server.py`

```ps1
## Building
cd runtime-core; maturin develop --release; cd ..  # runtime-core
cd wasm-orbit; wasm-pack build --target web; cd .. # wasm bindings for frontend
npm --prefix frontend run build --silent  # frontend
# backend does not need to be built


## Testing
pytest backend #  backend
npm test --prefix frontend  # frontend
cargo test -q # runtime-core

## Training
cd backend; python train.py; cd ..
```

# Copilot's bad habbits and how to avoid them

When writing commands that change directories or run multi-line scripts, Copilot may produce code that assumes a persistent working directory or uses shell-specific features that don't work in all environments. To ensure reliability, follow these guidelines:

**Working-directory note:** Copilot may not preserve the current working directory between commands and can assume the repository root.

> Set-Location: Cannot find path 'C:\Users\JeffHall\git\FractalSync\frontend\frontend' because it does not exist.

To avoid ambiguity, prefer commands with explicit paths (e.g., `npm --prefix frontend run build`) or one of these safer patterns:

- Portable alternative: `npm --prefix frontend run build` — avoids changing directories.
- PowerShell: use `Push-Location` / `Pop-Location` for safe directory changes:
  - Short: `Push-Location frontend; npm run build; Pop-Location`
  - Safer (With cleanup):
    ```powershell
    Push-Location frontend
    try { npm run build } finally { Pop-Location }
    ```

**Shell portability note (terminal / agent guidance):** This guidance is intended for safely running commands in terminals or by automation agents — it is not a repository policy. Avoid POSIX-style heredoc examples like `python << 'PY'` in PowerShell shells (they raise a parser error and can leave you in a Python REPL), and prefer these safer alternatives:

- One-liners: `python -c "print('hi')"` (note quoting differs by shell).
- Multi-line snippets: write the code to a script and run it: `python script.py`.

Quick recovery tips (if you accidentally enter a Python REPL):

- PowerShell: press Ctrl+Z then Enter, or type `exit()` and Enter.
- POSIX shells: press Ctrl+D, or type `exit()` and Enter.

---

If anything here seems off or incomplete (e.g., ports, paths, or training params), tell us and we’ll refine this doc.
