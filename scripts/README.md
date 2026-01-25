Developer script catalog

scripts/dev_local.ps1 (PowerShell)
- Builds `runtime-core` into your active venv using `maturin develop --release`.
- Runs `cargo test --release` in `runtime-core`.
- Starts the backend API server (`python api/server.py`) in the background.
- Builds the frontend and starts `npm run dev` (foreground or background, see `-Foreground`).

Usage:
- PowerShell: `.\in\dev_local.ps1` (or `powershell -ExecutionPolicy ByPass -File scripts/dev_local.ps1`)
- POSIX: `./scripts/dev_local.sh`

Notes:
- These scripts assume tooling is available on your PATH (maturin, cargo, npm, python).
- Activate or create the backend `.venv` as needed prior to running; the scripts attempt to auto-activate `.venv` if present in `backend/`.
- The scripts are convenience helpers and not a replacement for CI pipelines. They are intended to make local development quick and repeatable.