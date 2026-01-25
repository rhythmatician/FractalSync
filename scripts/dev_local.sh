#!/usr/bin/env bash
# Cross-platform-ish helper for POSIX shells to build runtime-core, run tests,
# start backend server, and start frontend build/dev server.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[dev_local] Building runtime-core and installing into venv (maturin develop --release)"
if ! command -v maturin >/dev/null 2>&1; then
  echo "ERROR: maturin not found on PATH. Install it via 'pip install maturin' or your packaging workflow." >&2
  exit 1
fi

cd runtime-core
maturin develop --release

echo "[dev_local] Running runtime-core tests (cargo test --release)"
cargo test --release
cd "$ROOT_DIR"

# Start backend server in background
cd backend
if [ -f .venv/bin/activate ]; then
  echo "[dev_local] Activating .venv"
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "[dev_local] No .venv found in backend/ -- ensure your environment is prepared"
fi

echo "[dev_local] Starting backend server in background (python api/server.py)"
python api/server.py &
BACKEND_PID=$!
cd "$ROOT_DIR"

# Build and run frontend dev server
cd frontend
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found on PATH" >&2
  exit 1
fi

echo "[dev_local] Building frontend (npm run build)"
npm run build

echo "[dev_local] Starting frontend dev server (npm run dev)"
npm run dev &
FRONTEND_PID=$!

echo "[dev_local] Done. Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"
echo "Visit http://localhost:3000 for the frontend."