#!/usr/bin/env bash
#
# Build and setup runtime_core for both Python backend and WebAssembly frontend.
#
# Prerequisites:
#   - Rust 1.70+ (rustup)
#   - Python 3.8+ with venv activated
#   - Node.js 16+
#   - maturin (installed via: cargo install maturin)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_ACTIVE="${VIRTUAL_ENV:-}"

if [ -z "$VENV_ACTIVE" ]; then
    echo "⚠ Python venv not activated. Please run: source .venv/bin/activate"
    exit 1
fi

echo "=== Building runtime_core ==="

# Build Python bindings (via maturin in development mode)
cd "$PROJECT_ROOT/runtime-core"
echo "Building Python bindings..."
maturin develop --release

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
npm install

echo ""
echo "✓ Build complete!"
echo ""
echo "Next steps:"
echo "  1. Start API server: cd backend && python api/server.py"
echo "  2. Start frontend: cd frontend && npm run dev"
echo "  3. Open http://localhost:3000"
