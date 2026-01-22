#!/usr/bin/env pwsh
#
# Build and setup runtime_core for both Python backend and WebAssembly frontend.
# Windows PowerShell version
#
# Prerequisites:
#   - Rust 1.70+ (rustup)
#   - Python 3.8+ with venv activated
#   - Node.js 16+
#   - maturin (installed via: cargo install maturin)
#   - wasm-pack (installed via: cargo install wasm-pack)

$ErrorActionPreference = "Stop"

$projectRoot = (Get-Location).Path
$venvActive = $env:VIRTUAL_ENV

if (-not $venvActive) {
    Write-Host "⚠ Python venv not activated. Please run: .\.venv\Scripts\Activate.ps1"
    exit 1
}

Write-Host "=== Building runtime_core ===" -ForegroundColor Cyan

# Build Python bindings (via maturin in development mode)
Push-Location "$projectRoot\runtime-core"
Write-Host "Building Python bindings..." -ForegroundColor Yellow
& maturin develop --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

# Build WebAssembly bindings
Write-Host "Building WebAssembly bindings..." -ForegroundColor Yellow
Push-Location "$projectRoot\wasm-orbit"
& wasm-pack build --target bundler --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
Push-Location "$projectRoot\frontend"
& npm install
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

Write-Host ""
Write-Host "✓ Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start API server: cd backend && python api/server.py"
Write-Host "  2. Start frontend: cd frontend && npm run dev"
Write-Host "  3. Open http://localhost:3000"
