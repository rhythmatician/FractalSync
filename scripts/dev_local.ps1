<#
PowerShell helper to build the Rust runtime, run its tests, start the backend server,
and run the frontend dev server. Designed for local development convenience.

Usage (default):
  .\scripts\dev_local.ps1

Run frontend dev server in the current terminal (useful to see Vite logs):
  .\scripts\dev_local.ps1 -Foreground

This script assumes you have:
- maturin installed and available on PATH (for building the Python bindings)
- cargo installed (Rust toolchain)
- node/npm available (for frontend tasks)
- a Python venv prepared (recommended: use the project's `.venv`)

The script will stop on errors to avoid cascading failures.
#>

param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition | Split-Path -Parent
Push-Location $root

Write-Host "[dev_local] Starting local development assembly from: $root"

# Build and install runtime-core into the active venv
Push-Location runtime-core
if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
    Write-Error "maturin not found on PATH. Install maturin (pip install maturin) or use your packaging workflow."
    Exit 1
}
Write-Host "[dev_local] Building runtime-core (maturin develop --release)"
maturin develop --release

Write-Host "[dev_local] Running runtime-core tests (cargo test --release)"
cargo test --release
Pop-Location

# Start backend server in background (if python available)
Push-Location backend
if (Test-Path ".venv/Scripts/Activate.ps1") {
    Write-Host "[dev_local] Activating .venv"
    . .venv/Scripts/Activate.ps1
} else {
    Write-Host "[dev_local] No .venv found in backend/ -- make sure your environment is set up"
}

Write-Host "[dev_local] Starting backend server in background (python api/server.py)"
Start-Process -FilePath "python" -ArgumentList "api/server.py" -NoNewWindow
Pop-Location

# Build & run frontend
Push-Location frontend
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "npm not found on PATH"
    Exit 1
}
Write-Host "[dev_local] Building frontend (npm run build)"
npm run build

if ($Foreground) {
    Write-Host "[dev_local] Running frontend dev server in foreground (npm run dev)"
    npm run dev
} else {
    Write-Host "[dev_local] Launching frontend dev server in background (npm run dev)"
    Start-Process -FilePath "npm" -ArgumentList "run dev" -NoNewWindow
}
Pop-Location

Write-Host "[dev_local] Done. Backend should be running, frontend dev server started."
Write-Host "Visit http://localhost:3000 (Vite) or check process list for started services."