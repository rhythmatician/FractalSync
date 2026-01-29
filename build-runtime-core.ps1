#!/usr/bin/env pwsh
# Build runtime_core with explicit PATH setup
# Works around VS Code terminal PATH issues

# Ensure we're in the venv
if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Python venv not activated"
    exit 1
}

# Try to find Rust in standard location
$cargoPath = "$HOME\.cargo\bin"

if (Test-Path $cargoPath) {
    Write-Host "✓ Found Cargo in $cargoPath"
    $env:PATH = "$cargoPath;$env:PATH"
} else {
    Write-Host "⚠ Cargo not found in $cargoPath"
    Write-Host "  Please ensure Rust is installed: https://rustup.rs/"
}

# Verify cargo is available
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($null -eq $cargo) {
    Write-Host "❌ cargo still not found in PATH"
    Write-Host "   Try restarting VS Code or using another terminal"
    exit 1
}

Write-Host "✓ cargo available: $(cargo --version)"

# Navigate to runtime-core
Set-Location runtime-core

# Build with maturin
Write-Host ""
Write-Host "Building runtime_core with maturin..."
Write-Host "=========================================="

maturin develop --release

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Build successful!"
    Write-Host ""
    Write-Host "You can now run:"
    Write-Host "  python diagnostic.py"
    Write-Host "  python test_e2e.py"
    Write-Host "  python ..\backend\api\server.py"
} else {
    Write-Host ""
    Write-Host "❌ Build failed with exit code $LASTEXITCODE"
    exit 1
}
