# Generate runtime_core Python stubs and write them to backend/stubs/runtime_core
# Usage: Run from repo root in PowerShell: .\scripts\generate_runtime_core_stubs.ps1

Push-Location runtime-core
try {
    # Build the wheel and install it in editable mode so pyo3-stubgen can introspect
    maturin develop --release
} finally {
    Pop-Location
}

# Generate the stubs
python -m pyo3_stubgen.generate runtime_core -o backend/stubs/runtime_core --package-name runtime_core ; if ($LASTEXITCODE -ne 0) { Write-Host "pyo3-stubgen failed, continuing" }

# Additionally attempt to produce stubs from metadata exported by the bindings
python scripts/generate_runtime_core_stubs_from_metadata.py -o backend/stubs/runtime_core ; if ($LASTEXITCODE -ne 0) { Write-Host "metadata-based stub generation failed" }

Write-Host "Runtime core stubs generated to backend/stubs/runtime_core/"