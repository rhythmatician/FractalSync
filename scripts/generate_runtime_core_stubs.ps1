# Generate runtime_core Python stubs and write them to backend/stubs/runtime_core
# Usage: Run from repo root in PowerShell: .\scripts\generate_runtime_core_stubs.ps1

Push-Location runtime-core
try {
    # Build the wheel and install it in editable mode so the generator can introspect
    maturin develop --release
} finally {
    Pop-Location
}

# Generate the stubs by introspecting the live module
python scripts/generate_runtime_core_stubs.py -o backend/stubs/runtime_core
if ($LASTEXITCODE -ne 0) { throw "stub generation failed" }

Write-Host "Runtime core stubs generated to backend/stubs/runtime_core/"
