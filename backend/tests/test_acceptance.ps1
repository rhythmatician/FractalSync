#!/usr/bin/env pwsh
# Quick validation script to verify all acceptance criteria

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "FractalSync Implementation Validation" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"
$script:passed = 0
$script:failed = 0

function Test-Step {
    param(
        [string]$Description,
        [scriptblock]$Test
    )
    
    Write-Host "Testing: $Description..." -NoNewline
    try {
        & $Test
        Write-Host " ✓ PASS" -ForegroundColor Green
        $script:passed++
    } catch {
        Write-Host " ✗ FAIL" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
        $script:failed++
    }
}

Write-Host "SECTION 1: Backend Training" -ForegroundColor Yellow
Write-Host "-" * 70

Test-Step "Training script runs without hanging" {
    Push-Location backend
    try {
        $output = & python train.py --epochs 1 --max-files 1 --no-gpu-rendering --num-workers 0 2>&1 | Out-String
        if ($output -notmatch "Training complete!") {
            throw "Training did not complete successfully"
        }
    } finally {
        Pop-Location
    }
}

Test-Step "ONNX model file exists" {
    $onnxFiles = Get-ChildItem backend/checkpoints/*.onnx -ErrorAction Stop
    if ($onnxFiles.Count -eq 0) {
        throw "No ONNX model files found"
    }
}

Test-Step "ONNX metadata file exists" {
    $metadataFiles = Get-ChildItem backend/checkpoints/*.onnx_metadata.json -ErrorAction Stop
    if ($metadataFiles.Count -eq 0) {
        throw "No ONNX metadata files found"
    }
}

Test-Step "Checkpoint file exists" {
    $checkpoints = Get-ChildItem backend/checkpoints/checkpoint_*.pt -ErrorAction Stop
    if ($checkpoints.Count -eq 0) {
        throw "No checkpoint files found"
    }
}

Test-Step "Training history file exists" {
    if (!(Test-Path backend/checkpoints/training_history.json)) {
        throw "Training history not found"
    }
}

Write-Host ""
Write-Host "SECTION 2: Backend API" -ForegroundColor Yellow
Write-Host "-" * 70

Test-Step "API server starts (testing for 5 seconds)" {
    $job = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python backend/api/server.py 2>&1
    }
    
    Start-Sleep -Seconds 5
    
    if ($job.State -ne "Running") {
        $output = Receive-Job $job
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -ErrorAction SilentlyContinue
        throw "API server failed to start: $output"
    }
    
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
}

Test-Step "API responds to status endpoint" {
    $job = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python backend/api/server.py 2>&1
    }
    
    Start-Sleep -Seconds 3
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/train/status" -UseBasicParsing
        if ($response.StatusCode -ne 200) {
            throw "Got status code $($response.StatusCode)"
        }
        
        $json = $response.Content | ConvertFrom-Json
        if ($json.status -ne "idle") {
            throw "Unexpected status: $($json.status)"
        }
    } finally {
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "SECTION 3: Frontend" -ForegroundColor Yellow
Write-Host "-" * 70

Test-Step "Frontend dependencies installed" {
    if (!(Test-Path frontend/node_modules)) {
        throw "node_modules not found, run: cd frontend && npm install"
    }
}

Test-Step "Frontend builds without errors" {
    Push-Location frontend
    try {
        $env:NODE_ENV = "production"
        $output = npm run build 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed: $output"
        }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "RESULTS" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Passed: $script:passed" -ForegroundColor Green
Write-Host "Failed: $script:failed" -ForegroundColor $(if ($script:failed -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($script:failed -eq 0) {
    Write-Host "✓ All acceptance criteria met!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start backend API:  cd backend && python api/server.py"
    Write-Host "  2. Start frontend:     cd frontend && npm run dev"
    Write-Host "  3. Open browser:       http://localhost:3001"
    Write-Host "  4. Load model from:    backend/checkpoints/model_orbit_control_*.onnx"
    exit 0
} else {
    Write-Host "✗ Some tests failed. See details above." -ForegroundColor Red
    Write-Host ""
    Write-Host "For help, see IMPLEMENTATION_SUMMARY.md" -ForegroundColor Yellow
    exit 1
}
