# run_songformer_all_audio.ps1
# Run from FractalSync repo root

$ErrorActionPreference = "Stop"

# --- Config (edit if needed) ---
$audioDir     = Join-Path $PSScriptRoot "..\backend\data\audio"
$scpPath      = Join-Path $PSScriptRoot "..\batch_all.scp"
$outputsDir   = Join-Path $PSScriptRoot "..\outputs"
$songformerDir = Join-Path $PSScriptRoot "..\src\SongFormer"

$pythonExe = "C:\Users\JeffHall\miniconda3\envs\songformer\python.exe"

# Extensions to include
$exts = @(".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".aif")

# --- Resolve paths relative to current script location ---
$audioDir      = (Resolve-Path $audioDir).Path
$scpPath       = (Resolve-Path (Split-Path $scpPath -Parent)).Path + "\" + (Split-Path $scpPath -Leaf)
$outputsDir    = (Resolve-Path (Split-Path $outputsDir -Parent)).Path + "\" + (Split-Path $outputsDir -Leaf)
$songformerDir  = (Resolve-Path $songformerDir).Path

# --- Collect audio files ---
$files = Get-ChildItem -Path $audioDir -File -Recurse |
  Where-Object { $exts -contains $_.Extension.ToLowerInvariant() } |
  Sort-Object FullName

if (-not $files -or $files.Count -eq 0) {
  throw "No audio files found under: $audioDir"
}

# --- Write batch_all.scp (one absolute path per line) ---
$lines = $files | ForEach-Object { $_.FullName }
Set-Content -Path $scpPath -Value $lines -Encoding UTF8

Write-Host "Wrote SCP file: $scpPath"
Write-Host "Audio files: $($files.Count)"

# --- Ensure outputs dir exists ---
New-Item -ItemType Directory -Path $outputsDir -Force | Out-Null

# --- Run SongFormer from src\SongFormer like your original command ---
Push-Location $songformerDir
try {
  $env:PYTHONPATH = "..;../third_party"
  $env:SONGFORMER_FORCE_CPU = "1"

  & $pythonExe -m infer.infer `
    -i (Join-Path $PSScriptRoot "..\batch_all.scp") `
    -o (Join-Path $PSScriptRoot "..\outputs") `
    --model SongFormer `
    --checkpoint SongFormer.safetensors `
    --config_path SongFormer.yaml `
    -gn 1 `
    -tn 1
}
finally {
  Pop-Location
}
