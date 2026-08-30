# run_songformer_all_audio_minimal.ps1
# Compact runner: process all audio files one-by-one with SongFormer (resilient to per-file failures)

$ErrorActionPreference = "Stop"

# --- Config (edit if needed) ---
$audioDir     = Join-Path $PSScriptRoot "..\backend\data\audio"
$outputsDir   = Join-Path $PSScriptRoot "..\outputs_songformer"
$songformerDir = Join-Path $PSScriptRoot "..\..\SongFormer"
$pythonExe = "C:\Users\JeffHall\miniconda3\envs\songformer\python.exe"  # override if needed

# Extensions to include
$exts = @('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.aiff', '.aif')

# Resolve and validate
$audioDir = (Resolve-Path $audioDir).Path
$songformerDir = (Resolve-Path $songformerDir).Path
New-Item -ItemType Directory -Path $outputsDir -Force | Out-Null

# Find config and checkpoint (same logic as main script)
$configName = 'SongFormer.yaml'
$checkpointName = 'SongFormer.safetensors'

# helper to search for a file under repo and return parent dir for configs
function Find-ConfigRunDir {
  param($repoRoot, $configName)
  # normalize repoRoot to a single path
  $repoRoot = $repoRoot | Select-Object -First 1
  $repoRoot = (Resolve-Path $repoRoot -ErrorAction SilentlyContinue).Path
  $srcPath = Join-Path $repoRoot 'src'
  $candidates = @(
    (Join-Path $srcPath 'SongFormer'),
    $srcPath,
    $repoRoot
  )
  foreach ($d in $candidates) {
    if ($d -and (Test-Path (Join-Path $d (Join-Path 'configs' $configName)))) { return (Resolve-Path $d).Path }
  }
  $found = Get-ChildItem -Path $repoRoot -Filter $configName -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*\configs\$configName" } | Select-Object -First 1
  if ($found) { return (Resolve-Path (Split-Path $found.Directory -Parent)).Path }
  return $null
}

$runDir = Find-ConfigRunDir -repoRoot $songformerDir -configName $configName
if (-not $runDir) { throw "Cannot find configs\$configName in SongFormer repo. Run fetch_pretrained or place configs in the repo." }

# checkpoint search
$possibleCkpts = @(
  (Join-Path $runDir $checkpointName),
  (Join-Path $songformerDir $checkpointName),
  (Join-Path $songformerDir "ckpts\$checkpointName"),
  (Join-Path $songformerDir "src\SongFormer\ckpts\$checkpointName")
)
$checkpointPath = $possibleCkpts | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $checkpointPath) {
  $foundCkpt = Get-ChildItem -Path $songformerDir -Filter $checkpointName -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($foundCkpt) { $checkpointPath = $foundCkpt.FullName }
}
if (-not $checkpointPath) { throw "Checkpoint '$checkpointName' not found in SongFormer repo. Run fetch_pretrained or place the checkpoint under the repo (eg ckpts/)." }

# Build PYTHONPATH
$pyPaths = @()
$pyPaths += (Join-Path $runDir '')
if (Test-Path (Join-Path $songformerDir 'src\third_party')) { $pyPaths += (Resolve-Path (Join-Path $songformerDir 'src\third_party')).Path }
elseif (Test-Path (Join-Path $songformerDir 'third_party')) { $pyPaths += (Resolve-Path (Join-Path $songformerDir 'third_party')).Path }
$env:PYTHONPATH = ($pyPaths | Select-Object -Unique) -join ';'
Write-Host "Using runDir: $runDir"; Write-Host "Using PYTHONPATH: $env:PYTHONPATH"; Write-Host "Using checkpoint: $checkpointPath"

# Collect audio files
$files = Get-ChildItem -Path $audioDir -File -Recurse | Where-Object { $exts -contains $_.Extension.ToLowerInvariant() } | Sort-Object FullName
if (-not $files -or $files.Count -eq 0) { throw "No audio files found under: $audioDir" }

# Prepare summary trackers
$success = @()
$failed = @()
$logDir = Join-Path $outputsDir "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# Process each file with a per-file SCP so infer runs independently and failures don't stop the rest
foreach ($f in $files) {
  $base = [IO.Path]::GetFileNameWithoutExtension($f.Name)
  $tmpScp = Join-Path $env:TEMP ("songformer_temp_$($base)_$([guid]::NewGuid().ToString()).scp")
  # write absolute path (one per line) to temp scp
  $f.FullName | Out-File -FilePath $tmpScp -Encoding utf8

  $fileOutDir = Join-Path $outputsDir $base
  New-Item -ItemType Directory -Path $fileOutDir -Force | Out-Null
  $logFile = Join-Path $logDir ("$base.log")

  Write-Host "Processing: $($f.FullName) -> $fileOutDir"
  Push-Location $runDir
  try {
    $env:SONGFORMER_FORCE_CPU = "1"
    # run and capture output
    & $pythonExe -m infer.infer -i $tmpScp -o $fileOutDir --model SongFormer --checkpoint "$checkpointPath" --config_path $configName --debug --save_probs -gn 1 -tn 1 *> $logFile 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "infer exited with code $code (see $logFile)" }
    Write-Host "OK: $base"
    $success += $f.FullName
  }
  catch {
    Write-Host "FAILED: $base -- see $logFile" -ForegroundColor Red
    $failed += @{ file = $f.FullName; log = $logFile; error = $_.Exception.Message }
  }
  finally {
    Pop-Location
    # clean up temp scp
    Remove-Item -Path $tmpScp -ErrorAction SilentlyContinue
  }
}

# Summary
Write-Host "\nDone. Success: $($success.Count). Failed: $($failed.Count). Logs in: $logDir"
if ($failed.Count -gt 0) {
  $failedFile = Join-Path $outputsDir 'failed_files.txt'
  $failed | ForEach-Object { "$_" } | Out-File -FilePath $failedFile -Encoding UTF8
  Write-Host "Failed list written to: $failedFile"
}

exit 0
