# run_songformer_all_audio.ps1
# Run from FractalSync repo root

$ErrorActionPreference = "Stop"

# --- Config (edit if needed) ---
$audioDir     = Join-Path $PSScriptRoot "..\backend\data\audio"
$scpPath      = Join-Path $PSScriptRoot "..\batch_all.scp"
$outputsDir   = Join-Path $PSScriptRoot "..\outputs"
$songformerDir = Join-Path $PSScriptRoot "..\..\SongFormer"

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
if (-not (Test-Path $songformerDir -PathType Container)) {
  throw "SongFormer directory not found at: $songformerDir. Update the script with the correct path or clone the SongFormer repo."
}

# Figure out the correct folders to add to PYTHONPATH (make absolute and explicit)
$pyPaths = @()
# If infer is at the repo root
if (Test-Path (Join-Path $songformerDir "infer")) {
  $pyPaths += (Resolve-Path $songformerDir).Path
}

# Common layout: src/SongFormer -> add src/SongFormer (this repo uses src/SongFormer)
if (Test-Path (Join-Path $songformerDir "src\SongFormer")) {
  $pyPaths += (Resolve-Path (Join-Path $songformerDir "src\SongFormer")).Path
}

# Common layout: src -> may contain SongFormer package subfolder
if (Test-Path (Join-Path $songformerDir "src")) {
  $pyPaths += (Resolve-Path (Join-Path $songformerDir "src")).Path
}

# If infer package is specifically under src (older layouts)
if (Test-Path (Join-Path $songformerDir "src\infer")) {
  $pyPaths += (Resolve-Path (Join-Path $songformerDir "src")).Path
}

# Add top-level third_party and src/third_party if they exist
if (Test-Path (Join-Path $songformerDir "third_party")) {
  $pyPaths += (Resolve-Path (Join-Path $songformerDir "third_party")).Path
}
if (Test-Path (Join-Path $songformerDir "src\third_party")) {
  $pyPaths += (Resolve-Path (Join-Path $songformerDir "src\third_party")).Path
}

# Fallback: search for infer directory anywhere under the repo; if found add its parent (the package root)
if (-not $pyPaths -or $pyPaths.Count -eq 0) {
  $found = Get-ChildItem -Path $songformerDir -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'infer' } | Select-Object -First 1
  if ($found) {
    $pyPaths += (Resolve-Path $found.Parent.FullName).Path
  }
}

if (-not $pyPaths -or $pyPaths.Count -eq 0) {
  throw "Could not find an 'infer' package inside $songformerDir. Either install SongFormer into your environment or set PYTHONPATH to the SongFormer repo root (the folder that contains an 'infer' package)."
}

# Deduplicate and set PYTHONPATH
$pyPaths = $pyPaths | Select-Object -Unique
$env:PYTHONPATH = ($pyPaths -join ';')
Write-Host "Using PYTHONPATH: $env:PYTHONPATH"
Write-Host "Using SongFormer dir: $songformerDir"

# Diagnostic: ensure both 'infer' and 'musicfm' are importable before proceeding
$infer_ok = & $pythonExe -c "import importlib.util; print('1' if importlib.util.find_spec('infer') is not None else '0')"
$musicfm_ok = & $pythonExe -c "import importlib.util; print('1' if importlib.util.find_spec('musicfm') is not None else '0')"
Write-Host "infer importable: $infer_ok"; Write-Host "musicfm importable: $musicfm_ok"
if ($infer_ok.Trim() -ne '1') {
  throw "Cannot import 'infer' from Python. Check PYTHONPATH ($env:PYTHONPATH) or install SongFormer into the environment."
}
if ($musicfm_ok.Trim() -ne '1') {
  throw "Module 'musicfm' not found. Fix by installing it into the songformer env: & `"$pythonExe`" -m pip install -e `"$songformerDir\src\third_party\musicfm`"`"` or add the proper path to PYTHONPATH."
}

Push-Location $songformerDir
try {
  $env:SONGFORMER_FORCE_CPU = "1"

  # Quick diagnostic: show whether Python can find the package
  & $pythonExe -c "import importlib.util, sys; print('sys.path[0..4]=', sys.path[:5]); print('find_spec(infer)=', importlib.util.find_spec('infer'))"

  # Prepare and show the exact command we're about to run (for reproducibility)
  $inputPath = (Join-Path $PSScriptRoot "..\batch_all.scp")
  $outputPath = (Join-Path $PSScriptRoot "..\outputs")

  # Locate configs/SongFormer.yaml under common locations and set run directory accordingly
  $configName = "SongFormer.yaml"
  $possibleConfigDirs = @(
    $songformerDir,
    (Join-Path $songformerDir "src\SongFormer"),
    (Join-Path $songformerDir "src")
  )

  $runDir = $null
  foreach ($d in $possibleConfigDirs) {
    if ($d -and (Test-Path (Join-Path $d (Join-Path "configs" $configName)))) {
      $runDir = (Resolve-Path $d).Path
      break
    }
  }

  if (-not $runDir) {
    # Try a recursive search for any configs/SongFormer.yaml
    $foundConfig = Get-ChildItem -Path $songformerDir -Filter $configName -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*\configs\$configName" } | Select-Object -First 1
    if ($foundConfig) {
      $runDir = (Resolve-Path (Split-Path $foundConfig.Directory -Parent)).Path
      # If file is in .../something/configs/SongFormer.yaml then runDir should be parent of configs
      # i.e., if configs is at $foundConfig.Directory, we need its parent
      $runDir = (Resolve-Path (Split-Path $foundConfig.Directory -Parent)).Path
    }
  }

  if (-not $runDir) {
    throw "Cannot find configs\$configName in SongFormer repo. Ensure repository has configs or run src/SongFormer/utils/fetch_pretrained.py to populate configs and ckpts."
  }

  # Locate checkpoint file SongFormer.safetensors in common places and prefer absolute path
  $checkpointName = "SongFormer.safetensors"
  $possibleCheckpoints = @(
    (Join-Path $runDir $checkpointName),
    (Join-Path $songformerDir $checkpointName),
    (Join-Path $songformerDir "ckpts\$checkpointName"),
    (Join-Path $songformerDir "src\SongFormer\ckpts\$checkpointName")
  )
  $checkpointPath = $possibleCheckpoints | Where-Object { Test-Path $_ } | Select-Object -First 1

  if (-not $checkpointPath) {
    # fallback: search for file recursively
    $foundCkpt = Get-ChildItem -Path $songformerDir -Filter $checkpointName -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($foundCkpt) { $checkpointPath = $foundCkpt.FullName }
  }

  if (-not $checkpointPath) {
    throw "Checkpoint '$checkpointName' not found in SongFormer repo. Run src/SongFormer/utils/fetch_pretrained.py or place the checkpoint under the repo (e.g. ckpts/)."
  }

  $cmd = "Push-Location $runDir; $env:PYTHONPATH = '$env:PYTHONPATH'; $env:SONGFORMER_FORCE_CPU = '1'; & '$pythonExe' -m infer.infer -i $inputPath -o $outputPath --model SongFormer --checkpoint `"$checkpointPath`" --config_path $configName --debug --save_probs -gn 1 -tn 1; Pop-Location"
  Write-Host "Running: $cmd"

  Push-Location $runDir
  try {
    & $pythonExe -m infer.infer `
      -i $inputPath `
      -o $outputPath `
      --model SongFormer `
      --checkpoint "$checkpointPath" `
      --config_path $configName `
      --debug `
      --save_probs `
      -gn 1 `
      -tn 1
  }
  finally {
    Pop-Location
  }
}
finally {
  Pop-Location
}
