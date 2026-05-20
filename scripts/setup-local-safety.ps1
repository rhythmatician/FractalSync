<#
Run this script to install local safety checks for the repository:
- sets git's core.hooksPath to `.githooks` (so the included hook will run on commits)
- optionally appends a small `mkdir` wrapper to the user's PowerShell profile that prevents creating a subdirectory with the same name as the parent (opt-in)
#>

Write-Host "Installing local safety checks..."

# Configure git to use .githooks in this repo
git config core.hooksPath ".githooks"
if ($LASTEXITCODE -ne 0) {
  Write-Error "Failed to set git core.hooksPath. Are you in the repo root?"
  exit 1
}
Write-Host "Set git core.hooksPath to .githooks"

# Ensure the hook file is executable on Unix-like systems (no-op on Windows)
try {
  if (Test-Path .githooks/prevent-frontend-subdir) {
    & chmod +x .githooks/prevent-frontend-subdir 2>$null
  }
} catch {}

# Offer to append PowerShell safety wrapper to user's profile
$profilePath = $PROFILE
Write-Host "PowerShell profile: $profilePath"
$install = Read-Host "Would you like to add a protective 'mkdir' wrapper to your PowerShell profile (prevents creating frontend/frontend)? (y/N)"
if ($install -match '^[Yy]') {
  $snippet = @'
# BEGIN: Prevent accidental creating of a subdir with the same name as parent
function mkdir {
  param([Parameter(Mandatory=$true, ValueFromPipeline=$true, Position=0)] [string] $Path)
  # Resolve the full target path (handles relative/absolute)
  try {
    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Path
  } catch {
    # If it doesn't exist yet, compute the intended path relative to cwd
    $resolved = [System.IO.Path]::Combine((Get-Location).Path, $Path)
  }
  $leaf = [System.IO.Path]::GetFileName($resolved)
  $parent = [System.IO.Path]::GetFileName((Get-Location).Path)
  if ($leaf -and $parent -and $leaf -ieq $parent) {
    Write-Error "ERROR: Cannot create subdirectory with the same name as parent directory: '$parent'"
    return
  }
  Microsoft.PowerShell.Management\New-Item -ItemType Directory -Path $resolved | Out-Null
}
Set-Alias md mkdir -Scope Global
Set-Alias ni mkdir -Scope Global
# END: Prevent accidental creating of a subdir with the same name as parent
'@
  Add-Content -Path $profilePath -Value "`n$snippet`n"
  Write-Host "Appended protective mkdir function to $profilePath" -ForegroundColor Green
} else {
  Write-Host "Skipped modifying PowerShell profile. You can run this script again to enable it." -ForegroundColor Yellow
}

Write-Host "Done. Git hook is active for this repo. It will block commits that include or leave 'frontend/frontend'." -ForegroundColor Green
