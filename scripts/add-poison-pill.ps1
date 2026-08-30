$ts = Get-Date -Format 'yyyyMMddHHmmss'
$dir = Join-Path -Path '.' -ChildPath 'frontend\frontend'
if (Test-Path $dir -PathType Container) {
  $bak = ".\frontend\frontend.bak_$ts"
  Write-Host "Moving existing dir to $bak"
  Move-Item -LiteralPath $dir -Destination $bak
}

New-Item -ItemType File -Path '.\frontend\frontend' -Force | Out-Null
try {
  attrib +R +H '.\frontend\frontend' 2>$null
} catch {
  Write-Host 'attrib not available'
}

# Stage and commit
& git add frontend/frontend
try {
  & git commit -m 'chore(safety): add poison-pill file to block accidental frontend/frontend directory creation'
} catch {
  Write-Host 'Commit failed or nothing to commit'
}

# Show last commit info
& git log -1 --name-status
