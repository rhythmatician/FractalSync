param(
    [string]$Executable = 'target/release/examples/operational_field_report.exe',
    [string]$Asset = 'tmp/geometry145-frozen-operational.bin',
    [string]$OutputDirectory = 'tmp/operational-field-native',
    [string]$Manifest = '',
    [string]$ReferencePolicy = ''
)
$ErrorActionPreference = 'Stop'
$exePath = (Resolve-Path -LiteralPath $Executable).Path
$assetPath = (Resolve-Path -LiteralPath $Asset).Path
$nativeArguments=@('--asset', ('"' + $assetPath + '"'))
if ($Manifest) { $manifestPath=(Resolve-Path -LiteralPath $Manifest).Path; $nativeArguments+=@('--manifest',('"'+$manifestPath+'"')) }
if ($ReferencePolicy) { $policyPath=(Resolve-Path -LiteralPath $ReferencePolicy).Path; $nativeArguments+=@('--policy',('"'+$policyPath+'"')) }
$outPath = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($outPath) | Out-Null
$records = @()
for ($runIndex = 0; $runIndex -lt 3; $runIndex++) {
    $stdoutPath = Join-Path $outPath "run-$runIndex.json"
    $stderrPath = Join-Path $outPath "run-$runIndex.stderr.txt"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $exePath -ArgumentList $nativeArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $workingSetPeak = [long]0
    while (-not $process.HasExited) {
        $process.Refresh()
        try { $workingSetPeak = [Math]::Max($workingSetPeak, $process.PeakWorkingSet64) } catch [InvalidOperationException] {}
        Start-Sleep -Milliseconds 20
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "Native experiment failed; see $stderrPath" }
    $records += [pscustomobject]@{
        run = $runIndex
        exit_code = $process.ExitCode
        process_seconds = $timer.Elapsed.TotalSeconds
        sampled_os_peak_working_set_bytes = $workingSetPeak
        stdout_file = "run-$runIndex.json"
    }
}
$metadata = [pscustomobject]@{
    device = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
    os = [Environment]::OSVersion.VersionString
    asset_sha256 = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    executable_sha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLowerInvariant()
    manifest_sha256 = if ($Manifest) {(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()} else {$null}
    reference_policy_sha256 = if ($ReferencePolicy) {(Get-FileHash -LiteralPath $policyPath -Algorithm SHA256).Hash.ToLowerInvariant()} else {$null}
    memory_note = 'OS peak working set sampled every 20 ms; includes asset/input copies, reference diagnostics, cache-stress scratch and serialization, not only coefficients'
    runs = $records
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outPath 'metadata.json') -Encoding utf8
$records | Format-Table
