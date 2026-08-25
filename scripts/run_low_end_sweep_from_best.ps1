param(
    [string]$BaseCheckpoint = "checkpoints/sweep_005_005/checkpoint_best.pt",
    [int]$Epochs = 25,
    [double]$LearningRate = 2e-4,
    [string]$DataDir = "backend/data/audio",
    [double]$TemporalSmoothnessWeight = 0.03,
    [double]$RolloutBatchFraction = 0.10,
    [int]$RolloutHorizon = 32,
    [double]$RolloutTeacherForcing = 0.3,
    [double]$RolloutLossWeight = 0.10,
    [double]$CurriculumWeight = 1.0,
    [double]$CurriculumDecay = 0.70,
    [int]$BatchSize = 32,
    [switch]$SkipGate
)

$ErrorActionPreference = "Stop"
$python = ".\\.venv\\Scripts\\python"

if (-not (Test-Path $BaseCheckpoint)) {
    throw "Base checkpoint not found: $BaseCheckpoint"
}

$jobs = @(
    @{ name = "sweep_low_002_002"; seq = "0.02"; hit = "0.02" },
    @{ name = "sweep_low_003_003"; seq = "0.03"; hit = "0.03" },
    @{ name = "sweep_low_005_005"; seq = "0.05"; hit = "0.05" }
)

$summary = @()

foreach ($j in $jobs) {
    $saveDir = "checkpoints/$($j.name)"

    & $python backend/train.py `
        --data-dir $DataDir `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --learning-rate $LearningRate `
        --use-curriculum `
        --curriculum-weight $CurriculumWeight `
        --curriculum-decay $CurriculumDecay `
        --no-gpu-rendering `
        --temporal-smoothness-weight $TemporalSmoothnessWeight `
        --sequence-loss-weight $($j.seq) `
        --hit-alignment-weight $($j.hit) `
        --rollout-batch-fraction $RolloutBatchFraction `
        --rollout-horizon $RolloutHorizon `
        --rollout-teacher-forcing $RolloutTeacherForcing `
        --rollout-loss-weight $RolloutLossWeight `
        --resume-checkpoint $BaseCheckpoint `
        --resume-reset-optimizer `
        --save-dir $saveDir

    if ($SkipGate) {
        & $python backend/scripts/evaluate_training_history.py `
            --history "$saveDir/training_history.json" `
            --out-dir "$saveDir/analysis"
        $gateCode = 0
    }
    else {
        & $python backend/scripts/evaluate_training_history.py `
            --history "$saveDir/training_history.json" `
            --out-dir "$saveDir/analysis" `
            --gate `
            --min-epochs 20 `
            --min-alignment-score 55 `
            --max-final-loss 0.03 `
            --max-hit-alignment-loss 0.35 `
            --max-rollout-loss 0.08
        $gateCode = $LASTEXITCODE
    }

    $jsonPath = "$saveDir/analysis/summary.json"
    if (Test-Path $jsonPath) {
        $obj = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $summary += [PSCustomObject]@{
            run = $j.name
            sequence_loss_weight = [double]$j.seq
            hit_alignment_weight = [double]$j.hit
            gate_exit_code = [int]$gateCode
            gate_passed = [bool]$obj.gate.passed
            alignment_score = [double]$obj.alignment.score
            final_loss = [double]$obj.last.loss
            hit_alignment_loss = [double]$obj.last.hit_alignment_loss
            rollout_loss = [double]$obj.last.rollout_loss
            best_checkpoint_exists = (Test-Path "$saveDir/checkpoint_best.pt")
        }
    }
}

$summarySorted = $summary | Sort-Object alignment_score -Descending

$jsonOut = "checkpoints/sweep_low_end_summary.json"
$csvOut = "checkpoints/sweep_low_end_summary.csv"
$mdOut = "checkpoints/sweep_low_end_summary.md"

$summarySorted | ConvertTo-Json -Depth 5 | Set-Content $jsonOut
$summarySorted | Export-Csv -NoTypeInformation $csvOut

$md = @()
$md += "| run | sequence_loss_weight | hit_alignment_weight | gate_passed | alignment_score | final_loss | hit_alignment_loss | rollout_loss | best_checkpoint_exists |"
$md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
foreach ($row in $summarySorted) {
    $md += "| $($row.run) | $([string]::Format('{0:F2}', $row.sequence_loss_weight)) | $([string]::Format('{0:F2}', $row.hit_alignment_weight)) | $($row.gate_passed) | $([string]::Format('{0:F4}', $row.alignment_score)) | $([string]::Format('{0:F4}', $row.final_loss)) | $([string]::Format('{0:F4}', $row.hit_alignment_loss)) | $([string]::Format('{0:F4}', $row.rollout_loss)) | $($row.best_checkpoint_exists) |"
}
$md -join "`n" | Set-Content $mdOut

Write-Host "Low-end sweep complete"
Write-Host "  JSON: $jsonOut"
Write-Host "  CSV:  $csvOut"
Write-Host "  MD:   $mdOut"
$summarySorted | Format-Table -AutoSize
