param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Torchrun = "torchrun",
    [string]$Config = "configs\drm_500m.yaml",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$OutputRoot = "runs\drm_500m_5b",
    [int]$Gpus = 1,
    [int64]$TargetTokens = 5000000000,
    [int]$BatchSize = 4,
    [int]$GradAccumSteps = 1,
    [int]$SeqLen = 512,
    [double]$LearningRate = 3e-4,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int64]$EvalTokensInterval = 50000000,
    [int64]$CheckpointTokensInterval = 250000000,
    [string]$Resume = "latest",
    [switch]$DryRun,
    [switch]$DryRunForward
)

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$TrainArgs = @(
    "scripts\train_drm_memmap.py",
    "--config", $Config,
    "--dataset-manifest", $DatasetManifest,
    "--output-root", $OutputRoot,
    "--target-tokens", "$TargetTokens",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--seq-len", "$SeqLen",
    "--lr", "$LearningRate",
    "--precision", $Precision,
    "--eval-tokens-interval", "$EvalTokensInterval",
    "--checkpoint-tokens-interval", "$CheckpointTokensInterval",
    "--device", $Device,
    "--resume", $Resume
)

if ($DryRun) {
    $TrainArgs += "--dry-run"
}

if ($DryRunForward) {
    $TrainArgs += "--dry-run-forward"
}

if ($Gpus -gt 1) {
    & $Torchrun --nproc_per_node=$Gpus @TrainArgs
} else {
    & $Python @TrainArgs
}
