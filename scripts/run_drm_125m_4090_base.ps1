param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Config = "configs\drm_125m_4090.yaml",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$OutputRoot = "runs\drm_125m_4090_base",
    [int64]$TargetTokens = 150000000,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int]$SeqLen = 512,
    [double]$LearningRate = 3e-4,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int64]$EvalTokensInterval = 10000000,
    [int64]$CheckpointTokensInterval = 50000000,
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

& $Python @TrainArgs
