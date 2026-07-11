param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutputRoot = "runs\wiki_en_factorized_matched",
    [int]$Steps = 1000,
    [int]$GradAccumSteps = 1,
    [int]$BatchSize = 4,
    [int]$SeqLen = 512,
    [int]$EvalInterval = 100,
    [int]$EvalBatches = 1,
    [int]$LogInterval = 10,
    [double]$LearningRate = 3e-4,
    [int[]]$Seeds = @(1, 2, 3),
    [string]$Device = "auto",
    [switch]$DryRun,
    [switch]$SkipProfile
)

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @(
    "scripts\run_scale_lm_comparison.py",
    "--models", "drm_125m", "gpt2_125m", "opt_125m",
    "--dataset", "wikipedia-en",
    "--wikipedia-output", "data\wikipedia_en_20231101_sample.txt",
    "--output-root", $OutputRoot,
    "--steps", "$Steps",
    "--seeds"
) + $Seeds + @(
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--seq-len", "$SeqLen",
    "--lr", "$LearningRate",
    "--eval-interval", "$EvalInterval",
    "--eval-batches", "$EvalBatches",
    "--no-eval-first",
    "--log-interval", "$LogInterval",
    "--device", $Device,
    "--hf-vocab-size", "256",
    "--save-best-checkpoint"
)

if (-not $SkipProfile) {
    $ArgsList += @(
        "--profile-drm",
        "--profile-batch-size", "1",
        "--profile-seq-len", "32",
        "--profile-repeats", "3"
    )
}

if ($DryRun) {
    $ArgsList += "--dry-run"
}

& $Python @ArgsList
