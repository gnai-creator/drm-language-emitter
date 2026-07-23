param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Checkpoint = "runs\drm_125m_4090_base\checkpoint_last.pt",
    [string]$Device = "cuda",
    [string]$Dtype = "auto",
    [int]$MaxNewTokens = 160,
    [double]$Temperature = 0.8,
    [int]$TopK = 40,
    [int]$MaxTurns = 3
)

if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python scripts\chat_drm_125m_4090_base.py `
    --checkpoint $Checkpoint `
    --device $Device `
    --dtype $Dtype `
    --max-new-tokens $MaxNewTokens `
    --temperature $Temperature `
    --top-k $TopK `
    --max-turns $MaxTurns
