param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Checkpoint = "",
    [string]$RunDir = "runs\gpt2_125m_4090_base",
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

$ChatArgs = @(
    "scripts\chat_gpt2_125m_4090_base.py",
    "--run-dir", $RunDir,
    "--device", $Device,
    "--dtype", $Dtype,
    "--max-new-tokens", "$MaxNewTokens",
    "--temperature", "$Temperature",
    "--top-k", "$TopK",
    "--max-turns", "$MaxTurns"
)

if ($Checkpoint) {
    $ChatArgs += @("--checkpoint", $Checkpoint)
}

& $Python @ChatArgs
