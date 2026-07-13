param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Output = "data\wikipedia_en_20231101_500m.txt",
    [int64]$MaxChars = 500000000,
    [int]$MinDocChars = 200,
    [switch]$Overwrite
)

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @(
    "scripts\prepare_wikipedia_en.py",
    "--output", $Output,
    "--max-chars", "$MaxChars",
    "--min-doc-chars", "$MinDocChars",
    "--streaming"
)

if ($Overwrite) {
    $ArgsList += "--overwrite"
}

& $Python @ArgsList
