param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m model.auto_tune
