param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [switch]$AutoTune,
    [switch]$DryRun
)

Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @("-m", "data.update_prices")
if ($DryRun) {
    $ArgsList += "--dry-run"
}
if ($AutoTune) {
    $ArgsList += "--auto-tune"
}

& $Python @ArgsList
