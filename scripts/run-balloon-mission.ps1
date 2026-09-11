param([ValidateSet('a_hit', 'fallback', 'both')][string]$Scenario = 'fallback')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $projectRoot '.venv\python.exe'
# Reuse the existing Blocks process or start it before creating targets.
& (Join-Path $PSScriptRoot 'start-scene.ps1')
[string[]]$scenarios = if ($Scenario -eq 'both') { @('a_hit', 'fallback') } else { @($Scenario) }
foreach ($case in $scenarios) {
    if ($case -ne $scenarios[0]) {
        & $pythonExe (Join-Path $projectRoot 'python\setup_scene.py')
        if ($LASTEXITCODE -ne 0) { throw 'Scene setup failed.' }
    }
    Write-Host "Running $case. Press M in Blocks for the overview."
    & $pythonExe (Join-Path $projectRoot 'python\run_balloon_mission.py') --scenario $case
    if ($LASTEXITCODE -ne 0) { throw "Experiment $case failed. Check logs/mission_*/report.json." }
}
& $pythonExe (Join-Path $projectRoot 'python\summarize_experiments.py')
Write-Host 'Completed: target contact confirmed, both drones landed and disarmed; scene paused.'
