$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Write-Host 'Keep Blocks open. Press M in its window for the fixed overview.'
& (Join-Path $projectRoot '.venv\python.exe') (Join-Path $projectRoot 'python\run_approach_demo.py')
if ($LASTEXITCODE -ne 0) { throw 'Demo failed. Check logs/approach_*/mission.log; simulation is paused if reachable.' }
Write-Host 'Completed. Simulation is paused for inspection; rerun this script to replay.'
