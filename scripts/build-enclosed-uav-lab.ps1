$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskProject = Join-Path $taskRoot 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
$taskBuilder = Join-Path $taskRoot 'scripts/unreal/build_enclosed_uav_lab.py'
$taskEditor = 'D:/UE_4.27/Engine/Binaries/Win64/UE4Editor-Cmd.exe'
# Never write assets while this same project is running a game/editor session.
$taskRunning = Get-CimInstance Win32_Process -Filter "Name='UE4Editor.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($taskProject) }
if ($taskRunning) { throw 'Close this project after confirming both drones are grounded before rebuilding its assets.' }
$taskLog = Join-Path $taskRoot ('logs/enclosed_lab_build_operator_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log')
[Environment]::SetEnvironmentVariable('UE-LocalDataCachePath',(Join-Path $taskRoot '.unreal-cache'),'Process')
& $taskEditor $taskProject "-ExecutePythonScript=$taskBuilder" '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities' -unattended -NullRHI -nosplash "-abslog=$taskLog"
$taskExit = $LASTEXITCODE
$taskLatest = Get-Content -LiteralPath (Join-Path $taskRoot 'logs/enclosed_lab_build_latest.json') -Raw | ConvertFrom-Json
if ($taskExit -ne 0 -or $taskLatest.status -ne 'PASS_EDITOR_GEOMETRY_COLLISION') { throw "Build failed; inspect $($taskLatest.report)" }
Write-Host "READY: $($taskLatest.report)"
