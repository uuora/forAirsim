$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv\pythonw.exe'
if (!(Test-Path -LiteralPath $taskPython)) { $taskPython = Join-Path $taskRoot '.venv\python.exe' }
$taskPanel = Join-Path $taskRoot 'python\run_route_experiment_panel.py'
foreach ($taskPath in @($taskPython, $taskPanel)) {
    if (!(Test-Path -LiteralPath $taskPath)) { throw "Missing panel runtime: $taskPath" }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $taskRoot 'python'
Start-Process -FilePath $taskPython -ArgumentList @(('"'+$taskPanel+'"')) -WorkingDirectory $taskRoot -WindowStyle Normal
