param([switch]$Check)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = @('.venv/python.exe','.venv/Scripts/python.exe') | ForEach-Object { Join-Path $taskRoot $_ } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$taskPython) { throw 'Run scripts/setup-geometric-control.ps1 with Python 3.10 first.' }
$taskPanel = Join-Path $taskRoot 'python/run_geometric_control_panel.py'
$env:PYTHONUTF8='1'
if ($Check) {
    & $taskPython $taskPanel --check
    if ($LASTEXITCODE -ne 0) { throw 'Offline Qt entry check failed.' }
} else {
    $taskWindowPython = Join-Path (Split-Path $taskPython) 'pythonw.exe'
    if (!(Test-Path -LiteralPath $taskWindowPython)) { $taskWindowPython=$taskPython }
    Start-Process -FilePath $taskWindowPython -ArgumentList @(('"'+$taskPanel+'"')) -WorkingDirectory $taskRoot -WindowStyle Hidden
}
