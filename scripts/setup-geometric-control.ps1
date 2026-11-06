param([string]$Python='',[string]$EngineRoot='',[switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskPython=@('.venv/python.exe','.venv/Scripts/python.exe') | ForEach-Object { Join-Path $taskRoot $_ } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$taskPython) {
    if ($CheckOnly) { throw 'No .venv Python. Install Python 3.10 and rerun without -CheckOnly.' }
    if ($Python) { & $Python -m venv (Join-Path $taskRoot '.venv') } else { & py -3.10 -m venv (Join-Path $taskRoot '.venv') }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.10 environment.' }
    $taskPython=Join-Path $taskRoot '.venv/Scripts/python.exe'
}
$env:PYTHONUTF8='1'
if (!$CheckOnly) {
    $taskClient=Join-Path $taskRoot 'external/AirSim/PythonClient'
    if (!(Test-Path -LiteralPath $taskClient)) { throw 'Initialize AirSim: git submodule update --init --recursive' }
    & $taskPython -m pip install -r (Join-Path $taskRoot 'requirements-geometric.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $taskPython -m pip install --no-deps -e $taskClient
    if ($LASTEXITCODE -ne 0) { throw 'AirSim client installation failed.' }
}
if ($EngineRoot) {
    if (!(Test-Path -LiteralPath (Join-Path $EngineRoot 'Engine/Binaries/Win64/UE4Editor.exe'))) { throw 'EngineRoot must contain UE 4.27 Engine/Binaries/Win64/UE4Editor.exe' }
    $taskRuntime=Join-Path $taskRoot '.runtime'
    New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
    $taskLocal=Join-Path $taskRuntime 'local_machine.json'
    $taskOptions=@{}
    if (Test-Path -LiteralPath $taskLocal) {
        $taskExisting=Get-Content -LiteralPath $taskLocal -Raw | ConvertFrom-Json
        foreach ($property in $taskExisting.PSObject.Properties) { $taskOptions[$property.Name]=$property.Value }
    }
    $taskOptions['engine_root']=$EngineRoot
    $taskOptions | ConvertTo-Json | Set-Content -LiteralPath $taskLocal -Encoding UTF8
}
& $taskPython -c "import numpy,PyQt5,psutil,airsim,matplotlib; print('Geometric dependencies OK; no simulator or flight started')"
if ($LASTEXITCODE -ne 0) { throw 'Dependency import check failed.' }
& (Join-Path $PSScriptRoot 'open-geometric-control-panel.ps1') -Check
