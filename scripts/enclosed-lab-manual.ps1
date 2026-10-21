param([ValidateSet('Open','Check','BuildParameters','OpenParameters')][string]$Action = 'Open')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskProject = Join-Path $taskRoot 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
$taskEditorFolder = 'D:/UE_4.27/Engine/Binaries/Win64'
$taskScript = Join-Path $taskRoot 'scripts/unreal/enclosed_lab_manual.py'
$taskLatestPath = Join-Path $taskRoot 'logs/enclosed_lab_manual_latest.json'
$taskManualConfig = Join-Path $taskRoot 'configs/enclosed_uav_lab_manual.json'
$taskManualSettings = Join-Path $taskRoot 'configs/enclosed_uav_lab_manual_settings.json'
$taskMap = '/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual'
$taskExpectedProject = $taskProject.Replace('\','/').ToLowerInvariant()
$taskRunning = Get-CimInstance Win32_Process -Filter "Name='UE4Editor.exe' OR Name='UE4Editor-Cmd.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Replace('\','/').ToLowerInvariant().Contains($taskExpectedProject) }
if ($taskRunning) { throw 'Save All and close this Blocks editor/game session first. No running session was stopped.' }
[Environment]::SetEnvironmentVariable('UE-LocalDataCachePath',(Join-Path $taskRoot '.unreal-cache'),'Process')

function Invoke-TaskManualCommand([string]$Mode) {
    [Environment]::SetEnvironmentVariable('ENCLOSED_LAB_MANUAL_MODE',$Mode,'Process')
    $taskLog = Join-Path $taskRoot ('logs/enclosed_lab_manual_'+$Mode+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log')
    & (Join-Path $taskEditorFolder 'UE4Editor-Cmd.exe') $taskProject "-ExecutePythonScript=$taskScript" '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities' -unattended -NullRHI -nosplash "-abslog=$taskLog"
    $taskCode = $LASTEXITCODE
    $taskReceipt = Get-Content -LiteralPath $taskLatestPath -Raw | ConvertFrom-Json
    if ($taskCode -ne 0 -or $taskReceipt.status -notlike 'PASS_*') { throw "Operation failed. Inspect $($taskReceipt.report)" }
    return $taskReceipt
}

if ($Action -eq 'Check') {
    $taskResult = Invoke-TaskManualCommand 'audit'
    Write-Host "Geometry check: $($taskResult.status)"
    Write-Host "New export folder: $($taskResult.output)"
    Invoke-Item -LiteralPath $taskResult.output
    exit 0
}
$taskPrepared = Invoke-TaskManualCommand 'prepare'
if ($Action -eq 'OpenParameters') {
    Start-Process -FilePath 'notepad.exe' -ArgumentList ('"'+$taskManualConfig+'"')
    exit 0
}
if ($Action -eq 'BuildParameters') {
    $taskValidated = Get-Content -LiteralPath $taskManualConfig -Raw | ConvertFrom-Json
    if ($taskValidated.unreal_map -ne $taskMap) { throw 'Parameter construction is restricted to the manual working map.' }
    [Environment]::SetEnvironmentVariable('ENCLOSED_LAB_CONFIG',$taskManualConfig,'Process')
    [Environment]::SetEnvironmentVariable('ENCLOSED_LAB_BASE','/Game/EnclosedUAVLabManual','Process')
    $taskBuilder = Join-Path $taskRoot 'scripts/unreal/build_enclosed_uav_lab.py'
    $taskLog = Join-Path $taskRoot ('logs/enclosed_lab_manual_parameters_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log')
    & (Join-Path $taskEditorFolder 'UE4Editor-Cmd.exe') $taskProject "-ExecutePythonScript=$taskBuilder" '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities' -unattended -NullRHI -nosplash "-abslog=$taskLog"
    $taskBuildCode = $LASTEXITCODE
    $taskBuildReceipt = Get-Content -LiteralPath (Join-Path $taskRoot 'logs/enclosed_lab_manual_build_latest.json') -Raw | ConvertFrom-Json
    if ($taskBuildCode -ne 0 -or $taskBuildReceipt.status -ne 'PASS_EDITOR_GEOMETRY_COLLISION') { throw "Build failed: $($taskBuildReceipt.report)" }
    $taskResult = Invoke-TaskManualCommand 'audit'
    Write-Host "Working map rebuilt and checked: $($taskResult.output)"
    exit 0
}
[Environment]::SetEnvironmentVariable('ENCLOSED_LAB_MANUAL_MODE','open','Process')
$taskGuiLog = Join-Path $taskRoot ('logs/enclosed_lab_manual_editor_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log')
$taskArguments = @(('"'+$taskProject+'"'), $taskMap,
    '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities', '-nosplash', ('"-settings='+$taskManualSettings+'"'), ('"-abslog='+$taskGuiLog+'"'))
# ExecutePythonScript auto-closes UE4.27 after the script, so it is intentionally absent here.
# A visible editor is the interface the user requested. No Play/game/arm command is issued.
$taskProcess = Start-Process -FilePath (Join-Path $taskEditorFolder 'UE4Editor.exe') -ArgumentList $taskArguments -WorkingDirectory $taskRoot -PassThru
Write-Host "Manual editor PID: $($taskProcess.Id)"
Write-Host "Working map: $taskMap"
