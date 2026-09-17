$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskProject = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskScript = Join-Path $taskRoot 'scripts\unreal\build_maritime_harbor.py'
$taskLog = Join-Path $taskRoot 'logs\maritime_harbor_build.log'
$taskReport = Join-Path $taskRoot 'logs\maritime_harbor_build.json'
$taskPidFile = Join-Path $taskRoot 'logs\maritime_harbor_process.json'
if (Test-Path -LiteralPath $taskPidFile) {
    $taskSaved = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
    $taskRunning = Get-Process -Id $taskSaved.id -ErrorAction SilentlyContinue
    if ($taskRunning -and $taskRunning.Path -eq 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe' -and $taskRunning.StartTime.ToUniversalTime() -eq ([datetime]$taskSaved.started).ToUniversalTime()) {
        throw 'Close the running maritime simulation before rebuilding its loaded map and materials.'
    }
}
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'logs') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $taskRoot '.unreal-cache') -Force | Out-Null
[Environment]::SetEnvironmentVariable('UE-LocalDataCachePath',(Join-Path $taskRoot '.unreal-cache'),'Process')
Write-Host 'Building editor-only material bridge (one compiler process)...'
& 'D:\UE_4.27\Engine\Build\BatchFiles\Build.bat' BlocksEditor Win64 Development "-Project=$taskProject" '-CompilerVersion=14.38.33130' '-Module=MaritimeEditorTools' -MaxParallelActions=1 -WaitMutex -NoHotReloadFromIDE
if ($LASTEXITCODE -ne 0) { throw 'Editor bridge compilation failed.' }
$taskBuildId = (Get-Content 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.modules' -Raw | ConvertFrom-Json).BuildId
@{ BuildId=$taskBuildId; Modules=@{ MaritimeEditorTools='UE4Editor-MaritimeEditorTools.dll' } } | ConvertTo-Json | Set-Content -Encoding Ascii (Join-Path (Split-Path $taskProject) 'Plugins\MaritimeEditorTools\Binaries\Win64\UE4Editor.modules')
Write-Host 'Importing research vessel and generating animated ocean map...'
& 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor-Cmd.exe' $taskProject "-ExecutePythonScript=$taskScript" '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities' -unattended -NullRHI -nosplash "-abslog=$taskLog"
if (!(Test-Path -LiteralPath $taskReport)) { throw "No build report: $taskLog" }
$taskResult = Get-Content -LiteralPath $taskReport -Raw | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $taskResult.status -ne 'PASS') { throw "Map build failed. Inspect $taskReport and $taskLog" }
Write-Host 'Map ready. Start Run-Maritime-Scene.cmd.' -ForegroundColor Green
