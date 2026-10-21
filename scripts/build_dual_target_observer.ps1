$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskProject = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskModuleLog = Join-Path $taskRoot 'logs\dual_target_observer_compile.log'
$taskEditor = 'D:\UE_4.27\Engine\Build\BatchFiles\Build.bat'
$taskUsingProject = Get-CimInstance Win32_Process -Filter "Name = 'UE4Editor.exe' OR Name = 'UE4Editor-Cmd.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($taskProject) }
if ($taskUsingProject) { throw 'The Blocks project is running. Close its simulation before compiling the observer DLL.' }
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'logs') -Force | Out-Null
& $taskEditor BlocksEditor Win64 Development "-Project=$taskProject" '-CompilerVersion=14.38.33130' '-Module=Blocks' -MaxParallelActions=2 -WaitMutex -NoHotReloadFromIDE 2>&1 |
    Tee-Object -FilePath $taskModuleLog
if ($LASTEXITCODE -ne 0) { throw 'Observer compilation failed. See logs/dual_target_observer_compile.log.' }
$taskEngineId = (Get-Content 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.modules' -Raw | ConvertFrom-Json).BuildId
$taskModuleManifest = Join-Path (Split-Path $taskProject) 'Binaries\Win64\UE4Editor.modules'
$taskManifest = Get-Content $taskModuleManifest -Raw | ConvertFrom-Json
if ($taskManifest.BuildId -ne $taskEngineId) {
    throw 'Blocks module BuildId differs from this installed UE build; do not launch an incompatible DLL.'
}
Write-Host 'Native mouse observer compiled. The launcher will install it into the isolated control map.'
