$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskProject = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskLog = Join-Path $taskRoot 'logs\dual_target_dynamics_compile.log'
$taskActive = Get-CimInstance Win32_Process -Filter "Name = 'UE4Editor.exe' OR Name = 'UE4Editor-Cmd.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Replace('/', '\').IndexOf(
        $taskProject, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
if ($taskActive) { throw 'Close the Blocks simulation before rebuilding its AirSim physics module.' }
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'logs') -Force | Out-Null
& 'D:\UE_4.27\Engine\Build\BatchFiles\Build.bat' BlocksEditor Win64 Development "-Project=$taskProject" `
    '-CompilerVersion=14.38.33130' '-Module=AirSim' -MaxParallelActions=2 -WaitMutex -NoHotReloadFromIDE 2>&1 |
    Tee-Object -FilePath $taskLog
if ($LASTEXITCODE -ne 0) { throw 'AirSim dynamics compilation failed. Inspect logs/dual_target_dynamics_compile.log.' }
$taskEngine = Get-Content 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.modules' -Raw | ConvertFrom-Json
$taskManifest = Join-Path (Split-Path $taskProject) 'Plugins\AirSim\Binaries\Win64\UE4Editor.modules'
$taskModule = Get-Content $taskManifest -Raw | ConvertFrom-Json
if ($taskModule.BuildId -ne $taskEngine.BuildId) { throw 'AirSim module BuildId does not match the installed UE build.' }
Write-Host 'AirSim physics profile compiled. Restart the simulator to load the new module.'
