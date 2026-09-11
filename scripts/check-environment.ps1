$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$engineRoot = 'D:\UE_4.27'
$sourceRoot = Join-Path $projectRoot 'external\AirSim'
$report = [ordered]@{
    Timestamp = (Get-Date).ToString('o')
    UnrealEditor = Test-Path "$engineRoot\Engine\Binaries\Win64\UE4Editor.exe"
    UnrealBuildScript = Test-Path "$engineRoot\Engine\Build\BatchFiles\Build.bat"
    AirLibRelease = Test-Path "$sourceRoot\AirLib\lib\x64\Release\AirLib.lib"
    BlocksPlugin = Test-Path "$sourceRoot\Unreal\Environments\Blocks\Plugins\AirSim\AirSim.uplugin"
    BlocksPluginBinary = Test-Path "$sourceRoot\Unreal\Environments\Blocks\Plugins\AirSim\Binaries\Win64\UE4Editor-AirSim.dll"
    Python = Test-Path "$projectRoot\.venv\python.exe"
    WindowsSDK19041 = Test-Path 'C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0'
    FreeDiskGB = [math]::Round((Get-PSDrive D).Free / 1GB, 1)
    AirSimCommit = (& git -C $sourceRoot rev-parse HEAD)
}
$report | ConvertTo-Json | Tee-Object -FilePath "$projectRoot\docs\environment-status.json"
if ($report.Python) {
    & "$projectRoot\.venv\python.exe" -c "import airsim, cv2, numpy; airsim.MultirotorClient(); print('Python imports and RPC client creation OK')"
    if ($LASTEXITCODE -ne 0) { throw 'Python client verification failed' }
    & "$projectRoot\.venv\python.exe" -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency verification failed' }
}
