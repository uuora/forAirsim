$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $projectRoot '.venv\python.exe'
$editorExe = 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$uproject = Join-Path $projectRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
foreach ($required in @($pythonExe, $editorExe, $uproject)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing: $required" }
}
function Test-AirSimPort {
    $connection = [System.Net.Sockets.TcpClient]::new()
    try {
        $pending = $connection.ConnectAsync('127.0.0.1', 41451)
        return ($pending.Wait(500) -and $connection.Connected)
    } catch { return $false } finally { $connection.Dispose() }
}
if (-not (Test-AirSimPort)) {
    $runningBlocks = Get-CimInstance Win32_Process -Filter "Name='UE4Editor.exe'" |
        Where-Object { $_.CommandLine -like "*$uproject*" }
    if (-not $runningBlocks) {
        # This is the interactive simulator window for the user.
        Start-Process -FilePath $editorExe -ArgumentList @("`"$uproject`"", '-game', '-windowed', '-ResX=1280', '-ResY=720') -WindowStyle Normal
    }
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Test-AirSimPort)) {
        if ((Get-Date) -gt $deadline) { throw 'AirSim RPC not ready. Check Blocks window and its Saved/Logs directory; an editor session needs Play.' }
        Start-Sleep -Seconds 2
    }
}
& $pythonExe (Join-Path $projectRoot 'python\setup_scene.py')
if ($LASTEXITCODE -ne 0) { throw 'Scene setup failed; see logs/scene_*/scene.log' }
