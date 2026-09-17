param([switch]$ShowWindow)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv\python.exe'
$taskEditor = 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$taskProject = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskMapFile = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Content\MaritimeHarborV2\Maps\MaritimeHarbor.umap'
$taskSettings = Join-Path $taskRoot 'configs\maritime_harbor_settings.json'
$taskPidFile = Join-Path $taskRoot 'logs\maritime_harbor_process.json'
$taskLog = Join-Path $taskRoot 'logs\maritime_harbor_runtime.log'
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'logs') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $taskRoot '.unreal-cache') -Force | Out-Null
[Environment]::SetEnvironmentVariable('UE-LocalDataCachePath',(Join-Path $taskRoot '.unreal-cache'),'Process')
function Test-MaritimeRpc {
    $taskSocket = [Net.Sockets.TcpClient]::new()
    try { return ($taskSocket.ConnectAsync('127.0.0.1',41452).Wait(400) -and $taskSocket.Connected) }
    catch { return $false } finally { $taskSocket.Dispose() }
}
function Show-MaritimeWindow($Process) {
    if (-not ('MaritimeWindowV2' -as [type])) {
        Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class MaritimeWindowV2 {
  public delegate bool EnumWindowsProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int n);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr h);
  public static void Restore(int processId) {
    EnumWindows((h,l) => { uint p; GetWindowThreadProcessId(h,out p);
      if(p==processId && GetWindowTextLength(h)>0) { ShowWindowAsync(h,9); SetForegroundWindow(h); }
      return true; },IntPtr.Zero);
  }
}
'@
    }
    [MaritimeWindowV2]::Restore($Process.Id)
}
try {
    Write-Host '[1/3] Checking saved maritime map...' -ForegroundColor Cyan
    foreach ($taskPath in @($taskPython,$taskEditor,$taskProject,$taskMapFile,$taskSettings)) {
        if (!(Test-Path -LiteralPath $taskPath)) { throw "Missing: $taskPath. See scripts/build-maritime-harbor.ps1." }
    }
    # AirSim creates configured cameras without replacing NaN defaults.
    $taskConfig = Get-Content -LiteralPath $taskSettings -Raw | ConvertFrom-Json
    $taskCameras = @($taskConfig.ExternalCameras.PSObject.Properties)
    foreach ($taskVehicle in $taskConfig.Vehicles.PSObject.Properties.Value) {
        $taskCameras += @($taskVehicle.Cameras.PSObject.Properties)
    }
    foreach ($taskCamera in $taskCameras) {
        foreach ($taskField in @('X','Y','Z','Pitch','Yaw','Roll')) {
            $taskValue = $taskCamera.Value.$taskField
            if ($null -eq $taskValue -or [double]::IsNaN([double]$taskValue) -or [double]::IsInfinity([double]$taskValue)) {
                throw "Camera '$($taskCamera.Name)' requires a finite $taskField in $taskSettings. Unreal was not started."
            }
        }
    }
    $taskProcess = $null
    if (Test-Path -LiteralPath $taskPidFile) {
        $taskSaved = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
        $taskCandidate = Get-Process -Id $taskSaved.id -ErrorAction SilentlyContinue
        if ($taskCandidate -and $taskCandidate.Path -eq $taskEditor -and $taskCandidate.StartTime.ToUniversalTime() -eq ([datetime]$taskSaved.started).ToUniversalTime()) { $taskProcess = $taskCandidate }
    }
    if (!(Test-MaritimeRpc)) {
        if (!$taskProcess) {
            Write-Host '[2/3] Opening Unreal: ship + animated ocean + balloons...' -ForegroundColor Cyan
            $taskStyle = if ($ShowWindow) {'Normal'} else {'Hidden'}
            $taskArgs = @(('"'+$taskProject+'"'),'/Game/MaritimeHarborV2/Maps/MaritimeHarbor','-game','-windowed','-ResX=1280','-ResY=720',('-settings="'+$taskSettings+'"'),('-abslog="'+$taskLog+'"'),'-nosplash')
            $taskProcess = Start-Process -FilePath $taskEditor -ArgumentList $taskArgs -WindowStyle $taskStyle -PassThru
            [pscustomobject]@{id=$taskProcess.Id;started=$taskProcess.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $taskPidFile -Encoding UTF8
        }
        $taskDeadline=(Get-Date).AddSeconds(150)
        $taskTicks=0
        while (!(Test-MaritimeRpc)) {
            $taskProcess.Refresh()
            if ($taskProcess.HasExited) { throw "Unreal exited ($($taskProcess.ExitCode)). Runtime log: $taskLog" }
            if ((Get-Date) -gt $taskDeadline) { throw "Unreal startup timed out. Inspect: $taskLog" }
            if ($taskTicks % 5 -eq 0) { Write-Host '  Loading map / compiling shaders; please wait...' }
            Start-Sleep -Seconds 2
            $taskTicks++
        }
    } else { Write-Host '[2/3] Maritime simulation already running.' -ForegroundColor Cyan }
    if ($ShowWindow -and $taskProcess) { Show-MaritimeWindow $taskProcess }
    Write-Host '[3/3] Checking scene and saving a preview...' -ForegroundColor Cyan
    & $taskPython (Join-Path $taskRoot 'python\check_maritime_harbor.py')
    if ($LASTEXITCODE -ne 0) { throw "Scene verification failed. See $taskLog" }
    if ($ShowWindow -and $taskProcess) { Show-MaritimeWindow $taskProcess }
    Write-Host 'READY: ship, moving ocean surface and three balloons. Close the Unreal window to exit.' -ForegroundColor Green
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
