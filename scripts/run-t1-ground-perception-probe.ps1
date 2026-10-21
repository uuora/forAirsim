param([switch]$LiveSemantic)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv/python.exe'
$taskEditor = 'D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'
$taskProject = Join-Path $taskRoot 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
$taskProtocol = Join-Path $taskRoot 'configs/t1_perception_ground_probe_v1p1/protocol.json'
$taskSettings = Join-Path $taskRoot 'configs/t1_perception_ground_probe_v1p1/settings.json'
$env:PYTHONUTF8 = '1'
$env:YOLO_CONFIG_DIR = Join-Path $taskRoot 'logs/t1_model_runtime/ultralytics'
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force | Out-Null
function Test-T1GroundRpc {
    $taskSocket = [Net.Sockets.TcpClient]::new()
    try { return ($taskSocket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $taskSocket.Connected) }
    catch { return $false } finally { $taskSocket.Dispose() }
}
if (Test-T1GroundRpc) { throw 'RPC occupied: preserve current simulation.' }
$taskFolder = Join-Path $taskRoot ('logs/t1_ground_camera_campaign_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'))
New-Item -ItemType Directory -Path $taskFolder | Out-Null
$taskManifestPath = Join-Path $taskFolder 'manifest.json'
$taskManifest = [ordered]@{status='STARTING'; kind='CAMERA_ONLY_DEVELOPMENT'; flight_requests=$false;
    formal_efficacy_results=$false; planned_UE_sessions=1; started=(Get-Date).ToString('o');
    settings=$taskSettings; settings_sha256=(Get-FileHash -LiteralPath $taskSettings).Hash.ToLowerInvariant();
    protocol=$taskProtocol; protocol_sha256=(Get-FileHash -LiteralPath $taskProtocol).Hash.ToLowerInvariant();
    launcher_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant();
    live_semantic=[bool]$LiveSemantic;
    probe_output=(Join-Path $taskFolder 'probe'); engine_closed_after_safe_completion=$false}
function Save-T1GroundManifest {
    $taskManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8
}
Save-T1GroundManifest
$taskSnapshot=Join-Path $taskFolder 'source_snapshot'
New-Item -ItemType Directory -Path $taskSnapshot | Out-Null
foreach($taskName in @('t1_ground_perception_probe.py','setup_scene.py','camera_geometry.py',
    'braking_startup_readiness.py','t1_live_ground_semantic.py','t1_semantic_perception.py','yolo_detector.py','vlx_api_client.py')) {
    Copy-Item -LiteralPath (Join-Path $taskRoot ('python/'+$taskName)) -Destination (Join-Path $taskSnapshot $taskName)
}
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskSnapshot 'run-t1-ground-perception-probe.ps1')
try {
    $taskLog = Join-Path $taskFolder 'unreal.log'
    $taskArguments = @(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap',
        '-game','-windowed','-ResX=1280','-ResY=720',('-settings="'+$taskSettings+'"'),
        ('-abslog="'+$taskLog+'"'),'-nosplash')
    $taskEngine = Start-Process -FilePath $taskEditor -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
    $taskManifest.engine_process_id=$taskEngine.Id
    $taskManifest.engine_started_utc=$taskEngine.StartTime.ToUniversalTime().ToString('o')
    Save-T1GroundManifest
    Write-Host "T1 ground camera probe: own Unreal PID $($taskEngine.Id)"
    $taskDeadline = (Get-Date).AddSeconds(150)
    while (!(Test-T1GroundRpc)) {
        $taskEngine.Refresh()
        if ($taskEngine.HasExited -or (Get-Date) -gt $taskDeadline) { throw 'Unreal startup failed/timed out.' }
        Start-Sleep -Seconds 2
    }
    $taskManifest.status='GROUND_READINESS'
    Save-T1GroundManifest
    & $taskPython (Join-Path $taskRoot 'python/braking_startup_readiness.py') --output (Join-Path $taskFolder 'ground_readiness.json')
    if ($LASTEXITCODE -ne 0) { throw 'Ground readiness failed before camera probe.' }
    $taskManifest.status='CAPTURING'
    Save-T1GroundManifest
    $taskProbeArguments=@((Join-Path $taskRoot 'python/t1_ground_perception_probe.py'),'--output',$taskManifest.probe_output,'--protocol',$taskProtocol)
    if($LiveSemantic){$taskProbeArguments+='--live-semantic'}
    & $taskPython @taskProbeArguments
    if ($LASTEXITCODE -ne 0) { throw 'Camera probe failed; preserve report and owned process.' }
    $taskReportPath = Join-Path $taskManifest.probe_output 'report.json'
    $taskReport = Get-Content -LiteralPath $taskReportPath -Raw | ConvertFrom-Json
    if ($taskReport.status -ne 'PASS' -or !$taskReport.safe_ground_closure -or !$taskReport.paused_after) {
        throw 'Ground closure missing; preserve scene.'
    }
    $taskManifest.report=$taskReportPath
    $taskManifest.report_sha256=(Get-FileHash -LiteralPath $taskReportPath).Hash.ToLowerInvariant()
    $taskManifest.saved_frames=$taskReport.frames.Count
    $taskEngine.Refresh()
    if ($taskEngine.HasExited -or $taskEngine.Path -ne ([IO.Path]::GetFullPath($taskEditor)) -or
        $taskEngine.StartTime.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskManifest.engine_started_utc).UtcTicks) {
        throw 'Owned process identity mismatch: preserve process.'
    }
    Stop-Process -Id $taskEngine.Id
    $taskEngine.WaitForExit()
    $taskManifest.engine_closed_after_safe_completion=$true
    $taskManifest.status='PASS'
} catch {
    $taskManifest.status='FAIL'
    $taskManifest.error=$_.Exception.Message
    throw
} finally {
    $taskManifest.finished=(Get-Date).ToString('o')
    Save-T1GroundManifest
    Write-Host "T1 ground manifest: $taskManifestPath"
}
