param(
    [ValidateSet('geometry_oracle','sensor_surface_estimate')][string]$TargetSource = 'geometry_oracle',
    [ValidateRange(1,3)][int]$Sessions = 1,
    [string]$ExperimentConfig,
    [switch]$ProbeOnly,
    [switch]$ReverseFirstOrder
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -le 5) {
    $taskBuiltinModules = Join-Path $PSHOME 'Modules'
    $env:PSModulePath = $taskBuiltinModules + [IO.Path]::PathSeparator + $env:PSModulePath
}
$env:PYTHONUTF8 = '1'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv\python.exe'
$taskEditor = 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$taskProject = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskSettings = Join-Path $taskRoot 'configs\settings_trajectory_interface.json'
if (!$ExperimentConfig) { $ExperimentConfig = Join-Path $taskRoot 'configs\single_trajectory_interface.json' }
$ExperimentConfig = (Resolve-Path -LiteralPath $ExperimentConfig).Path
foreach ($taskPath in @($taskPython,$taskEditor,$taskProject,$taskSettings,$ExperimentConfig)) {
    if (!(Test-Path -LiteralPath $taskPath)) { throw "Missing: $taskPath" }
}
$taskCondition = Get-Content -LiteralPath $ExperimentConfig -Raw | ConvertFrom-Json
& (Join-Path $PSScriptRoot 'assert-trajectory-source-freeze.ps1') -SourceHashes $taskCondition.flight_source_sha256
$taskBalloon = Get-Content -LiteralPath (Join-Path $taskRoot 'configs\balloon.json') -Raw | ConvertFrom-Json
$taskOffset = @(0,1,2 | ForEach-Object {
    [double]$taskCondition.target_world_ned_m[$_] - [double]$taskBalloon.position_ned_m[$_]
})
function Test-TrajectoryRpc {
    $taskSocket = [Net.Sockets.TcpClient]::new()
    try { return ($taskSocket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $taskSocket.Connected) }
    catch { return $false } finally { $taskSocket.Dispose() }
}
if (Test-TrajectoryRpc) { throw 'RPC is occupied; preserve the existing simulation.' }
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$taskFolder = Join-Path $taskRoot "logs\trajectory_interface_pilot_$taskStamp"
New-Item -ItemType Directory -Path $taskFolder | Out-Null
$taskManifestPath = Join-Path $taskFolder 'manifest.json'
$taskManifest = [ordered]@{
    schema_version=1; status='RUNNING'; kind='TRAJECTORY_CONTROLLER_INTERFACE_PILOT';
    target_source=$TargetSource; probe_only=[bool]$ProbeOnly; formal_thesis_effect_results=$false;
    planned_sessions=$Sessions; planned_flights_per_session=$(if ($ProbeOnly) {0} else {2});
    settings_path=$taskSettings;
    settings_sha256=(Get-FileHash -LiteralPath $taskSettings -Algorithm SHA256).Hash.ToLowerInvariant();
    experiment_config=$ExperimentConfig;
    experiment_config_sha256=(Get-FileHash -LiteralPath $ExperimentConfig -Algorithm SHA256).Hash.ToLowerInvariant();
    reverse_first_order=[bool]$ReverseFirstOrder;
    started=(Get-Date).ToString('o'); sessions=@()
}
function Save-TrajectoryManifest {
    $taskManifest | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8
}
Save-TrajectoryManifest
Push-Location $taskRoot
try {
    for ($taskIndex=1; $taskIndex -le $Sessions; $taskIndex++) {
        if (Test-TrajectoryRpc) { throw 'RPC remained occupied before an independent session.' }
        $taskRuntimeLog = Join-Path $taskFolder "session_${taskIndex}_unreal.log"
        $taskArguments = @(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap',
            '-game','-windowed','-ResX=1280','-ResY=720',('-settings="'+$taskSettings+'"'),
            ('-abslog="'+$taskRuntimeLog+'"'),'-nosplash')
        $taskEngine = Start-Process -FilePath $taskEditor -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
        $taskSession = [ordered]@{
            session_index=$taskIndex; status='STARTING'; engine_process_id=$taskEngine.Id;
            engine_started=$taskEngine.StartTime.ToUniversalTime().ToString('o'); runtime_log=$taskRuntimeLog;
            probe_path=(Join-Path $taskFolder "session_${taskIndex}_ground_probe");
            batch_path=(Join-Path $taskFolder "session_${taskIndex}_batch")
        }
        $taskManifest.sessions += $taskSession
        Save-TrajectoryManifest
        Write-Host "Trajectory interface session ${taskIndex}: starting own Unreal process $($taskEngine.Id)"
        $taskDeadline = (Get-Date).AddSeconds(150)
        while (!(Test-TrajectoryRpc)) {
            $taskEngine.Refresh()
            if ($taskEngine.HasExited) { throw "Unreal exited before RPC readiness: $taskRuntimeLog" }
            if ((Get-Date) -gt $taskDeadline) { throw "Unreal startup timed out: $taskRuntimeLog" }
            Start-Sleep -Seconds 2
        }
        $taskOldScenes = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'scene_*' | Select-Object -ExpandProperty FullName)
        & $taskPython (Join-Path $taskRoot 'python\setup_scene.py') --target-offset $taskOffset[0] $taskOffset[1] $taskOffset[2]
        if ($LASTEXITCODE -ne 0) { throw 'Scene setup failed; no flight started.' }
        $taskNewScenes = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'scene_*' |
            Where-Object { $_.FullName -notin $taskOldScenes })
        if ($taskNewScenes.Count -ne 1) { throw 'Scene report was not unique.' }
        $taskScenePath = Join-Path $taskNewScenes[0].FullName 'scene_report.json'
        $taskScene = Get-Content -LiteralPath $taskScenePath -Raw | ConvertFrom-Json
        if ($taskScene.status -ne 'PASS') { throw 'Scene preflight did not pass.' }
        foreach ($taskAxis in 0,1,2) {
            if ([math]::Abs($taskScene.balloon.observed_position_ned_m[$taskAxis] - $taskCondition.target_world_ned_m[$taskAxis]) -gt 0.02) {
                throw 'Scene target disagrees with frozen configuration.'
            }
        }
        $taskSession.scene_report=$taskScenePath
        $taskSession.scene_report_sha256=(Get-FileHash -LiteralPath $taskScenePath -Algorithm SHA256).Hash.ToLowerInvariant()
        $taskSession.status='GROUND_PROBE'
        Save-TrajectoryManifest
        & $taskPython (Join-Path $taskRoot 'python\probe_trajectory_camera.py') --output-dir $taskSession.probe_path
        if ($LASTEXITCODE -ne 0) { throw 'Ground interface probe failed; no flight started; preserve scene.' }
        $taskProbePath = Join-Path $taskSession.probe_path 'report.json'
        $taskProbe = Get-Content -LiteralPath $taskProbePath -Raw | ConvertFrom-Json
        if ($taskProbe.status -ne 'PASS' -or !$taskProbe.paused -or !$taskProbe.vehicles_released_and_grounded_at_end) {
            throw 'Ground probe was incomplete; no flight started.'
        }
        $taskSession.probe_report_sha256=(Get-FileHash -LiteralPath $taskProbePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if (!$ProbeOnly) {
            $taskSession.status='RUNNING_FLIGHTS'
            $taskSessionPath = Join-Path $taskFolder "session_$taskIndex.json"
            $taskSession | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $taskSessionPath -Encoding UTF8
            Save-TrajectoryManifest
            $taskBatchArguments = @((Join-Path $taskRoot 'python\run_trajectory_batch.py'), '--config', $ExperimentConfig,
                '--target-source', $TargetSource, '--output-dir', $taskSession.batch_path, '--session-record', $taskSessionPath)
            if (($taskIndex % 2 -eq 0) -xor [bool]$ReverseFirstOrder) { $taskBatchArguments += '--reverse-order' }
            & $taskPython @taskBatchArguments
            if ($LASTEXITCODE -ne 0) { throw 'Interface flight failed; paused scene preserved for inspection.' }
            $taskBatchPath = Join-Path $taskSession.batch_path 'batch_report.json'
            $taskBatch = Get-Content -LiteralPath $taskBatchPath -Raw | ConvertFrom-Json
            if ($taskBatch.status -ne 'PASS' -or $taskBatch.runs.Count -ne 2) { throw 'Incomplete paired flight batch.' }
            $taskSession.batch_report_sha256=(Get-FileHash -LiteralPath $taskBatchPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $taskSession.task_success_count=$taskBatch.task_success_count
        }
        # Close only this owned process after all ground/release checks have passed.
        $taskEngine.Refresh()
        if ($taskEngine.HasExited -or $taskEngine.Path -ne $taskEditor -or
            $taskEngine.StartTime.ToUniversalTime().ToString('o') -ne $taskSession.engine_started) {
            throw 'Owned process identity changed; preserve current state.'
        }
        $taskSession.status='PASS'
        Save-TrajectoryManifest
        Stop-Process -Id $taskEngine.Id
        $taskEngine.WaitForExit()
        $taskSession.engine_closed_after_safe_completion=$true
        Save-TrajectoryManifest
    }
    $taskManifest.status='PASS'
} catch {
    $taskManifest.status='FAIL'
    $taskManifest.error=$_.Exception.Message
    if ($taskManifest.sessions.Count -gt 0 -and $taskManifest.sessions[-1].status -ne 'PASS') {
        $taskManifest.sessions[-1].status='STOPPED_FOR_INSPECTION'
    }
    throw
} finally {
    $taskManifest.finished=(Get-Date).ToString('o')
    Save-TrajectoryManifest
    Pop-Location
    Write-Host "Trajectory manifest: $taskManifestPath"
}
