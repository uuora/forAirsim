param([ValidateRange(1,3)][int]$Sessions = 3, [string]$ExperimentConfig)
$ErrorActionPreference = 'Stop'
# A Windows PowerShell child may inherit the bundled PowerShell 7 module path.
# Restore its own built-in modules before hashing or writing JSON manifests.
if ($PSVersionTable.PSVersion.Major -le 5) {
    $taskBuiltinModules = Join-Path $PSHOME 'Modules'
    $env:PSModulePath = $taskBuiltinModules + [IO.Path]::PathSeparator + $env:PSModulePath
}
$env:PYTHONUTF8 = '1'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv\python.exe'
$taskEditor = 'D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$taskProject = Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskSettings = Join-Path $taskRoot 'configs\settings.json'
if (!$ExperimentConfig) { $ExperimentConfig = Join-Path $taskRoot 'configs\single_drone_matched_speed.json' }
$ExperimentConfig = (Resolve-Path -LiteralPath $ExperimentConfig).Path
$taskCondition = Get-Content -LiteralPath $ExperimentConfig -Raw | ConvertFrom-Json
$taskBalloon = Get-Content -LiteralPath (Join-Path $taskRoot 'configs\balloon.json') -Raw | ConvertFrom-Json
$taskOffset = @(0,1,2 | ForEach-Object {
    ([double]$taskCondition.target_world_ned_m[$_]) - ([double]$taskBalloon.position_ned_m[$_])
})
foreach ($taskPath in @($taskPython,$taskEditor,$taskProject,$taskSettings,$ExperimentConfig)) {
    if (!(Test-Path -LiteralPath $taskPath)) { throw "Missing: $taskPath" }
}
function Test-PilotRpc {
    $socket = [Net.Sockets.TcpClient]::new()
    try { return ($socket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $socket.Connected) }
    catch { return $false } finally { $socket.Dispose() }
}
if (Test-PilotRpc) { throw 'RPC 41451 is already occupied. Preserve the existing simulation before starting this pilot.' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$taskFolder = Join-Path $taskRoot "logs\phase1_pilot_$stamp"
New-Item -ItemType Directory -Path $taskFolder | Out-Null
$manifestPath = Join-Path $taskFolder 'manifest.json'
$manifest = [ordered]@{status='RUNNING'; planned_sessions=$Sessions; pairs_per_session=2;
    scope='Blocks single-target condition pilot'; started=(Get-Date).ToString('o');
    condition_id=$taskCondition.condition_id; target_offset_ned_m=$taskOffset;
    settings_sha256=(Get-FileHash -LiteralPath $taskSettings -Algorithm SHA256).Hash.ToLower();
    experiment_config=$ExperimentConfig;
    experiment_config_sha256=(Get-FileHash -LiteralPath $ExperimentConfig -Algorithm SHA256).Hash.ToLower();
    sessions=@()}
function Save-PilotManifest { $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8 }
Save-PilotManifest
Push-Location $taskRoot
try {
    for ($index=1; $index -le $Sessions; $index++) {
        if (Test-PilotRpc) { throw 'RPC remained occupied before the next independent session.' }
        $runtimeLog = Join-Path $taskFolder "session_${index}_unreal.log"
        $arguments = @(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap',
            '-game','-windowed','-ResX=1280','-ResY=720',('-settings="'+$taskSettings+'"'),
            ('-abslog="'+$runtimeLog+'"'),'-nosplash')
        $engine = Start-Process -FilePath $taskEditor -ArgumentList $arguments -WindowStyle Hidden -PassThru
        $session = [ordered]@{session_index=$index; status='STARTING'; engine_process_id=$engine.Id;
            engine_started=$engine.StartTime.ToUniversalTime().ToString('o'); runtime_log=$runtimeLog;
            batch_path=(Join-Path $taskRoot "logs\single_batch_${stamp}_session_$index")}
        $manifest.sessions += $session
        Save-PilotManifest
        Write-Host "Session $index/${Sessions}: starting Unreal process $($engine.Id)"
        $deadline = (Get-Date).AddSeconds(150)
        while (!(Test-PilotRpc)) {
            $engine.Refresh()
            if ($engine.HasExited) { throw "Unreal exited before RPC readiness. Inspect $runtimeLog" }
            if ((Get-Date) -gt $deadline) { throw "Unreal startup timed out. Inspect $runtimeLog" }
            Start-Sleep -Seconds 2
        }
        $previous = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'scene_*' | Select-Object -ExpandProperty FullName)
        & $taskPython (Join-Path $taskRoot 'python\setup_scene.py') --target-offset $taskOffset[0] $taskOffset[1] $taskOffset[2]
        if ($LASTEXITCODE -ne 0) { throw 'Scene preflight failed; no flight batch was started.' }
        $newScenes = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'scene_*' |
            Where-Object { $_.FullName -notin $previous })
        if ($newScenes.Count -ne 1) { throw 'Could not uniquely identify the scene preflight report.' }
        $sceneReport = Join-Path $newScenes[0].FullName 'scene_report.json'
        if ((Get-Content -LiteralPath $sceneReport -Raw | ConvertFrom-Json).status -ne 'PASS') { throw 'Scene preflight did not pass.' }
        $taskObserved = (Get-Content -LiteralPath $sceneReport -Raw | ConvertFrom-Json).balloon.observed_position_ned_m
        foreach ($axis in 0,1,2) {
            if ([math]::Abs($taskObserved[$axis] - $taskCondition.target_world_ned_m[$axis]) -gt 0.02) {
                throw 'Scene target does not match the experiment configuration.'
            }
        }
        $session.scene_report = $sceneReport
        $session.scene_report_sha256 = (Get-FileHash -LiteralPath $sceneReport -Algorithm SHA256).Hash.ToLower()
        $session.status = 'RUNNING_BATCH'
        $sessionFile = Join-Path $taskFolder "session_$index.json"
        $session | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
        Save-PilotManifest
        & $taskPython (Join-Path $taskRoot 'python\run_single_batch.py') --repeats 2 --output-dir $session.batch_path --session-record $sessionFile --config $ExperimentConfig
        if ($LASTEXITCODE -ne 0) { throw "Flight batch failed in session $index; paused simulation is preserved for inspection." }
        $batchFile = Join-Path $session.batch_path 'batch_report.json'
        $batch = Get-Content -LiteralPath $batchFile -Raw | ConvertFrom-Json
        if ($batch.status -ne 'PASS' -or $batch.runs.Count -ne 4) { throw 'Incomplete batch; preserve simulation.' }
        # All four runs have verified landing and released DroneA API control.
        # Close only the process launched and recorded by this script.
        $engine.Refresh()
        if ($engine.HasExited -or $engine.Path -ne $taskEditor -or
            $engine.StartTime.ToUniversalTime().ToString('o') -ne $session.engine_started) { throw 'Engine process identity changed.' }
        $session.status = 'PASS'
        $session.batch_sha256 = (Get-FileHash -LiteralPath $batchFile -Algorithm SHA256).Hash.ToLower()
        Save-PilotManifest
        Stop-Process -Id $engine.Id
        $engine.WaitForExit()
        $session.engine_closed_after_safe_batch = $true
        Save-PilotManifest
        Write-Host "Session ${index}: four flights safely completed."
    }
    $manifest.status='PASS'
} catch {
    $manifest.status='FAIL'
    $manifest.error=$_.Exception.Message
    if ($manifest.sessions.Count -gt 0 -and $manifest.sessions[-1].status -ne 'PASS') { $manifest.sessions[-1].status='STOPPED_FOR_INSPECTION' }
    throw
} finally {
    $manifest.finished=(Get-Date).ToString('o')
    Save-PilotManifest
    Pop-Location
    Write-Host "Pilot manifest: $manifestPath"
}
& $taskPython (Join-Path $taskRoot 'python\analyze_phase1_pilot.py') --manifest $manifestPath --output (Join-Path $taskFolder 'analysis')
if ($LASTEXITCODE -ne 0) { throw "Flights completed; derived analysis failed. Raw records are preserved at $taskFolder" }
