param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv/python.exe'
$taskEditor = 'D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'
$taskProject = Join-Path $taskRoot 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
$taskPlanPath = [IO.Path]::GetFullPath($Plan)
$taskPlan = Get-Content -LiteralPath $taskPlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
$env:PYTHONUTF8 = '1'
function Test-T1DataRpc {
    $taskSocket=[Net.Sockets.TcpClient]::new()
    try { return ($taskSocket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $taskSocket.Connected) }
    catch { return $false } finally { $taskSocket.Dispose() }
}
function Get-T1DataHash([string]$Path) { return (Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant() }
function Save-T1DataManifest { $taskManifest | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8 }
if ($taskPlan.status -ne 'FROZEN_BEFORE_COLLECTION_AND_TRAINING' -or !$taskPlan.ground_only) {throw 'Frozen ground-only plan required.'}
if ((Get-T1DataHash $taskPlan.settings_path) -ne $taskPlan.settings_sha256) {throw 'Settings changed.'}
if (Test-T1DataRpc) {throw 'RPC occupied: preserve current simulation.'}
if (Test-Path -LiteralPath $taskPlan.campaign_root) {throw 'Campaign destination already exists.'}
New-Item -ItemType Directory -Path $taskPlan.campaign_root | Out-Null
$taskCampaignReceipt=[ordered]@{status='RUNNING';plan=$taskPlanPath;plan_sha256=(Get-T1DataHash $taskPlanPath);
    started=(Get-Date).ToUniversalTime().ToString('o');new_flights=0;cloud_requests=0;attempts=@()}
$taskCampaignReceiptPath=Join-Path $taskPlan.campaign_root 'campaign_receipt.json'
function Save-T1DataCampaign { $taskCampaignReceipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $taskCampaignReceiptPath -Encoding UTF8 }
Save-T1DataCampaign
try {
foreach($taskEpisode in $taskPlan.episodes) {
    if (Test-T1DataRpc) {throw 'RPC occupied before next episode.'}
    if ((Get-T1DataHash $taskEpisode.protocol_path) -ne $taskEpisode.protocol_sha256) {throw 'Protocol changed.'}
    $taskFolder=Split-Path $taskEpisode.manifest_path -Parent
    New-Item -ItemType Directory -Path $taskFolder | Out-Null
    $taskManifestPath=$taskEpisode.manifest_path
    $taskManifest=[ordered]@{status='STARTING';kind='CAMERA_ONLY_DETECTOR_DATA';episode_id=$taskEpisode.episode_id;
        split=$taskEpisode.split;seed=$taskEpisode.seed;acquisition_seed=$taskEpisode.acquisition_seed;flight_requests=$false;formal_efficacy_results=$false;
        protocol=$taskEpisode.protocol_path;protocol_sha256=$taskEpisode.protocol_sha256;
        settings=$taskPlan.settings_path;settings_sha256=$taskPlan.settings_sha256;
        plan=$taskPlanPath;plan_sha256=$taskCampaignReceipt.plan_sha256;launcher_sha256=(Get-T1DataHash $PSCommandPath);
        probe_output=(Join-Path $taskFolder 'probe');engine_closed_after_safe_completion=$false}
    $taskCampaignReceipt.attempts+=@{episode_id=$taskEpisode.episode_id;manifest_path=$taskManifestPath}
    Save-T1DataCampaign;Save-T1DataManifest
    $taskSnapshot=Join-Path $taskFolder 'source_snapshot'
    New-Item -ItemType Directory -Path $taskSnapshot | Out-Null
    foreach($taskName in @('t1_ground_perception_probe.py','setup_scene.py','camera_geometry.py','braking_startup_readiness.py','t1_balloon_data_probe.py')) {
        Copy-Item -LiteralPath (Join-Path $taskRoot ('python/'+$taskName)) -Destination (Join-Path $taskSnapshot $taskName)
    }
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskSnapshot 'run-t1-balloon-data-campaign.ps1')
    try {
        $taskArguments=@(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap','-game','-windowed','-ResX=1280','-ResY=720',
            ('-settings="'+$taskPlan.settings_path+'"'),('-abslog="'+(Join-Path $taskFolder 'unreal.log')+'"'),'-nosplash')
        $taskEngine=Start-Process -FilePath $taskEditor -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
        $taskManifest.engine_process_id=$taskEngine.Id
        $taskManifest.engine_started_utc=$taskEngine.StartTime.ToUniversalTime().ToString('o')
        Save-T1DataManifest
        Write-Host "Collect $($taskEpisode.episode_id), owned UE PID $($taskEngine.Id)"
        $taskDeadline=(Get-Date).AddSeconds(150)
        while (!(Test-T1DataRpc)) {
            $taskEngine.Refresh()
            if ($taskEngine.HasExited -or (Get-Date) -gt $taskDeadline) {throw 'UE startup failed/timed out.'}
            Start-Sleep -Seconds 2
        }
        & $taskPython (Join-Path $taskRoot 'python/braking_startup_readiness.py') --output (Join-Path $taskFolder 'ground_readiness.json')
        if ($LASTEXITCODE -ne 0) {throw 'Ground readiness failed.'}
        $taskManifest.status='CAPTURING';Save-T1DataManifest
        & $taskPython (Join-Path $taskRoot 'python/t1_balloon_data_probe.py') --output $taskManifest.probe_output --protocol $taskEpisode.protocol_path
        $taskProbeExit=$LASTEXITCODE
        if (Test-Path -LiteralPath $taskEpisode.report_path) {
            $taskReport=Get-Content -LiteralPath $taskEpisode.report_path -Raw -Encoding UTF8 | ConvertFrom-Json
            $taskManifest.report=$taskEpisode.report_path;$taskManifest.report_sha256=Get-T1DataHash $taskEpisode.report_path
            $taskManifest.saved_frames=$taskReport.frames.Count
            if ($taskReport.safe_ground_closure -and $taskReport.paused_after) {
                $taskEngine.Refresh()
                if ($taskEngine.HasExited -or $taskEngine.Path -ne ([IO.Path]::GetFullPath($taskEditor)) -or
                    $taskEngine.StartTime.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskManifest.engine_started_utc).UtcTicks) {throw 'Owned process identity mismatch.'}
                Stop-Process -Id $taskEngine.Id
                $taskEngine.WaitForExit()
                $taskManifest.engine_closed_after_safe_completion=$true
            }
        }
        if ($taskProbeExit -ne 0 -or $taskReport.status -ne 'PASS' -or !$taskManifest.engine_closed_after_safe_completion) {throw 'Probe failed or ground closure missing.'}
        $taskManifest.status='PASS'
    } catch {$taskManifest.status='FAIL';$taskManifest.error=$_.Exception.Message;throw}
    finally {$taskManifest.finished_utc=(Get-Date).ToUniversalTime().ToString('o');Save-T1DataManifest}
    Save-T1DataCampaign
}
$taskCampaignReceipt.status='PASS'
} catch {$taskCampaignReceipt.status='FAIL';$taskCampaignReceipt.error=$_.Exception.Message;throw}
finally {$taskCampaignReceipt.finished_utc=(Get-Date).ToUniversalTime().ToString('o');Save-T1DataCampaign}
