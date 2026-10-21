param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskPython=Join-Path $taskRoot '.venv/python.exe'
$taskEditor='D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'
$taskProject=Join-Path $taskRoot 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
$taskPlanPath=[IO.Path]::GetFullPath($Plan)
$taskPlan=Get-Content -LiteralPath $taskPlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
$env:PYTHONUTF8='1'
$env:PYTHONPATH=Join-Path $taskRoot 'python'
$env:YOLO_CONFIG_DIR=Join-Path $taskRoot 'logs/t1_model_runtime/ultralytics'
$env:TORCH_HOME=Join-Path $taskRoot 'logs/t1_model_runtime/torch'
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force | Out-Null
function Test-T1IdentityRpc {
    $taskSocket=[Net.Sockets.TcpClient]::new()
    try{return ($taskSocket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $taskSocket.Connected)}
    catch{return $false}finally{$taskSocket.Dispose()}
}
function Get-T1IdentityHash([string]$Path){return (Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant()}
function Save-T1IdentityManifest{$taskManifest | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8}
function Save-T1IdentityCampaign{$taskCampaign | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $taskCampaignPath -Encoding UTF8}
function Test-T1IdentityInventory {
    if((Get-T1IdentityHash $taskPlan.settings_path) -ne $taskPlan.settings_sha256){throw 'Settings changed after protocol freeze.'}
    if((Get-T1IdentityHash $taskPlan.weights_config_path) -ne $taskPlan.weights_config_sha256){throw 'Model configuration changed after protocol freeze.'}
    if((Get-T1IdentityHash $taskPlan.weights_path) -ne $taskPlan.weights_sha256){throw 'Checkpoint changed after protocol freeze.'}
    foreach($taskSource in $taskPlan.source_inventory){
        if((Get-T1IdentityHash $taskSource.path) -ne $taskSource.sha256){throw ('Declared source changed: '+$taskSource.relative_path)}
    }
}
if($taskPlan.status -ne 'FROZEN_BEFORE_SEQUENCE_COLLECTION' -or !$taskPlan.ground_only -or $taskPlan.planned_UE_sessions -ne 4){throw 'Frozen four-session ground-only plan required.'}
Test-T1IdentityInventory
if(Test-T1IdentityRpc){throw 'RPC occupied: preserve current simulation.'}
if(Test-Path -LiteralPath $taskPlan.campaign_root){throw 'Campaign destination already exists.'}
New-Item -ItemType Directory -Path $taskPlan.campaign_root | Out-Null
$taskCampaignPath=Join-Path $taskPlan.campaign_root 'campaign_receipt.json'
$taskCampaign=[ordered]@{status='RUNNING';kind='T1_GROUND_CONTINUOUS_IDENTITY_CAMPAIGN';plan=$taskPlanPath;
    plan_sha256=(Get-T1IdentityHash $taskPlanPath);started_utc=(Get-Date).ToUniversalTime().ToString('o');
    new_flights=0;formal_efficacy_results=$false;planned_UE_sessions=4;attempts=@()}
Save-T1IdentityCampaign
try{
foreach($taskEpisode in $taskPlan.episodes){
    Test-T1IdentityInventory
    if(Test-T1IdentityRpc){throw 'RPC occupied before the next independent UE session.'}
    if((Get-T1IdentityHash $taskEpisode.protocol_path) -ne $taskEpisode.protocol_sha256){throw 'Sequence protocol changed.'}
    $taskFolder=Split-Path $taskEpisode.manifest_path -Parent
    New-Item -ItemType Directory -Path $taskFolder | Out-Null
    $taskManifestPath=$taskEpisode.manifest_path
    $taskManifest=[ordered]@{status='STARTING';kind='T1_GROUND_CONTINUOUS_IDENTITY_DEVELOPMENT';
        episode_id=$taskEpisode.episode_id;condition=$taskEpisode.condition;acquisition_seed=$taskEpisode.acquisition_seed;
        recovery_observability=$taskEpisode.recovery_observability;new_flights=0;formal_efficacy_results=$false;
        planned_slots=60;protocol=$taskEpisode.protocol_path;protocol_sha256=$taskEpisode.protocol_sha256;
        settings=$taskPlan.settings_path;settings_sha256=$taskPlan.settings_sha256;
        weights_config=$taskPlan.weights_config_path;weights_config_sha256=$taskPlan.weights_config_sha256;
        weights=$taskPlan.weights_path;weights_sha256=$taskPlan.weights_sha256;
        plan=$taskPlanPath;plan_sha256=$taskCampaign.plan_sha256;launcher_sha256=(Get-T1IdentityHash $PSCommandPath);
        probe_output=(Join-Path $taskFolder 'probe');engine_closed_after_safe_completion=$false}
    $taskCampaign.attempts+=@{episode_id=$taskEpisode.episode_id;manifest_path=$taskManifestPath}
    Save-T1IdentityCampaign
    Save-T1IdentityManifest
    $taskSnapshot=Join-Path $taskFolder 'source_snapshot'
    New-Item -ItemType Directory -Path $taskSnapshot | Out-Null
    foreach($taskSource in $taskPlan.source_inventory){
        $taskSnapshotTarget=Join-Path $taskSnapshot $taskSource.relative_path
        New-Item -ItemType Directory -Path (Split-Path $taskSnapshotTarget -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $taskSource.path -Destination $taskSnapshotTarget
        if((Get-T1IdentityHash $taskSnapshotTarget) -ne $taskSource.sha256){throw 'Source snapshot mismatch.'}
    }
    foreach($taskSnapshotFile in @($taskEpisode.protocol_path,$taskPlan.settings_path,$taskPlan.weights_config_path,$taskPlanPath)){
        Copy-Item -LiteralPath $taskSnapshotFile -Destination (Join-Path $taskSnapshot ([IO.Path]::GetFileName($taskSnapshotFile)))
    }
    Copy-Item -LiteralPath $taskPlan.weights_path -Destination (Join-Path $taskSnapshot 'specialist_best.pt')
    if((Get-T1IdentityHash (Join-Path $taskSnapshot 'specialist_best.pt')) -ne $taskPlan.weights_sha256){throw 'Checkpoint snapshot mismatch.'}
    try{
        $taskArguments=@(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap','-game','-windowed','-ResX=1280','-ResY=720',
            ('-settings="'+$taskPlan.settings_path+'"'),('-abslog="'+(Join-Path $taskFolder 'unreal.log')+'"'),'-nosplash')
        $taskEngine=Start-Process -FilePath $taskEditor -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
        $taskManifest.engine_process_id=$taskEngine.Id
        $taskManifest.engine_started_utc=$taskEngine.StartTime.ToUniversalTime().ToString('o')
        Save-T1IdentityManifest
        Write-Host "Continuous identity $($taskEpisode.episode_id), owned UE PID $($taskEngine.Id)"
        $taskDeadline=(Get-Date).AddSeconds(150)
        while(!(Test-T1IdentityRpc)){
            $taskEngine.Refresh()
            if($taskEngine.HasExited -or (Get-Date) -gt $taskDeadline){throw 'UE startup failed or timed out.'}
            Start-Sleep -Milliseconds 200
        }
        & $taskPython (Join-Path $taskRoot 'python/braking_startup_readiness.py') --output (Join-Path $taskFolder 'ground_readiness.json')
        if($LASTEXITCODE -ne 0){throw 'Ground readiness failed before sequence acquisition.'}
        $taskManifest.status='CAPTURING'
        Save-T1IdentityManifest
        & $taskPython (Join-Path $taskRoot 'python/t1_identity_sequence_probe.py') --output $taskManifest.probe_output --protocol $taskEpisode.protocol_path --observer-module $taskPlan.observer_module --weights-config $taskPlan.weights_config_path
        $taskProbeExit=$LASTEXITCODE
        $taskReport=$null
        if(Test-Path -LiteralPath $taskEpisode.report_path){
            $taskReport=Get-Content -LiteralPath $taskEpisode.report_path -Raw -Encoding UTF8 | ConvertFrom-Json
            $taskManifest.report=$taskEpisode.report_path
            $taskManifest.report_sha256=Get-T1IdentityHash $taskEpisode.report_path
            $taskManifest.saved_frames=$taskReport.frames.Count
            $taskManifest.scheduled_slots=$taskReport.scheduled_slot_count
            $taskManifest.not_captured_slots=$taskReport.not_captured_slot_count
            # Failed probe evidence is retained; ownership and fresh closure proof
            # still permit closing only the UE process launched by this script.
            if($taskReport.safe_ground_closure -and $taskReport.paused_after){
                $taskEngine.Refresh()
                if($taskEngine.HasExited -or $taskEngine.Path -ne ([IO.Path]::GetFullPath($taskEditor)) -or
                    $taskEngine.StartTime.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskManifest.engine_started_utc).UtcTicks){throw 'Owned process path/start time mismatch: preserve process.'}
                Stop-Process -Id $taskEngine.Id
                $taskEngine.WaitForExit()
                $taskManifest.engine_closed_after_safe_completion=$true
            }
        }
        if($taskProbeExit -ne 0 -or $null -eq $taskReport -or $taskReport.status -ne 'PASS' -or !$taskManifest.engine_closed_after_safe_completion){throw 'Sequence failed or safe ground closure missing; preserve all attempt receipts.'}
        $taskManifest.status='PASS'
    }catch{
        $taskManifest.status='FAIL'
        $taskManifest.error_type=$_.Exception.GetType().Name
        throw
    }finally{
        $taskManifest.finished_utc=(Get-Date).ToUniversalTime().ToString('o')
        Save-T1IdentityManifest
    }
    Save-T1IdentityCampaign
}
$taskCampaign.status='PASS'
}catch{
    $taskCampaign.status='FAIL'
    $taskCampaign.error_type=$_.Exception.GetType().Name
    throw
}finally{
    $taskCampaign.finished_utc=(Get-Date).ToUniversalTime().ToString('o')
    Save-T1IdentityCampaign
}
