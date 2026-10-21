param(
    [Parameter(Mandatory=$true)][string]$Prepared,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference='Stop'
$taskRoot=[IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$taskPython=Join-Path $taskRoot '.venv/python.exe'
$taskEditor=[IO.Path]::GetFullPath('D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe')
$taskProject=Join-Path $taskRoot 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
$taskPrepared=[IO.Path]::GetFullPath($Prepared)
$taskOutput=[IO.Path]::GetFullPath($Output)
$taskPlanPath=Join-Path $taskPrepared 'plan.json'
$taskPlan=Get-Content -LiteralPath $taskPlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
$env:PYTHONUTF8='1'
$env:PYTHONPATH=Join-Path $taskRoot 'python'
$env:YOLO_CONFIG_DIR=Join-Path $taskRoot 'logs/t1_model_runtime/ultralytics'
$env:TORCH_HOME=Join-Path $taskRoot 'logs/t1_model_runtime/torch'

function Get-T1FlightHash([string]$Path){
    return (Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant()
}
function Test-T1FlightRpc {
    $taskSocket=[Net.Sockets.TcpClient]::new()
    try{return ($taskSocket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $taskSocket.Connected)}
    catch{return $false}finally{$taskSocket.Dispose()}
}
function Save-T1FlightManifest {
    $taskManifest | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8
}
function Assert-T1FlightArtifact([string]$Path,[string]$Hash) {
    if($Hash -notmatch '^[0-9a-f]{64}$' -or (Get-T1FlightHash $Path) -cne $Hash){
        throw 'Declared artifact digest changed after freeze.'
    }
}
function Assert-T1FlightSource {
    Assert-T1FlightArtifact $taskPlanPath $taskExpectedPlanHash
    Assert-T1FlightArtifact $taskPlan.settings_path $taskPlan.settings_sha256
    Assert-T1FlightArtifact $taskPlan.detector_config_path $taskPlan.detector_config_sha256
    Assert-T1FlightArtifact $taskPlan.weights_path $taskPlan.weights_sha256
    Assert-T1FlightArtifact $taskPlan.training_receipt_path $taskPlan.training_receipt_sha256
    Assert-T1FlightArtifact $taskPlan.legacy_freeze_path $taskPlan.legacy_freeze_sha256
    foreach($taskSource in $taskPlan.source_inventory){
        if($taskSource.relative_path -notmatch '^(python/[A-Za-z0-9_]+\.py|scripts/[A-Za-z0-9_-]+\.ps1)$'){
            throw 'Source inventory contains an undeclared or secret-bearing path.'
        }
        $taskExpectedPath=[IO.Path]::GetFullPath((Join-Path $taskRoot $taskSource.relative_path))
        if([IO.Path]::GetFullPath($taskSource.path) -ne $taskExpectedPath){throw 'Source inventory path does not match workspace declaration.'}
        Assert-T1FlightArtifact $taskSource.path $taskSource.sha256
        Assert-T1FlightArtifact $taskSource.snapshot_path $taskSource.sha256
        $taskProperty=$taskPlan.source_sha256.PSObject.Properties[$taskSource.relative_path]
        if($null -eq $taskProperty -or $taskProperty.Value -cne $taskSource.sha256){throw 'Source inventory and source_sha256 map differ.'}
    }
    if($taskPlan.legacy_source_inventory.Count -ne 12){throw 'Exactly twelve historical module locks required.'}
    foreach($taskSource in $taskPlan.legacy_source_inventory){
        Assert-T1FlightArtifact $taskSource.path $taskSource.sha256
        Assert-T1FlightArtifact $taskSource.snapshot_path $taskSource.sha256
    }
    $taskLegacy=Get-Content -LiteralPath $taskPlan.legacy_freeze_path -Raw -Encoding UTF8 | ConvertFrom-Json
    & (Join-Path $taskRoot 'scripts/assert-trajectory-source-freeze.ps1') -SourceHashes $taskLegacy.source_sha256
}
function Assert-T1FlightOwner {
    $taskEngine.Refresh()
    if($taskEngine.HasExited -or $taskEngine.Id -ne $taskManifest.engine_process_id -or
        $taskEngine.Path -ne $taskEditor -or
        $taskManifest.engine_executable_path -ne $taskEditor -or
        $taskEngine.StartTime.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskManifest.engine_started_utc).UtcTicks){
        throw 'Owned UE PID, path, or UTC start time differs; preserve process.'
    }
}

$taskExpectedPlanHash=Get-T1FlightHash $taskPlanPath
if($taskPlan.kind -ne 'T1_IDENTITY_LOW_SPEED_FLIGHT_PILOT' -or
    $taskPlan.status -ne 'FROZEN_BEFORE_FLIGHT' -or
    $taskPlan.mode -notin @('ZERO_GATE','SHORT_XY') -or $taskPlan.formal_efficacy_results -ne $false -or
    $taskPlan.planned_UE_sessions -ne 1 -or $taskPlan.planned_flights -ne 1){
    throw 'Frozen single-session development flight plan required.'
}
if($taskPlan.control_architecture -ne 'INDEPENDENT_CONTROL_OWN_STATE_BOUNDED_PERSISTENCE_V10P1'){
    throw 'Independent control review-handoff V10 bounded persistence strict-stop plan required.'
}
if($taskPlan.period_s -ne .4 -or $taskPlan.world_z_ned_m -ne -3.0 -or
    $taskPlan.command_speed_cap_m_s -ne .03 -or $taskPlan.command_lease_s -ne .12 -or
    $taskPlan.self_monitor_period_s -ne .05){throw 'Declared low-speed pilot timing or command budget differs.'}
if(Test-Path -LiteralPath $taskOutput){throw 'Output directory already exists; do not overwrite evidence.'}
Assert-T1FlightSource
if(Test-T1FlightRpc){throw 'RPC occupied; preserve the existing simulation.'}
New-Item -ItemType Directory -Path $taskOutput | Out-Null
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force | Out-Null
$taskManifestPath=Join-Path $taskOutput 'manifest.json'
$taskManifest=[ordered]@{
    kind='T1_IDENTITY_LOW_SPEED_FLIGHT_ATTEMPT';status='PREPARING';session_id=$taskPlan.session_id;
    mode=$taskPlan.mode;plan=$taskPlanPath;plan_sha256=(Get-T1FlightHash $taskPlanPath);
    started_utc=(Get-Date).ToUniversalTime().ToString('o');formal_efficacy_results=$false;
    launcher_sha256=(Get-T1FlightHash $PSCommandPath);source_lock_scope=$taskPlan.source_lock_scope;
    runner_output=(Join-Path $taskOutput 'flight');runner_started=$false;runner_exit_code=$null;
    engine_closed_after_safe_completion=$false;fresh_ground_closure_verified=$false;
    planned_flights=1;actual_flight_status='NOT_RUNNER_STARTED'
}
Save-T1FlightManifest
$taskEngine=$null
try {
    $taskSnapshot=Join-Path $taskOutput 'source_snapshot'
    New-Item -ItemType Directory -Path $taskSnapshot | Out-Null
    foreach($taskSource in $taskPlan.source_inventory){
        $taskDestination=Join-Path $taskSnapshot $taskSource.relative_path
        New-Item -ItemType Directory -Path (Split-Path $taskDestination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $taskSource.snapshot_path -Destination $taskDestination
        Assert-T1FlightArtifact $taskDestination $taskSource.sha256
    }
    foreach($taskSource in $taskPlan.legacy_source_inventory){
        $taskDestination=Join-Path $taskSnapshot ('legacy_v6p2/'+[IO.Path]::GetFileName($taskSource.path))
        New-Item -ItemType Directory -Path (Split-Path $taskDestination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $taskSource.snapshot_path -Destination $taskDestination
        Assert-T1FlightArtifact $taskDestination $taskSource.sha256
    }
    foreach($taskArtifact in @($taskPlanPath,$taskPlan.settings_path,$taskPlan.detector_config_path,
        $taskPlan.training_receipt_path,$taskPlan.legacy_freeze_path)){
        $taskDestination=Join-Path $taskSnapshot ([IO.Path]::GetFileName($taskArtifact))
        Copy-Item -LiteralPath $taskArtifact -Destination $taskDestination
        Assert-T1FlightArtifact $taskDestination (Get-T1FlightHash $taskArtifact)
    }
    $taskWeightSnapshot=Join-Path $taskSnapshot 'specialist_best.pt'
    Copy-Item -LiteralPath $taskPlan.weights_path -Destination $taskWeightSnapshot
    Assert-T1FlightArtifact $taskWeightSnapshot $taskPlan.weights_sha256
    Assert-T1FlightSource
    if(Test-T1FlightRpc){throw 'RPC became occupied before owned UE launch.'}
    $taskArguments=@(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap','-game','-windowed',
        '-ResX=1280','-ResY=720',('-settings="'+$taskPlan.settings_path+'"'),
        ('-abslog="'+(Join-Path $taskOutput 'unreal.log')+'"'),'-nosplash')
    & $taskPython (Join-Path $taskRoot 'python/t1_pipeline_plan_v1p1.py') --plan $taskPlanPath --output (Join-Path $taskOutput 'plan_preflight.json') 1> (Join-Path $taskOutput 'plan_preflight.stdout.log') 2> (Join-Path $taskOutput 'plan_preflight.stderr.log')
    if($LASTEXITCODE -ne 0){throw 'Complete motion plan failed before UE startup; no flight.'}
    $taskManifest.plan_preflight_path=Join-Path $taskOutput 'plan_preflight.json'
    $taskManifest.plan_preflight_sha256=Get-T1FlightHash $taskManifest.plan_preflight_path
    Save-T1FlightManifest
    $taskEngine=Start-Process -FilePath $taskEditor -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
    $taskManifest.engine_process_id=$taskEngine.Id
    $taskManifest.engine_started_utc=$taskEngine.StartTime.ToUniversalTime().ToString('o')
    # Query the launched process afresh; the initial Process.Path can be null.
    $taskOwnerDeadline=(Get-Date).AddSeconds(5)
    $taskFreshOwner=$null
    while((Get-Date) -lt $taskOwnerDeadline){
        $taskFreshOwner=Get-Process -Id $taskEngine.Id -ErrorAction Stop
        if($taskFreshOwner.Path){break}
        Start-Sleep -Milliseconds 50
    }
    if($null -eq $taskFreshOwner -or !$taskFreshOwner.Path -or
        [IO.Path]::GetFullPath($taskFreshOwner.Path) -ne $taskEditor -or
        $taskFreshOwner.StartTime.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskManifest.engine_started_utc).UtcTicks){
        throw 'Fresh owned UE executable path and UTC start required; preserve process.'
    }
    $taskManifest.engine_executable_path=$taskFreshOwner.Path
    $taskManifest.engine_identity_verified_utc=(Get-Date).ToUniversalTime().ToString('o')
    $taskManifest.engine_identity_receipt_scope='FRESH_PROCESS_QUERY_BEFORE_RPC_AND_FLIGHT' 
    $taskManifest.status='ENGINE_STARTING'
    Save-T1FlightManifest
    Write-Host "T1 $($taskPlan.mode), owned UE PID $($taskEngine.Id)"
    $taskDeadline=(Get-Date).AddSeconds(150)
    while(!(Test-T1FlightRpc)){
        Assert-T1FlightOwner
        if((Get-Date) -gt $taskDeadline){throw 'Owned UE RPC startup timed out.'}
        Start-Sleep -Milliseconds 200
    }
    Assert-T1FlightOwner
    Assert-T1FlightSource
    & $taskPython (Join-Path $taskRoot 'python/braking_startup_readiness.py') --output (Join-Path $taskOutput 'ground_readiness.json') 1> (Join-Path $taskOutput 'ground_readiness.stdout.log') 2> (Join-Path $taskOutput 'ground_readiness.stderr.log')
    if($LASTEXITCODE -ne 0){throw 'Ground readiness failed; preserve the owned UE and evidence.'}
    $taskReadiness=Get-Content -LiteralPath (Join-Path $taskOutput 'ground_readiness.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if($taskReadiness.status -ne 'READY'){throw 'Verified ground readiness missing before flight.'}
    $taskManifest.status='RUNNING'
    $taskManifest.runner_started=$true
    $taskManifest.actual_flight_status='SEE_RUNNER_REPORT; NOT_INFERRED_FROM_PROCESS_LAUNCH'
    Save-T1FlightManifest
    & $taskPython (Join-Path $taskRoot 'python/t1_identity_flight_v10p1.py') --plan $taskPlanPath --output $taskManifest.runner_output 1> (Join-Path $taskOutput 'runner.stdout.log') 2> (Join-Path $taskOutput 'runner.stderr.log')
    $taskManifest.runner_exit_code=$LASTEXITCODE
    $taskReportPath=Join-Path $taskManifest.runner_output 'report.json'
    $taskReport=$null
    if(Test-Path -LiteralPath $taskReportPath){
        $taskReport=Get-Content -LiteralPath $taskReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $taskManifest.report=$taskReportPath
        $taskManifest.report_sha256=Get-T1FlightHash $taskReportPath
        $taskManifest.runner_status=$taskReport.status
        $taskManifest.safe_end=$taskReport.safe_end
    }
    if($null -ne $taskReport -and $taskReport.safe_end.grounded_and_released -is [bool] -and
        $taskReport.safe_end.grounded_and_released -eq $true -and $taskReport.safe_end.paused -is [bool] -and
        $taskReport.safe_end.paused -eq $true){
        Assert-T1FlightOwner
        # A new Python process performs read-only vehicle/self-state checks.
        # It neither arms nor acquires control, moves an actor, nor unpauses UE.
        $taskClosurePath=Join-Path $taskOutput 'fresh_ground_closure.json'
        $taskClosureCode=@'
import json, sys, time
from pathlib import Path
import airsim
from braking_startup_readiness import vehicle_sample
output=Path(sys.argv[1])
if output.exists(): raise SystemExit(3)
receipt={'kind':'FRESH_PAUSED_GROUND_ONLY_CLOSURE_PROOF','status':'NOT_READY',
    'arm_requests':False,'control_acquisition':False,'motion_requests':False,'unpause_requests':False,'samples':[]}
try:
    client=airsim.MultirotorClient(timeout_value=5)
    client.ping()
    stable=None
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        row={'readback_completed_host_s':None,'paused':bool(client.simIsPause()),'vehicles':{}}
        for name in ('DroneA','DroneB'): row['vehicles'][name]=vehicle_sample(client,name)
        row['readback_completed_host_s']=time.monotonic()
        receipt['samples'].append(row)
        ready=row['paused'] and all(v['ground_readiness'] for v in row['vehicles'].values())
        stable=(row['readback_completed_host_s'] if stable is None else stable) if ready else None
        if stable is not None and row['readback_completed_host_s']-stable>=.5:
            receipt.update(status='READY',grounded_and_released=True,paused=True,
                stable_query_receipt_span_s=row['readback_completed_host_s']-stable)
            break
        time.sleep(.1)
except Exception as error:
    receipt['error_type']=type(error).__name__
finally:
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
raise SystemExit(0 if receipt['status']=='READY' else 2)
'@
        & $taskPython -c $taskClosureCode $taskClosurePath 1> (Join-Path $taskOutput 'fresh_ground_closure.stdout.log') 2> (Join-Path $taskOutput 'fresh_ground_closure.stderr.log')
        if($LASTEXITCODE -ne 0){throw 'Fresh ground-only closure proof failed; preserve owned UE.'}
        $taskClosure=Get-Content -LiteralPath $taskClosurePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if($taskClosure.status -ne 'READY' -or $taskClosure.grounded_and_released -ne $true -or $taskClosure.paused -ne $true){throw 'Fresh ground/released/paused proof missing.'}
        $taskManifest.fresh_ground_closure_verified=$true
        $taskManifest.fresh_ground_closure_path=$taskClosurePath
        $taskManifest.fresh_ground_closure_sha256=Get-T1FlightHash $taskClosurePath
        Save-T1FlightManifest
        Assert-T1FlightOwner
        Stop-Process -Id $taskEngine.Id
        $taskEngine.WaitForExit()
        $taskManifest.engine_closed_after_safe_completion=$true
    }
    if($taskManifest.runner_exit_code -ne 0 -or $null -eq $taskReport -or $taskReport.status -ne 'PASS' -or
        !$taskManifest.engine_closed_after_safe_completion){throw 'Flight pilot failed or safe closure missing; all receipts retained.'}
    $taskManifest.status='PASS'
}catch{
    $taskManifest.status='FAIL'
    $taskManifest.error_type=$_.Exception.GetType().Name
    throw
}finally{
    $taskManifest.finished_utc=(Get-Date).ToUniversalTime().ToString('o')
    Save-T1FlightManifest
}
