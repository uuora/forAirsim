param([Parameter(Mandatory=$true)][string[]]$Configs,
    [ValidateSet('braking_response_experiment.py','braking_response_experiment_v1p1.py','braking_response_experiment_v1p2.py','braking_response_experiment_v1p3.py')]
    [string]$Runner='braking_response_experiment.py')
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskPython=Join-Path $taskRoot '.venv\python.exe'
$taskEditor='D:\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$taskProject=Join-Path $taskRoot 'external\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$taskSettings=Join-Path $taskRoot 'configs\settings_trajectory_interface.json'
function Test-BrakingRpc {
    $taskSocket=[Net.Sockets.TcpClient]::new()
    try{return ($taskSocket.ConnectAsync('127.0.0.1',41451).Wait(400) -and $taskSocket.Connected)}
    catch{return $false} finally{$taskSocket.Dispose()}
}
if(Test-BrakingRpc){throw 'RPC occupied; preserve the current simulator.'}
$taskFolder=Join-Path $taskRoot ('logs\braking_campaign_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'))
New-Item -ItemType Directory -Path $taskFolder | Out-Null
$taskManifest=[ordered]@{kind='INDEPENDENT_BRAKING_RESPONSE_DEVELOPMENT';status='RUNNING';
    formal_thesis_effect_results=$false;started=(Get-Date).ToString('o');
    settings_path=$taskSettings;settings_sha256=(Get-FileHash -LiteralPath $taskSettings).Hash.ToLowerInvariant();
    launcher_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant();
    runner=$Runner;runner_sha256=(Get-FileHash -LiteralPath (Join-Path $taskRoot ('python\'+$Runner))).Hash.ToLowerInvariant();
    startup_readiness_source_sha256=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'python\braking_startup_readiness.py')).Hash.ToLowerInvariant();sessions=@()}
foreach($taskConfigName in $Configs){
    $taskConfigPath=(Resolve-Path -LiteralPath $taskConfigName).Path
    $taskConfig=Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
    & (Join-Path $PSScriptRoot 'assert-trajectory-source-freeze.ps1') -SourceHashes $taskConfig.flight_source_sha256
    $taskManifest.sessions += [ordered]@{configuration=$taskConfigPath;
        configuration_sha256=(Get-FileHash -LiteralPath $taskConfigPath).Hash.ToLowerInvariant();
        status='NOT_STARTED';planned_pulses=@($taskConfig.braking_measurement.pulses).Count}
}
$taskManifestPath=Join-Path $taskFolder 'manifest.json'
function Save-BrakingManifest {$taskManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8}
Save-BrakingManifest
Push-Location $taskRoot
try{
    for($taskIndex=0;$taskIndex -lt $taskManifest.sessions.Count;$taskIndex++){
        $taskSession=$taskManifest.sessions[$taskIndex]
        $taskConfig=Get-Content -LiteralPath $taskSession.configuration -Raw | ConvertFrom-Json
        & (Join-Path $PSScriptRoot 'assert-trajectory-source-freeze.ps1') -SourceHashes $taskConfig.flight_source_sha256
        if((Get-FileHash -LiteralPath $taskSession.configuration).Hash.ToLowerInvariant() -ne $taskSession.configuration_sha256){throw 'Config freeze changed'}
        if(Test-BrakingRpc){throw 'RPC occupied before independent session'}
        $taskLog=Join-Path $taskFolder "session_${taskIndex}_unreal.log"
        $taskArgs=@(('"'+$taskProject+'"'),'/Game/FlyingCPP/Maps/FlyingExampleMap','-game','-windowed','-ResX=1280','-ResY=720',('-settings="'+$taskSettings+'"'),('-abslog="'+$taskLog+'"'))
        $taskEngine=Start-Process -FilePath $taskEditor -ArgumentList $taskArgs -WindowStyle Hidden -PassThru
        $taskSession.status='ENGINE_STARTED';$taskSession.engine_process_id=$taskEngine.Id
        $taskSession.engine_started=$taskEngine.StartTime.ToUniversalTime().ToString('o')
        Save-BrakingManifest
        $taskDeadline=(Get-Date).AddSeconds(120)
        while(!(Test-BrakingRpc)){
            $taskEngine.Refresh()
            if($taskEngine.HasExited -or (Get-Date) -gt $taskDeadline){throw 'Owned simulator failed to become ready'}
            Start-Sleep -Milliseconds 500
        }
        $taskReadiness=Join-Path $taskFolder "session_${taskIndex}_ground_readiness.json"
        & $taskPython (Join-Path $taskRoot 'python\braking_startup_readiness.py') --output $taskReadiness
        $taskSession.ground_readiness_path=$taskReadiness
        $taskSession.ground_readiness_sha256=(Get-FileHash -LiteralPath $taskReadiness).Hash.ToLowerInvariant()
        Save-BrakingManifest
        if($LASTEXITCODE -ne 0){throw 'Ground startup readiness failed; flight not dispatched'}
        $taskSession.status='TRIAL_DISPATCH_STARTED';Save-BrakingManifest
        $taskPointer=Join-Path $taskFolder "session_${taskIndex}_trial.json"
        & $taskPython (Join-Path $taskRoot ('python\'+$Runner)) --config $taskSession.configuration --receipt-path $taskPointer
        $taskExit=$LASTEXITCODE
        if(!(Test-Path -LiteralPath $taskPointer)){throw 'Trial pointer absent; flight state not assumed'}
        $taskTrial=(Get-Content -LiteralPath $taskPointer -Raw | ConvertFrom-Json).report_path
        $taskSession.report_path=$taskTrial
        if(!(Test-Path -LiteralPath $taskTrial)){throw 'Terminal report absent; preserve simulator'}
        $taskRaw=Get-Content -LiteralPath $taskTrial -Raw | ConvertFrom-Json
        $taskSession.report_sha256=(Get-FileHash -LiteralPath $taskTrial).Hash.ToLowerInvariant()
        $taskSession.measurement_passed=$taskRaw.measurement_passed
        $taskSession.flight_dispatch_started=$taskRaw.flight_dispatch_started
        $taskSession.status='TERMINAL_RECORDED';Save-BrakingManifest
        if($taskRaw.status -ne 'PASS' -or !$taskRaw.paused -or !$taskRaw.landing_evidence.landed_flag -or !$taskRaw.landing_evidence.api_control_released){throw 'Safe landing/release not verified; preserve simulator'}
        $taskEngine.Refresh()
        $taskStartedUtc=([datetimeoffset]$taskSession.engine_started).UtcTicks
        if($taskEngine.HasExited -or $taskEngine.Path -ne $taskEditor -or $taskEngine.StartTime.ToUniversalTime().Ticks -ne $taskStartedUtc){throw 'Owned engine identity changed'}
        Stop-Process -Id $taskEngine.Id
        $taskEngine.WaitForExit()
        $taskSession.engine_closed_after_safe_completion=$true
        $taskSession.status='COMPLETED';Save-BrakingManifest
        if($taskExit -ne 0 -or !$taskRaw.measurement_passed){throw 'Measurement gate failed; further sessions not started'}
    }
    $taskManifest.status='COMPLETED'
}catch{
    $taskManifest.status='STOPPED_FOR_INSPECTION';$taskManifest.error=$_.Exception.Message
    throw
}finally{
    $taskManifest.finished=(Get-Date).ToString('o');Save-BrakingManifest;Pop-Location
    Write-Host "Braking manifest: $taskManifestPath"
}
