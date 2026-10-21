param(
    [string]$Conditions = 'configs/trajectory_matched_pilot_v4/conditions.json'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv\python.exe'
$taskConditionsPath = (Resolve-Path -LiteralPath $Conditions).Path
$taskConditions = Get-Content -LiteralPath $taskConditionsPath -Raw | ConvertFrom-Json
& (Join-Path $PSScriptRoot 'assert-trajectory-source-freeze.ps1') -SourceHashes $taskConditions.current_flight_source_sha256
$taskFolder = Join-Path $taskRoot ('logs\trajectory_matched_sweep_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'))
New-Item -ItemType Directory -Path $taskFolder | Out-Null
$taskPath = Join-Path $taskFolder 'sweep_report.json'
$taskSweep = [ordered]@{
    schema_version=1; kind='MATCHED_START_SENSOR_LATERAL_DEVELOPMENT_PILOT'; status='RUNNING';
    formal_thesis_effect_results=$false; conditions_path=$taskConditionsPath;
    conditions_sha256=(Get-FileHash -LiteralPath $taskConditionsPath -Algorithm SHA256).Hash.ToLowerInvariant();
    planned_sessions=@($taskConditions.session_plan).Count; planned_flights=2*@($taskConditions.session_plan).Count;
    started=(Get-Date).ToString('o'); sessions=@()
}
function Save-MatchedSweep {
    $taskSweep | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $taskPath -Encoding UTF8
}
Save-MatchedSweep
Push-Location $taskRoot
try {
    foreach ($taskPlanned in $taskConditions.session_plan) {
        & (Join-Path $PSScriptRoot 'assert-trajectory-source-freeze.ps1') -SourceHashes $taskConditions.current_flight_source_sha256
        $taskConfig = $taskPlanned.configuration_path
        if (!(Test-Path -LiteralPath $taskConfig)) { $taskConfig = Join-Path (Split-Path $taskConditionsPath -Parent) $taskPlanned.configuration_path }
        $taskConfig = (Resolve-Path -LiteralPath $taskConfig).Path
        $taskFrozen = @($taskConditions.conditions | Where-Object {$_.condition_id -eq $taskPlanned.condition_id})
        if ($taskFrozen.Count -ne 1 -or (Get-FileHash -LiteralPath $taskConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskFrozen[0].configuration_sha256) {
            throw 'Frozen condition configuration hash changed before execution.'
        }
        $taskEntry = [ordered]@{
            planned_session_index=$taskPlanned.planned_session_index;
            condition_id=$taskPlanned.condition_id; repetition=$taskPlanned.repetition;
            policy_order=$taskPlanned.method_order; configuration=$taskConfig; status='STARTED';
            started=(Get-Date).ToString('o')
        }
        $taskSweep.sessions += $taskEntry
        Save-MatchedSweep
        $taskPrevious = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'trajectory_interface_pilot_*' | Select-Object -ExpandProperty FullName)
        $taskReverse = $taskPlanned.method_order[0] -eq 'perception'
        Write-Host "Matched start pilot: $($taskEntry.condition_id), repetition $($taskEntry.repetition), $($taskEntry.policy_order -join ', ')"
        try {
            & (Join-Path $PSScriptRoot 'run-trajectory-interface-pilot.ps1') -TargetSource sensor_surface_estimate -Sessions 1 -ExperimentConfig $taskConfig -ReverseFirstOrder:$taskReverse
        } finally {
            $taskNew = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'trajectory_interface_pilot_*' | Where-Object { $_.FullName -notin $taskPrevious })
            if ($taskNew.Count -eq 1) {
                $taskManifest = Join-Path $taskNew[0].FullName 'manifest.json'
                $taskEntry.manifest=$taskManifest
                if (Test-Path -LiteralPath $taskManifest) {
                    $taskEntry.manifest_sha256=(Get-FileHash -LiteralPath $taskManifest -Algorithm SHA256).Hash.ToLowerInvariant()
                }
                Save-MatchedSweep
            }
        }
        if ($taskNew.Count -ne 1) { throw 'Independent pilot manifest was not unique.' }
        $taskAnalysis = Join-Path $taskNew[0].FullName 'analysis'
        & $taskPython (Join-Path $taskRoot 'python\analyze_trajectory_pilot.py') --manifest $taskManifest --output $taskAnalysis
        if ($LASTEXITCODE -ne 0) { throw 'Pilot analysis failed.' }
        & $taskPython (Join-Path $taskRoot 'python\snapshot_trajectory_sources.py') --manifest $taskManifest
        if ($LASTEXITCODE -ne 0) { throw 'Trial source snapshot verification failed.' }
        $taskSummary = Get-Content -LiteralPath (Join-Path $taskAnalysis 'summary.json') -Raw | ConvertFrom-Json
        $taskEntry.analysis=(Join-Path $taskAnalysis 'summary.json')
        $taskEntry.analysis_sha256=(Get-FileHash -LiteralPath $taskEntry.analysis -Algorithm SHA256).Hash.ToLowerInvariant()
        $taskEntry.started_flights=$taskSummary.started_flights
        $taskEntry.task_success_count=$taskSummary.task_success_count
        $taskEntry.safe_landing_count=$taskSummary.safe_landing_and_release_count
        $taskEntry.status='COMPLETED'
        Save-MatchedSweep
        foreach ($taskRun in $taskSummary.runs) {
            if (!$taskRun.safe_landing_and_release -or $taskRun.physical_sampled_budget_status -ne 'NO_EXCEEDANCE_IN_RECORDED_SAMPLES' -or $taskRun.command_norm_violations -ne 0) {
                throw 'Safe completion or approach motion gate failed; stop further sessions.'
            }
        }
        # Acceptance and pair checks are read from the audited raw report, not task success.
        & $taskPython (Join-Path $taskRoot 'python\audit_trajectory_matched_gate.py') --summary (Join-Path $taskAnalysis 'summary.json')
        if ($LASTEXITCODE -ne 0) { throw 'Actual starting-state acceptance gate failed; stop further sessions.' }
    }
    $taskSweep.status='COMPLETED'
} catch {
    $taskSweep.status='STOPPED_FOR_INSPECTION'
    $taskSweep.error=$_.Exception.Message
    throw
} finally {
    $taskSweep.finished=(Get-Date).ToString('o')
    Save-MatchedSweep
    Pop-Location
    Write-Host "Matched sweep ledger: $taskPath"
}
