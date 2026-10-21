$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -le 5) {
    $env:PSModulePath = (Join-Path $PSHOME 'Modules') + [IO.Path]::PathSeparator + $env:PSModulePath
}
$env:PYTHONUTF8 = '1'
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskPython = Join-Path $taskRoot '.venv\python.exe'
$taskConditions = @('center','target_y_minus_06','target_y_plus_06')
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$taskOutput = Join-Path $taskRoot "logs\phase1_lateral_sweep_$taskStamp"
New-Item -ItemType Directory -Path $taskOutput | Out-Null
$taskManifestPath = Join-Path $taskOutput 'manifest.json'
$taskManifest = [ordered]@{status='RUNNING';started=(Get-Date).ToString('o');
    scope='Single drone, one static target; lateral position pilot, not formal evaluation';
    order=$taskConditions; sessions_per_condition=1; pairs_per_session=2; pilots=@()}
function Save-LateralManifest { $taskManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8 }
Save-LateralManifest
try {
    foreach ($condition in $taskConditions) {
        $taskConfig = Join-Path $taskRoot "configs\phase1_lateral_pilot\$condition.json"
        $taskPrevious = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'phase1_pilot_*' | Select-Object -ExpandProperty FullName)
        $taskEntry = [ordered]@{condition_id=$condition;status='RUNNING';config=$taskConfig;
            config_sha256=(Get-FileHash -LiteralPath $taskConfig -Algorithm SHA256).Hash.ToLower()}
        $taskManifest.pilots += $taskEntry
        Save-LateralManifest
        try {
            & (Join-Path $PSScriptRoot 'run-phase1-pilot.ps1') -Sessions 1 -ExperimentConfig $taskConfig
            $taskEntry.status='PASS'
        } finally {
            $taskNew = @(Get-ChildItem (Join-Path $taskRoot 'logs') -Directory -Filter 'phase1_pilot_*' | Where-Object { $_.FullName -notin $taskPrevious })
            if ($taskNew.Count -eq 1) {
                $taskEntry.manifest=Join-Path $taskNew[0].FullName 'manifest.json'
                $taskEntry.manifest_sha256=(Get-FileHash -LiteralPath $taskEntry.manifest -Algorithm SHA256).Hash.ToLower()
            }
            Save-LateralManifest
        }
    }
    $taskManifest.status='PASS'
} catch {
    $taskManifest.status='FAIL'
    $taskManifest.error=$_.Exception.Message
    if ($taskManifest.pilots.Count -gt 0 -and $taskManifest.pilots[-1].status -ne 'PASS') { $taskManifest.pilots[-1].status='STOPPED_FOR_INSPECTION' }
    throw
} finally {
    $taskManifest.finished=(Get-Date).ToString('o')
    Save-LateralManifest
    Write-Host "Lateral sweep manifest: $taskManifestPath"
}
