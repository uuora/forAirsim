param(
    [string]$Config,
    [string]$Plans,
    [string]$OutputDir,
    [string[]]$Groups,
    [switch]$AllGroups,
    [switch]$ShowWindow,
    [switch]$ProbeOnly
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$taskRoot = Split-Path $PSScriptRoot -Parent
if ($PSVersionTable.PSVersion.Major -le 5) {
    $taskBuiltinModules = Join-Path $PSHOME 'Modules'
    $env:PSModulePath = $taskBuiltinModules + [IO.Path]::PathSeparator + $env:PSModulePath
}
if (!$Config) { $Config = Join-Path $taskRoot 'configs\curve_abom_airsim_control_targets_v12.json' }
$Config = (Resolve-Path -LiteralPath $Config).Path
$taskConfig = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($taskConfig.port -ne 41461 -or $taskConfig.vehicle -ne 'ResearchDrone') { throw 'Research endpoint mismatch.' }
if (!$ProbeOnly) {
    if (!$Plans) { $Plans = Join-Path $taskRoot 'reports\control_targets_fixed_matched_20261030_v3\airsim_replay_plan.json' }
    $Plans = (Resolve-Path -LiteralPath $Plans).Path
    $taskPlansDocument = Get-Content -LiteralPath $Plans -Raw | ConvertFrom-Json
    $taskCatalogue = @($taskPlansDocument.plans)
    if ($taskPlansDocument.schema_version -ne 1 -or $taskCatalogue.Count -ne 6) { throw 'Expected the complete frozen six-plan catalogue.' }
    $taskIds = @($taskCatalogue | ForEach-Object { [string]$_.group_id })
    if ($AllGroups -and $PSBoundParameters.ContainsKey('Groups')) { throw 'Choose -Groups or -AllGroups, not both.' }
    if ($AllGroups) { $Groups = $taskIds }
    elseif (!$PSBoundParameters.ContainsKey('Groups')) {
        Write-Host 'Choose the method(s) to run. No method starts until you select it.'
        for ($taskChoice = 0; $taskChoice -lt $taskIds.Count; $taskChoice++) { Write-Host ("{0}. {1}" -f ($taskChoice+1), $taskIds[$taskChoice]) }
        $taskAnswer = Read-Host 'Method number(s), for example 4 or 4,6'
        $taskNumbers = @($taskAnswer.Split(',') | ForEach-Object { $_.Trim() })
        $Groups = @($taskNumbers | ForEach-Object {
            $taskNumber = 0
            if (![int]::TryParse($_, [ref]$taskNumber) -or $taskNumber -lt 1 -or $taskNumber -gt $taskIds.Count) { throw 'Select valid method numbers 1 to 6.' }
            $taskIds[$taskNumber-1]
        })
    }
    if (@($Groups).Count -eq 0) { throw 'Choose at least one method.' }
    if (@($Groups | Select-Object -Unique).Count -ne $Groups.Count) { throw 'Duplicate selected methods.' }
    foreach ($taskGroup in $Groups) {
        if (!$taskGroup -or $taskIds -cnotcontains $taskGroup) { throw "Unknown selected method: $taskGroup" }
    }
}
elseif ($AllGroups -or $PSBoundParameters.ContainsKey('Groups')) { throw 'The ground probe does not run selected methods.' }
if (!$OutputDir) { $OutputDir = Join-Path $taskRoot ('reports\airsim_curve_replay_pilot_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff')) }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $OutputDir) { throw 'Refusing to overwrite an existing pilot directory.' }
$taskEditor = [IO.Path]::GetFullPath($taskConfig.editor)
$taskProject = Join-Path $taskRoot $taskConfig.project
$taskPython = Join-Path $taskRoot '.venv\python.exe'
$taskRunner = Join-Path $taskRoot 'python\run_airsim_curve_replay_pilot.py'
$taskMapFile = Join-Path (Split-Path $taskProject -Parent) ('Content\' + $taskConfig.map.Substring(6).Replace('/', '\') + '.umap')
foreach ($taskPath in @($taskEditor, $taskProject, $taskPython, $taskRunner, $taskMapFile)) {
    if (!(Test-Path -LiteralPath $taskPath)) { throw "Missing existing runtime/map: $taskPath. This launcher does not compile or build maps." }
}
function Test-ResearchPort {
    $taskSocket = [Net.Sockets.TcpClient]::new()
    try { return ($taskSocket.ConnectAsync([string]$taskConfig.host, [int]$taskConfig.port).Wait(300) -and $taskSocket.Connected) }
    catch { return $false } finally { $taskSocket.Dispose() }
}
if (Test-ResearchPort) { throw 'Research port 41461 is occupied. Preserve the existing instance.' }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$taskRuntime = Join-Path $OutputDir 'runtime'
New-Item -ItemType Directory -Path $taskRuntime | Out-Null
$taskSettings = Join-Path $taskRuntime 'settings.json'
if ($ShowWindow) { $taskConfig.settings.ViewMode = 'FlyWithMe' }
$taskConfig.settings | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $taskSettings -Encoding UTF8
$taskEffectiveConfig = Join-Path $taskRuntime 'experiment_config.json'
$taskConfig | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $taskEffectiveConfig -Encoding UTF8
$taskManifestPath = Join-Path $OutputDir 'launcher_manifest.json'
$taskManifest = [ordered]@{schema_version=1; status='STARTING'; config=$Config; plans=$Plans; probe_only=[bool]$ProbeOnly;
    effective_config=$taskEffectiveConfig; selected_group_ids=@(); show_window=[bool]$ShowWindow;
    started_at_utc=(Get-Date).ToUniversalTime().ToString('o'); port=[int]$taskConfig.port;
    map=[string]$taskConfig.map; settings=$taskSettings; settings_sha256=(Get-FileHash -LiteralPath $taskSettings -Algorithm SHA256).Hash.ToLowerInvariant();
    log=(Join-Path $OutputDir 'unreal.log'); owned_process_closed=$false;
    scope='Owned isolated local simulator. Frozen-plan replay only; no ABOM search against AirSim.'}
if (!$ProbeOnly) { $taskManifest.selected_group_ids = @($Groups) }
$taskBinaryPaths = @($taskEditor, $taskProject, $taskMapFile,
    (Join-Path (Split-Path $taskProject -Parent) 'Binaries\Win64\UE4Editor-Blocks.dll'),
    (Join-Path (Split-Path $taskProject -Parent) 'Plugins\AirSim\Binaries\Win64\UE4Editor-AirSim.dll'))
$taskManifest.runtime_files = @($taskBinaryPaths | ForEach-Object {
    if (!(Test-Path -LiteralPath $_)) { throw "Required existing runtime file is absent: $_" }
    @{path=$_; bytes=(Get-Item -LiteralPath $_).Length; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
})
function Save-LauncherManifest { $taskManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $taskManifestPath -Encoding UTF8 }
Save-LauncherManifest
$taskEngine = $null
try {
    $taskArguments = @(('"'+$taskProject+'"'), [string]$taskConfig.map, '-game', '-windowed', '-ResX=1000', '-ResY=700',
        '-nosplash', '-NoSound', '-NoVSync', '-unattended', ('-settings="'+$taskSettings+'"'),
        ('-abslog="'+$taskManifest.log+'"'), ('-UserDir="'+$taskRuntime+'"'),
        '-ExecCmds="t.MaxFPS 120,t.IdleWhenNotForeground 0,Slate.bAllowThrottling 0"')
    $taskWindowStyle = if ($ShowWindow) { 'Normal' } else { 'Hidden' }
    $taskEngine = Start-Process -FilePath $taskEditor -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle $taskWindowStyle -PassThru
    $taskManifest.owned_pid = $taskEngine.Id
    $taskManifest.owned_start_utc = $taskEngine.StartTime.ToUniversalTime().ToString('o')
    Save-LauncherManifest
    Write-Host "Owned research simulator PID $($taskEngine.Id), waiting up to 120 seconds for port 41461."
    $taskDeadline = (Get-Date).AddSeconds(120)
    while (!(Test-ResearchPort)) {
        $taskEngine.Refresh()
        if ($taskEngine.HasExited) { throw "Owned simulator exited before RPC readiness. Inspect $($taskManifest.log)." }
        if ((Get-Date) -gt $taskDeadline) { throw 'Owned simulator did not open RPC within 120 seconds.' }
        Start-Sleep -Seconds 2
    }
    $taskManifest.status = 'GROUND_PROBE'
    Save-LauncherManifest
    & $taskPython $taskRunner --config $taskEffectiveConfig --output-dir (Join-Path $OutputDir 'probe') --probe-only
    if ($LASTEXITCODE -ne 0) { throw 'Ground/paused probe failed; no replay flight started.' }
    if (!$ProbeOnly) {
        $taskManifest.status = 'REPLAYING'
        Save-LauncherManifest
        $taskReplayArguments = @($taskRunner, '--config', $taskEffectiveConfig, '--plans', $Plans,
            '--output-dir', (Join-Path $OutputDir 'replay'), '--groups') + @($Groups)
        & $taskPython @taskReplayArguments
        if ($LASTEXITCODE -ne 0) { throw 'Replay pilot failed; logs and partial traces are preserved.' }
    }
    $taskManifest.status = 'COMPLETED'
}
catch {
    $taskManifest.status = 'FAIL'
    $taskManifest.failure_reason = $_.Exception.Message
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
}
finally {
    if ($taskEngine) {
        $taskEngine.Refresh()
        if (!$taskEngine.HasExited) {
            $taskCurrent = Get-Process -Id $taskEngine.Id -ErrorAction SilentlyContinue
            if ($taskCurrent -and $taskCurrent.Path -eq $taskEditor -and
                $taskCurrent.StartTime.ToUniversalTime().ToString('o') -eq $taskManifest.owned_start_utc) {
                Stop-Process -Id $taskEngine.Id
                $taskEngine.WaitForExit(10000) | Out-Null
                $taskManifest.owned_process_closed = $taskEngine.HasExited
            }
            else { $taskManifest.close_refused_identity_mismatch = $true }
        }
        else { $taskManifest.owned_process_closed = $true }
    }
    $taskManifest.finished_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    Save-LauncherManifest
    Write-Host "Pilot artifacts: $OutputDir"
}
if ($taskManifest.status -eq 'FAIL') { exit 1 }
