# One-off launcher inventory/backup/migration. Never execute experiment launchers.
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$taskArchive = Join-Path $taskRoot ('reports/launcher_cleanup_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$taskOriginals = Join-Path $taskArchive 'originals'
New-Item -ItemType Directory -Path $taskOriginals -Force | Out-Null
$taskMapping = [ordered]@{
    'Run-Fixed-Map-Mission.cmd'='01_固定地图实验.cmd'
    'Run-Control-Workbench.cmd'='02_识别与回放工作台.cmd'
    'Edit-Enclosed-UAV-Lab.cmd'='工具_编辑实验地图.cmd'
    'Check-Enclosed-UAV-Lab-Edits.cmd'='工具_检查已保存地图.cmd'
    'Edit-Enclosed-UAV-Lab-Parameters.cmd'='工具_编辑地图参数.cmd'
    'Build-Enclosed-UAV-Lab-Parameters.cmd'='工具_按参数重建地图.cmd'
}
$taskRetired = @('Capture-Dual-Cameras.cmd','Capture-Maritime-Experiments.cmd','Check-Maritime-Sensors.cmd','Collect-Air-Multiview.cmd','Run-Auto-Pipeline.cmd','Run-Balloon-Demo.cmd','Run-Enclosed-UAV-Lab-Preview.cmd','Run-Maritime-Scene.cmd','Run-Phase1-Lateral-Pilot.cmd','Run-Phase1-Pilot.cmd','Run-Research-Console.cmd','Run-Visual-Contact.cmd','Run-Visual-Fallback.cmd','Run-Visual-Range.cmd','Run-VLX-Demo.cmd','Run-VLX-GUI.cmd','Run-YOLO-Baseline.cmd','Run-YOLO-Live-Snapshot.cmd','Validate-Air-Observation.cmd')
$taskExpected = @($taskMapping.Keys) + $taskRetired
$taskFiles = @(Get-ChildItem -LiteralPath $taskRoot -File -Filter '*.cmd')
if (@(Compare-Object ($taskExpected | Sort-Object) ($taskFiles.Name | Sort-Object)).Count) { throw 'Launcher inventory changed; inspect before running migration.' }
foreach ($taskName in $taskMapping.Values) {
    if (Test-Path -LiteralPath (Join-Path $taskRoot $taskName)) { throw "Destination already exists: $taskName" }
}
$taskRows = foreach ($taskFile in $taskFiles) {
    $taskBackup = Join-Path $taskOriginals ($taskFile.Name + '.txt')
    Copy-Item -LiteralPath $taskFile.FullName -Destination $taskBackup
    $taskDigest = (Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskDigest -ne (Get-FileHash -LiteralPath $taskBackup -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Backup digest mismatch.' }
    [ordered]@{original=$taskFile.FullName;backup=$taskBackup;sha256=$taskDigest;action=$(if ($taskMapping.Contains($taskFile.Name)) {'rename_and_normalize'} else {'remove_root_entry'});replacement=$(if ($taskMapping.Contains($taskFile.Name)) {$taskMapping[$taskFile.Name]} elseif ($taskFile.Name -eq 'Run-VLX-GUI.cmd') {'02_识别与回放工作台.cmd'} else {$null});reason=$(if ($taskFile.Name -eq 'Run-VLX-GUI.cmd') {'duplicate_same_vlx_gui_qt'} elseif ($taskRetired -contains $taskFile.Name) {'outside_current_fixed_map_experiment_or_superseded'} else {'current_experiment_or_map_maintenance'})}
}
$taskUtf8 = [System.Text.UTF8Encoding]::new($false)
function Write-TaskLauncher([string]$taskName,[string[]]$taskDependencies,[string[]]$taskCommands,[string]$taskLabel) {
    $taskLines = @('@echo off','setlocal','chcp 65001 >nul','cd /d "%~dp0"','set PYTHONUTF8=1')
    foreach ($taskDependency in $taskDependencies) {
        $taskLines += @(('if not exist "'+$taskDependency+'" ('),('    echo 缺少依赖：'+$taskDependency),'    if /I not "%~1"=="--check" pause','    exit /b 1',')')
    }
    $taskLines += @('if /I "%~1"=="--check" (',('    echo 入口检查通过：'+$taskLabel),'    exit /b 0',')')
    $taskLines += $taskCommands
    $taskLines += @('if errorlevel 1 (','    echo 启动失败，请检查上方错误信息。','    pause','    exit /b 1',')','exit /b 0')
    [System.IO.File]::WriteAllText((Join-Path $taskRoot $taskName),($taskLines -join "`r`n")+"`r`n",$taskUtf8)
}
Write-TaskLauncher '01_固定地图实验.cmd' @('.venv\pythonw.exe','python\fixed_map_dashboard.py','configs\fixed_map_perception_v1.json') @('start "" "%~dp0.venv\pythonw.exe" "%~dp0python\fixed_map_dashboard.py"') '固定地图实验与感知预览'
Write-TaskLauncher '02_识别与回放工作台.cmd' @('.venv\python.exe','python\vlx_gui_qt.py') @('"%~dp0.venv\python.exe" "%~dp0python\vlx_gui_qt.py"') '识别与回放工作台'
Write-TaskLauncher '工具_编辑实验地图.cmd' @('scripts\enclosed-lab-manual.ps1') @('echo 地图维护工具：编辑后需要重新核对冻结版本。','powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enclosed-lab-manual.ps1" -Action Open') '打开实验地图编辑器'
Write-TaskLauncher '工具_检查已保存地图.cmd' @('scripts\enclosed-lab-manual.ps1') @('powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enclosed-lab-manual.ps1" -Action Check','if errorlevel 1 exit /b 1','pause') '检查已保存地图'
Write-TaskLauncher '工具_编辑地图参数.cmd' @('scripts\enclosed-lab-manual.ps1','configs\enclosed_uav_lab_manual.json') @('powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enclosed-lab-manual.ps1" -Action OpenParameters') '编辑地图参数'
Write-TaskLauncher '工具_按参数重建地图.cmd' @('scripts\enclosed-lab-manual.ps1','configs\enclosed_uav_lab_manual.json') @('echo 此操作会按JSON替换工作地图部件；请先保存或导出编辑结果。','powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enclosed-lab-manual.ps1" -Action BuildParameters','if errorlevel 1 exit /b 1','pause') '按参数重建地图'
# Delete only explicit, fully resolved root files after verified backups. No recursion.
foreach ($taskName in $taskExpected) {
    $taskTarget = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $taskName))
    if ((Split-Path $taskTarget -Parent) -ne $taskRoot -or [System.IO.Path]::GetExtension($taskTarget) -ne '.cmd') { throw 'Deletion escaped workspace root.' }
    Remove-Item -LiteralPath $taskTarget
}
$taskNew = @(Get-ChildItem -LiteralPath $taskRoot -File -Filter '*.cmd' | ForEach-Object { [ordered]@{path=$_.FullName;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })
$taskManifest = [ordered]@{created_local=(Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz');scope='root launchers only; fixed-map DroneA perception experiment';original_count=$taskFiles.Count;current_count=$taskNew.Count;renamed=6;retired=19;originals=$taskRows;current=$taskNew;experiment_launchers_executed=$false;map_or_model_modified=$false;historical_reports_modified=$false}
[System.IO.File]::WriteAllText((Join-Path $taskArchive 'manifest.json'),($taskManifest | ConvertTo-Json -Depth 7),$taskUtf8)
Write-Output $taskArchive
