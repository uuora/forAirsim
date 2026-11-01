@echo off
setlocal
set "task_root=%~dp0"
cd /d "%task_root%"
set PYTHONUTF8=1
if /I "%~1"=="--check" (
    "%task_root%.venv\python.exe" -c "import sys;sys.path.insert(0,'python');from prepared_route_panel import prepared_config;from dual_target_control.prepared_route_live import PreparedRouteRuntime;prepared_config();print('Protocol15 control experiment entry: READY')"
    if errorlevel 1 exit /b 1
    exit /b 0
)
if exist "%task_root%.venv\pythonw.exe" (
    start "" /D "%task_root%" "%task_root%.venv\pythonw.exe" "%task_root%python\run_route_experiment_panel.py"
) else (
    start "" /D "%task_root%" "%task_root%.venv\python.exe" "%task_root%python\run_route_experiment_panel.py"
)
