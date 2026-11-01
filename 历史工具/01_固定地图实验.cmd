@echo off
for %%I in ("%~dp0..") do set "task_project_root=%%~fI\"
setlocal
chcp 65001 >nul
cd /d "%task_project_root%"
set PYTHONUTF8=1
if not exist ".venv\pythonw.exe" (
    echo 缺少依赖：.venv\pythonw.exe
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if not exist "python\fixed_map_dashboard_v24.py" (
    echo 缺少依赖：python\fixed_map_dashboard_v24.py
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if not exist "configs\fixed_map_perception_v1.json" (
    echo 缺少依赖：configs\fixed_map_perception_v1.json
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if not exist "configs\fixed_map_perception_yolo_gpu640_v24.json" (
    echo 缺少感知配置：configs\fixed_map_perception_yolo_gpu640_v24.json
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if /I "%~1"=="--check" (
    echo 入口检查通过：01_固定地图实验
    exit /b 0
)
set PYTHONDONTWRITEBYTECODE=1
set FIXED_MAP_SESSION_OUTPUT=
set TEMP=%task_project_root%tmp\interface_v24
set TMP=%TEMP%
if not exist "%TEMP%" mkdir "%TEMP%"
set YOLO_CONFIG_DIR=%task_project_root%logs\t1_balloon_model_runtime\ultralytics
set TORCH_HOME=%task_project_root%logs\t1_balloon_model_runtime\torch
set MPLCONFIGDIR=%task_project_root%logs\t1_balloon_model_runtime\matplotlib
set XDG_CACHE_HOME=%task_project_root%.runtime-cache
set UE-LocalDataCachePath=%task_project_root%.unreal-cache
start "" "%task_project_root%.venv\pythonw.exe" "%task_project_root%python\fixed_map_dashboard_v24.py" --config "%task_project_root%configs\fixed_map_perception_yolo_gpu640_v24.json"
if errorlevel 1 (
    echo 启动失败，请检查上方错误信息。
    pause
    exit /b 1
)
exit /b 0
