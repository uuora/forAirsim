@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "scripts\enclosed-lab-manual.ps1" (
    echo 缺少依赖：scripts\enclosed-lab-manual.ps1
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if not exist "configs\enclosed_uav_lab_manual.json" (
    echo 缺少依赖：configs\enclosed_uav_lab_manual.json
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if /I "%~1"=="--check" (
    echo 入口检查通过：工具_按参数重建地图
    exit /b 0
)
echo 此操作会按JSON替换工作地图部件；请先保存或导出编辑结果。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enclosed-lab-manual.ps1" -Action BuildParameters
if errorlevel 1 (
    echo 启动失败，请检查上方错误信息。
    pause
    exit /b 1
)
pause
exit /b 0
