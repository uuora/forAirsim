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
if /I "%~1"=="--check" (
    echo 入口检查通过：工具_编辑实验地图
    exit /b 0
)
echo 地图维护工具：编辑后需要重新核对冻结版本。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enclosed-lab-manual.ps1" -Action Open
if errorlevel 1 (
    echo 启动失败，请检查上方错误信息。
    pause
    exit /b 1
)
exit /b 0
