@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\python.exe" (
    echo 缺少依赖：.venv\python.exe
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if not exist "python\vlx_gui_qt.py" (
    echo 缺少依赖：python\vlx_gui_qt.py
    if /I not "%~1"=="--check" pause
    exit /b 1
)
if /I "%~1"=="--check" (
    echo 入口检查通过：02_识别与回放工作台
    exit /b 0
)
"%~dp0.venv\python.exe" "%~dp0python\vlx_gui_qt.py"
if errorlevel 1 (
    echo 启动失败，请检查上方错误信息。
    pause
    exit /b 1
)
exit /b 0
