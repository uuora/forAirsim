@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\python.exe" (
  echo 缺少 Python 环境：.venv\python.exe
  pause
  exit /b 1
)
if /I "%~1"=="--check" (
  "%~dp0.venv\python.exe" -c "import sys;sys.path.insert(0,'python');from dual_target_control.core import ControlConfig;from dual_target_control.runtime import ControlRuntime;from dual_target_control.ui import main;print('三维双目标控制入口检查通过')"
  if errorlevel 1 exit /b 1
  exit /b 0
)
if /I "%~1"=="--demo" (
  start "" "%~dp0.venv\pythonw.exe" "%~dp0python\run_dual_target_control.py" --mode demo --config "%~dp0configs\dual_target_control.json"
) else (
  start "" "%~dp0.venv\pythonw.exe" "%~dp0python\run_dual_target_control.py" --mode airsim --config "%~dp0configs\dual_target_control.json"
)
exit /b 0
