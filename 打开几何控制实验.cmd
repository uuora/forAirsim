@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if /I "%~1"=="--check" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open-geometric-control-panel.ps1" -Check
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open-geometric-control-panel.ps1" %*
)
if errorlevel 1 (
  echo 启动失败，请检查上方说明。
  pause
  exit /b 1
)
