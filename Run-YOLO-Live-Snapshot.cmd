@echo off
set PYTHONUTF8=1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1" -ShowWindow
if errorlevel 1 goto failed
"%~dp0.venv\python.exe" "%~dp0python\capture_yolo_live_snapshot.py"
if errorlevel 1 goto failed
echo Live YOLO snapshot complete. See logs\yolo_live_latest.json.
pause
exit /b 0
:failed
echo Live YOLO snapshot failed. Inspect the terminal output.
pause
exit /b 1
