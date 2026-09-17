@echo off
set PYTHONUTF8=1
"%~dp0.venv\python.exe" "%~dp0python\run_yolo_baseline.py"
if errorlevel 1 goto failed
echo YOLO baseline complete. See logs\yolo_baseline_latest.json.
pause
exit /b 0
:failed
echo YOLO baseline failed. Inspect the terminal output.
pause
exit /b 1
