@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1" -ShowWindow
if errorlevel 1 goto failed
"%~dp0.venv\python.exe" "%~dp0python\collect_air_multiview.py"
if errorlevel 1 goto failed
echo Air multiview collection complete. See logs\air_multiview_latest.json.
pause
exit /b 0
:failed
echo Air multiview collection failed. Inspect logs\air_multiview_latest.json and report.json.
pause
exit /b 1
