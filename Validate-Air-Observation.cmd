@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1" -ShowWindow
if errorlevel 1 goto failed
"%~dp0.venv\python.exe" "%~dp0python\validate_air_observation.py"
if errorlevel 1 goto failed
echo Air observation validation complete. See logs\air_observation_latest.json.
pause
exit /b 0
:failed
echo Air observation validation failed. Inspect logs\air_observation_latest.json and report.json.
pause
exit /b 1
