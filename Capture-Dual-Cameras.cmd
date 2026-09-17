@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1" -ShowWindow
if errorlevel 1 goto failed
"%~dp0.venv\python.exe" "%~dp0python\capture_dual_cameras.py" --pairs 10
if errorlevel 1 goto failed
echo Capture complete. See datasets\dual_camera.
pause
exit /b 0
:failed
echo Capture failed. Review the error above.
pause
exit /b 1
