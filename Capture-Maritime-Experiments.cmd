@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1" -ShowWindow
if errorlevel 1 goto failed
"%~dp0.venv\python.exe" "%~dp0python\run_maritime_experiments.py" --pairs 10
if errorlevel 1 goto failed
echo Four profiles captured. Ground truth is separate and requires review.
pause
exit /b 0
:failed
echo Failed. Inspect the error and manifest before continuing.
pause
exit /b 1
