@echo off
cd /d "%~dp0"
"%~dp0.venv\python.exe" "%~dp0python\launch_demo.py" --scenario visual_contact
if errorlevel 1 echo Test failed. Inspect the paused simulation and mission report.
pause
