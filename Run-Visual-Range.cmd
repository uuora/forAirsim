@echo off
cd /d "%~dp0"
"%~dp0.venv\python.exe" "%~dp0python\launch_demo.py" --scenario visual_range
if errorlevel 1 echo Test failed. Read the error and inspect the simulation before rerunning.
pause
