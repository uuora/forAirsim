@echo off
cd /d "%~dp0"
"%~dp0.venv\python.exe" "%~dp0python\experiment_console.py"
if errorlevel 1 pause
