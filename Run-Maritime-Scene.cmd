@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1" -ShowWindow
if errorlevel 1 pause
