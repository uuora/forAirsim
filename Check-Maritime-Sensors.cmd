@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-maritime-scene.ps1"
if errorlevel 1 goto failed
"%~dp0.venv\python.exe" "%~dp0python\check_onboard_sensors.py" --settings "%~dp0configs\maritime_harbor_settings.json" --cameras
if errorlevel 1 goto failed
echo Sensor check complete. PASS_WITH_LIMITATIONS means some ranges are unusable.
echo See the sensor_report.json path printed above.
pause
exit /b 0
:failed
echo Sensor check failed. Review the report or error above.
pause
exit /b 1
