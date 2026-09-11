@echo off
if /I "%~1"=="--check" goto check
echo AirSim balloon simulation
echo 1. DroneA contacts the balloon
echo 2. DroneA misses, DroneB takes over
echo    Options 1 and 2 now keep a live two-camera window open during flight.
echo 3. Camera balloon detection preview (ends paused in the air)
echo 4. Open live DroneA/DroneB camera monitor
echo 5. Safe reset after a completed/landed run
choice /c 12345 /n /m "Choose 1, 2, 3, 4 or 5: "
if errorlevel 5 goto reset
if errorlevel 4 goto cameras
if errorlevel 3 goto vision
if errorlevel 2 goto fallback
"%~dp0.venv\python.exe" "%~dp0python\launch_demo.py" --scenario a_hit
goto done
:fallback
"%~dp0.venv\python.exe" "%~dp0python\launch_demo.py" --scenario fallback
goto done
:vision
"%~dp0.venv\python.exe" "%~dp0python\launch_demo.py" --scenario vision
goto done
:cameras
"%~dp0.venv\python.exe" "%~dp0python\setup_scene.py"
if errorlevel 1 echo Could not restore the balloon. Make sure Blocks is running. & goto done
"%~dp0.venv\python.exe" "%~dp0python\live_camera_view.py"
:reset
"%~dp0.venv\python.exe" "%~dp0python\reset_mission.py"
:done
if errorlevel 1 echo Demo stopped with an error. See the message above and the logs folder.
pause
exit /b
:check
"%~dp0.venv\python.exe" "%~dp0python\launch_demo.py" --check
exit /b %errorlevel%
