@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if not exist ".venv\python.exe" goto missing
if not "%~1"=="" goto console
start "" ".venv\pythonw.exe" "python\fast_panel.py"
exit /b %errorlevel%
:console
".venv\python.exe" "python\fast_panel.py" %*
exit /b %errorlevel%
:missing
echo Cannot find the shared forAirsim Python environment at .venv\python.exe.
pause
exit /b 1
