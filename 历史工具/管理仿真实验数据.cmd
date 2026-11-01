@echo off
for %%I in ("%~dp0..") do set "task_project_root=%%~fI\"
set "task_root=%task_project_root%"
if exist "%task_root%.venv\pythonw.exe" (
    start "" /D "%task_root%" "%task_root%.venv\pythonw.exe" "%task_root%python\run_route_experiment_panel.py" --tab data
) else (
    start "" /D "%task_root%" "%task_root%.venv\python.exe" "%task_root%python\run_route_experiment_panel.py" --tab data
)
