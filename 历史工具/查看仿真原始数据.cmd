@echo off
for %%I in ("%~dp0..") do set "task_project_root=%%~fI\"
start "" "%task_project_root%logs"
