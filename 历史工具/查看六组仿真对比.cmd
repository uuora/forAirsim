@echo off
for %%I in ("%~dp0..") do set "task_project_root=%%~fI\"
if exist "%task_project_root%reports\route_dashboard_v2\index.html" (
    start "" "%task_project_root%reports\route_dashboard_v2\index.html"
) else (
    start "" "%task_project_root%reports"
)
