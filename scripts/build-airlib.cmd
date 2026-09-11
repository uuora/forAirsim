@echo off
setlocal
call "D:\visual\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 -vcvars_ver=14.38
if errorlevel 1 exit /b 1
set "PATH=D:\visual\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
cd /d D:\forAirsim\external\AirSim
cmake -S external\rpclib\rpclib-2.3.0 -B external\rpclib\rpclib-2.3.0\build -G "Visual Studio 17 2022" -A x64 -T version=14.38.33130
if errorlevel 1 exit /b 1
cmake --build external\rpclib\rpclib-2.3.0\build --config Release --parallel 4
if errorlevel 1 exit /b 1
robocopy external\rpclib\rpclib-2.3.0\include AirLib\deps\rpclib\include /E /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1
robocopy external\rpclib\rpclib-2.3.0\build\Release AirLib\deps\rpclib\lib\x64\Release /E /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1
msbuild AirSim.sln /m:4 /p:Platform=x64 /p:Configuration=Release /p:VCToolsVersion=14.38.33130 /verbosity:minimal
if errorlevel 1 exit /b 1
robocopy MavLinkCom\include AirLib\deps\MavLinkCom\include /E /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1
robocopy MavLinkCom\lib AirLib\deps\MavLinkCom\lib /E /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1
robocopy AirLib Unreal\Plugins\AirSim\Source\AirLib /E /XD temp /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1
copy /y AirSim.props Unreal\Plugins\AirSim\Source\AirLib\AirSim.props
robocopy Unreal\Plugins\AirSim Unreal\Environments\Blocks\Plugins\AirSim /E /XD temp /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1
exit /b 0
