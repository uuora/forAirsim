# AirSim installation record

Started: 2026-09-08 (Windows 11, D:\forAirsim).

## Existing environment

- NVIDIA GeForce RTX 3070 Laptop GPU; approximately 16 GB physical RAM.
- Git 2.55.0.windows.2 at D:\Git.
- Visual Studio Community 2022 17.13.2 at D:\visual, C++ desktop workload present.
- MSVC 14.38 and 14.43; bundled CMake 3.30.5.
- Windows SDK 22000, 22621, 26100 and .NET Framework targeting packs present.
- UE 5.2.1 and UE 5.7.4 already installed; retained.

## Installed and verified

- Microsoft AirSim source: external\AirSim.
- Commit: d109f0dc25d4ae425eca7b279da33df5f5fcb42e.
- Isolated Conda Python 3.10 environment: D:\forAirsim\.venv.
- AirSim PythonClient installed editable from the same checkout.
- NumPy 1.26.4, OpenCV contrib 4.10.0.84, msgpack-rpc-python 0.4.1.
- Python imports, RPC client construction and pip dependency checks passed.
- AirSim native Release solution built using MSVC 14.38, including AirLib and rpclib.
- Plugin source and native dependencies copied into Blocks. UE editor plugin DLL still needs the UE project build.
- Reproducible native build script: scripts\build-airlib.cmd. It avoids upstream cleanup and uses four parallel jobs.
- Python dependency snapshot: docs\python-requirements-lock.txt.

## UE installation and remaining validation

- Target: Unreal Engine 4.27.2, D:\UE_4.27.
- Selected core, starter content, templates and engine source, approximately 20.1 GB installed.
- Android, iOS, Linux, HoloLens and editor debug symbols excluded.
- Installation started through Epic Games Launcher; completion and Blocks execution pending.
- Requested optional MSVC v142 and Windows SDK 19041 using Visual Studio Installer. First non-elevated attempt returned 5007; elevated attempt requires Windows authorization.

## Repaired stale Epic entry

The old UE_4.27 registration pointed to D:\Epic Games\UE_4.27, which did not exist.
The stale installation manifest and Epic shared installation record were backed up in docs\installation-backup before removal from their active registration directories.
The stale UE4 launcher slot was removed from GameUserSettings.ini; a sibling .airsim-backup file retains the pre-edit settings.
Existing UE5 registrations were preserved.

Run scripts\check-environment.ps1 to refresh docs\environment-status.json.
An installed editor file alone does not prove that the simulator launches or that flight APIs work; those checks remain necessary.

## First runtime verification

- Dual-vehicle settings template: configs\settings.json.
- Active AirSim settings: %USERPROFILE%\Documents\AirSim\settings.json.
- Blocks standalone game launched successfully and listened on RPC port 41451.
- AirSim reported both `DroneA` and `DroneB`.
- `python\run_dual_smoke.py` completed takeoff, separated movement, hover, controlled descent and disarm for both vehicles.
- Runtime result: `DUAL_DRONE_SMOKE_TEST_PASS`.
