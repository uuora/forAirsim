# AirSim dual-drone balloon simulation

This project uses Unreal Engine 4.27.2, Microsoft AirSim and SimpleFlight to simulate two multirotors. `DroneA` performs the first approach; `DroneB` waits for a fallback command when the first attempt misses.

## Current status

- AirSim and the Blocks environment build successfully.
- Two vehicles are configured in `configs/settings.json`.
- The active settings file is installed at `%USERPROFILE%\Documents\AirSim\settings.json`.
- The two-drone smoke test has passed: takeoff, separated movement, hover, descent and disarm.
- Runtime logs are written to `logs/` and are ignored by Git.
- Live RPC verification confirms both DroneA and DroneB; a red balloon and string can be recreated with `scripts/start-scene.ps1`.
- Both vehicles explicitly configure named IMU, barometer, GPS, magnetometer,
  forward-distance and downward-distance sensors. Missions record advisory 1 Hz
  health snapshots without using them to change flight commands yet.

## Start the scene

From the project root in PowerShell:

```powershell
& .\scripts\start-scene.ps1
```

This connects to an existing simulation or launches Blocks, verifies both vehicles,
and creates the balloon configured in `configs/balloon.json`. Each run produces
`logs/scene_<timestamp>/scene_report.json`, `scene.log`, and `overview.png`.
Balloon objects are runtime objects; rerun after restarting the level. See
`docs/development-log.md` for validation results and remaining work.

## Project layout

- `python/`: Python control and coordination code
- `configs/`: versioned configuration templates
- `scripts/`: PowerShell and native build helpers
- `logs/`: local runtime logs, not committed
- `docs/`: installation and environment records
- `external/AirSim/`: upstream AirSim checkout and build tree

## Staging and approach demo

With the scene running, execute `& .\scripts\run-approach-demo.ps1`.
Both drones stage in the air; A approaches to about 2 metres from the balloon
centre while B holds. Press M in Blocks for the fixed overview. The demo ends
with simulation physics paused for inspection, not landed; rerun to replay.
Flight reports, telemetry and screenshots are saved in `logs/approach_<timestamp>/`.

## Full balloon demonstration

Double-click `Run-Balloon-Demo.cmd`: choose 1 for A contact, or 2 for an intentional
A miss followed by B contact. The demonstration restores the balloon, verifies
fresh target collision evidence, removes the contacted balloon, returns both
drones to their pads and verifies ground contact before disarming. The scene
then pauses. See `docs/demo-guide.md` for Chinese instructions and limitations.

PowerShell: `& .\scripts\run-balloon-mission.ps1 -Scenario both`.
Logs and screenshots are in `logs/mission_<scenario>_<timestamp>/`;
`logs/experiment_summary.csv` includes passed and failed completed runs.

Choose **3** for the image-only balloon detection preview: both vehicles move to
observation positions, physics pauses, and a two-camera presence/absence comparison
opens. This mode leaves the drones airborne and paused. It does not use vision
to steer the drones. Results are saved under `logs/vision_<timestamp>/`.
Mission reports also include periodic annotated camera evidence while flying; image
detection is advisory and collision feedback decides a hit. A camera that is facing
away from the target can legitimately report no candidate.

Options 1 and 2 also open a live two-camera monitor for the entire mission. Closing
the monitor with Q/Esc does not stop the flight.
The monitor overlays the perception result: a green box/yellow center marks the
detected red balloon, and each panel displays `DETECTED` or `NO TARGET`.

The basic framework is now split into `configs/mission.json` (geometry and safety),
`python/mission_config.py` (validation), `python/target_provider.py` (ground-truth
and camera sensing ports), and `python/coordinator.py` (A-first/B-fallback
arbitration). The working mission runner remains the baseline while these modules
are integrated incrementally.

Choose **4** to open the live two-camera monitor (DroneA on the left, DroneB on the
right). Press Q or Esc in that window to close it.
