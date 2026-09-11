# Python control layer

All Python control, coordination, perception and evaluation code belongs in this directory.

## Run the current smoke test

```powershell
& "D:\forAirsim\.venv\python.exe" "D:\forAirsim\python\run_dual_smoke.py"
```

Each run prints progress and writes a timestamped log file under `logs/`.

## Verify two drones and add the balloon

From the project root, run `& .\scripts\start-scene.ps1`. This starts Blocks when needed and runs `python/setup_scene.py`.
With AirSim already running: `& .\.venv\python.exe .\python\setup_scene.py`.

The script verifies `DroneA` and `DroneB`, adds or updates `BalloonTarget` and its string,
applies a red texture, and checks pose, scale, segmentation ID and blocking geometry in
the target region. It saves a JSON report, log, front camera images and `overview.png`
under `logs/scene_<timestamp>/`. The overview temporarily moves a camera and restores
its prior pose in a `finally` block; neither drone is commanded to fly.

Edit `configs/balloon.json` for the target position (world NED metres; negative Z is up).
Objects are created at runtime, so run the script again after restarting the level.
The balloon is a stationary mesh with segmentation ID 42. This setup script does
not fly vehicles; the full mission script below implements contact and fallback.

## Steps 1 and 2: staging and approach

Run `scripts/run-approach-demo.ps1` with the configured Blocks scene open. Press M
in the simulation window to use the external overview camera. The settings template
now selects Manual view for future settings installations/restarts.

`run_approach_demo.py` lifts both drones to world Z=-3 m, stages A at (-5,-4,-3)
and B at (-5,4,-3), then moves A to (-2,0,-3) while B waits. It monitors new
collisions, separation (at least 2 m), endpoint error (under 0.3 m), speed (under
0.25 m/s), and B's standby drift (under 0.5 m). It observes the final hold for
10 seconds and PAUSES the simulation for inspection, leaving motors armed.
This is not a landing; the next run resumes physics and repeats the demonstration.
The route is only for this inspected Blocks area and target configuration; it
does not implement general obstacle avoidance, balloon contact or fallback.

Logs: `logs/approach_<timestamp>/mission.log`, `telemetry.jsonl`, `report.json`,
`01_staging.png`, `02_approach.png`. A failure is recorded and pauses physics if
RPC remains reachable; it never disarms an airborne vehicle as cleanup.

## Modules and next steps

- `run_dual_smoke.py`: environment and two-vehicle connectivity test
- `mission_state.py`: mission state machine for first attempt and fallback
- `target_provider.py`: ground-truth target first, vision target later
- `hit_judge.py`: balloon contact and timeout judgement
- `coordinator.py`: DroneA/DroneB task arbitration

## Full contact and fallback experiment

Implemented in `run_balloon_mission.py`, using `hit_judge.py` and `mission_state.py`.
Run through `scripts/run-balloon-mission.ps1 -Scenario a_hit|fallback|both` to
restore the balloon first. The fallback test deliberately offsets A's path;
only a fresh collision against the exact target body confirms contact.
`summarize_experiments.py` exports all completed runs to a CSV, including failures.
See `docs/demo-guide.md` for the landing evidence requirements and coordinate frames.

Before starting a new round after a completed run, use `& .venv/python.exe python/reset_mission.py`.
It checks that both vehicles are landed before releasing API control, recreates the balloon through
`setup_scene.py`, and writes `logs/reset_<timestamp>/reset_report.json`. It refuses to reset an
airborne vehicle so a failed flight remains available for diagnosis.

## RGB balloon detection

`balloon_detector.py` uses only BGR image pixels (HSV red mask, morphology,
area/shape filtering) and returns candidate boxes, visible centroids and image
offsets. `check_vision.py` saves both unmodified front-camera frames and annotated
results. `--validate` requires a paused scene and temporarily moves/restores the
target to test both presence and absence. `--show` opens a comparison viewer.
Launcher menu 3 first runs the approach demo, then validates and opens the preview.
The mission runner reads this detector through `CameraTargetProvider` and records
typed observations as advisory telemetry. It does not steer the aircraft or confirm hits.
`perception_state.py` filters those observations over time. The default policy in
`configs/mission.json` requires two consecutive detections for `DETECTED`, reports
short gaps as `TEMPORARILY_LOST`, and changes to `LOST` after three consecutive misses.
Run `python/analyze_perception.py <mission-report.json>` to create per-phase JSON and
Markdown summaries beside a mission report. These states remain diagnostic only.
For each RGB detection, `CameraTargetProvider` also requests AirSim
`DepthPerspective`, rescales the RGB bounding box to the depth resolution, and records
the median valid depth inside the box. RGB and depth are requested separately, so this
is an advisory range estimate rather than a synchronized sensor measurement.
Launcher menu 4 opens `live_camera_view.py`, which continuously displays both
`front_center` feeds in a separate window; it is useful when the UE external view
does not show a camera feed.
The monitor moves the lens to `(X=1.0, Y=0, Z=-0.05)` in each vehicle's local
NED frame so the rotor frame is outside the view.

## Basic framework modules

- `configs/mission.json`: vehicle names, world-NED pads, staging/approach points and safety limits.
- `mission_config.py`: validates that configuration before flight.
- `target_provider.py`: separates ground-truth sensing from camera sensing; both return the same `TargetObservation` shape.
- `perception_state.py`: converts consecutive camera observations into temporal tracking states.
- `analyze_perception.py`: summarizes raw detections, tracking states and depth samples by phase and vehicle.
- `guidance_advisor.py`: emits auditable `HOLD`, alignment, `ADVANCE` or `STOP` advice without commanding a vehicle.
- `recover_airborne.py`: returns both vehicles to their pads after an airborne abort, verifies landing, disarms and releases API control.
- `coordinator.py`: flight-independent A-first/B-fallback arbitration. It refuses a B dispatch until A has missed and the corridor is explicitly clear.

The mission runner now reads its geometry, safety limits and perception thresholds
from `MissionConfig`. Camera observations are logged alongside the collision-based
baseline; a future opt-in visual guidance loop can consume the same interface.

## Batch validation

Run `& .venv/python.exe python/batch_validate.py --repeats 3 --scenarios a_hit fallback`
to repeat both fixed-scene routes. Each iteration restores the balloon, runs one mission,
and records the mission directory in `logs/batch_<timestamp>/batch_report.json` and
`runs.csv`. The batch stops at the first failed mission so an unsafe or diagnostic state
is not reused. Start with one repeat before increasing the count.

Run `& .venv/python.exe python/analyze_batches.py` to combine completed batch reports.
It de-duplicates mission directory names and writes `logs/batch_summary.json` plus
`batch_summary.csv`, including per-scenario pass rates, safety extrema, and fallback
dispatch timing statistics.

For a controlled fallback-offset batch, add `--miss-offset-y -2.5`. Values whose
absolute magnitude is below 1.5 m are rejected before flight because they may cross
the current fixed balloon body.

Use `--target-offset X Y Z` to move the balloon and the corresponding approach/contact
route together. The initial test harness limits X/Y to ±1.0 m and Z to ±0.5 m around
the inspected target position. Batch summaries keep different target offsets in
separate rows.

`configs/validation_matrix.json` lists named position cases. Run one case with
`& .venv/python.exe python/matrix_validate.py --case y_plus_05`, or omit `--case`
to execute the full matrix. The matrix stops at the first failed batch and writes
`logs/matrix_<timestamp>/matrix_report.json`.

## Visual alignment and range-stop demo

Double-click `Run-Visual-Range.cmd` in the project root. It starts Blocks if needed,
recreates the balloon and runs the bounded visual demo. Start with both drones
stationary on the ground and API control released. If a previous run failed and
paused airborne, inspect/recover that scene before rerunning.

DroneA stages at a fixed observation point, aligns from image errors, then advances
at up to 0.15 m/s until the measured surface range is at most 2.5 m. It requires
three centered observations at that range, then returns and lands. DroneB waits.
The altitude setpoint is adjusted from image error using the tested NED position
controller. Staging, altitude feedback and safety bounds use simulator state;
this is a hybrid visual demo, not a wholly vision-only flight system.

Run `python/run_visual_alignment.py` for alignment only, or add `--range` for
forward movement. Reports and annotated frames are under `logs/mission_visual_*`.
Use `Run-Visual-Contact.cmd` to run the next handoff demo: visual alignment and
the 2.5 m range stop are completed first, then the tested simulator-coordinate
contact route is used for the final collision. A new `BalloonTarget` collision is
still required before the balloon is removed; DroneB remains in standby.
RGB/depth requests are sequential; range is an approximate surface estimate, not
validated three-dimensional localization. This demo does not touch the balloon.

## Visual A-miss/B-fallback demo

Double-click `Run-Visual-Fallback.cmd` to run the full two-vehicle handoff. The
launcher restores the balloon and requires both vehicles landed with API control
released. DroneA moves to its validated camera observation point, approaches with
RGB/depth feedback, then the test deliberately marks A as a miss and clears the
corridor. DroneB is dispatched only after that event, moves to its own validated
observation point, performs the same visual approach and range stop, and then
uses the tested final contact route. The final result is accepted only when a new
`BalloonTarget` collision names DroneB; both vehicles must then land and release
API control. Reports and annotated RGB frames are written to
`logs/mission_visual_fallback_<timestamp>/`.

This is the first visual double-vehicle fallback integration. Visual perception
controls approach gating and low-speed alignment; collision remains the authoritative
hit judge. The deliberate A miss is a deterministic test branch, not a learned
miss predictor.
