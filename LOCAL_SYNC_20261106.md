# Local development handoff, 2026-11-06

- Repository: codex/geometric-control-handoff-20261106 at b59a1e6a6f331dac6a9f26ca089af6bd44527e88.
- GitHub Git transport failed. Retrieved the GitHub source archive and reconstructed the exact commit using existing parent objects. Verified tree cae2112f12fffe9c68dc7d7419f84b000f0c02bf and commit SHA. Origin remains the official GitHub URL.
- Original D:/foraisim branch, uncommitted work, environment and AirSim files were preserved.
- AirSim checked out locally at d109f0dc25d4ae425eca7b279da33df5f5fcb42e. Applied the supplied airsim-local.patch; reverse check passes. No binary builds or live runtime verification performed.
- .venv is a directory junction to D:/foraisim/.venv. Existing Python 3.10.21 dependencies were reused; setup-geometric-control.ps1 -CheckOnly passes. No pip installs or dependency upgrades performed.
- All 39 geometric_validation tests pass. Idle Qt check passes without connection or flight.
- Evidence archive: 303 files verified and restored; final restore check confirms exact original bytes.

## Current scope

Independent geometric desired-attitude/projected-thrust outer loop with SimpleFlight inner loops. This is not a complete self-implemented SE(3) controller. Current waypoints stop at intermediate points. Continuous passing and Bezier/Hermite/B-spline comparison remain unimplemented. ABOM is not required.

Historical evidence: 8 attempts, 7 actual flights, 6 control passes. The 2x speed case completed but violated attitude limits. Actual sampling was about 32 Hz, versus a 50 Hz target. No new flights were performed locally.

## Reproduction discrepancy

Local re-audit matches 7 of 8 historical verdict pairs. The 2x case additionally fails the strict CSV string comparison on one cell: tracking row index 28, actual_tilt_rad, recomputed 0.4113569435096625 versus CSV 0.41135694350966245 (difference 5.551115123125783e-17). Its control failure remains unchanged. Do not claim all historical audits reproduce on this machine. Re-auditing writes audit.json; the restored audit files were returned to their exact archived bytes after diagnosis and all 303 files rechecked.

## Next work

1. Make the numerical CSV audit portable while preserving exact checks for timestamps, booleans and other discrete fields, rejecting nonfinite values and meaningful tampering. Add focused regression coverage; preserve historical data and separate new source conditions.
2. Introduce a distinct continuous-passing task/protocol with nonzero intermediate reference velocities, ordered measured passage/velocity checks, segment timing and a stationary terminal state.
3. Diagnose sampling and attitude dynamics before increasing speed; retain shared thresholds and stop on violation.
4. Only after control validation, implement the three concrete reference generators and time allocation comparisons.

No obstacles, visual input or capture_overview. No automatic publishing or supervisor report generation. Live use requires separate local UE/AirLib/DLL provenance and dynamics-receipt validation.

## Local runtime connection repaired

The four local attempts under logs/geometric_20261106_0940* through 0943* failed before connection/arming: the default editor path was absent and the isolated project had no compiled DLLs. All failure directories were retained.

Configured the machine-only engine root as D:/foraisim/UE_4.27. Copied the existing Blocks Binaries and AirSim plugin from D:/foraisim into this isolated checkout; did not overwrite the original installation. File dates alone do not establish binary compatibility: UTF-16 inspection confirmed the existing DLL contains the dynamics receipt, and live verification then succeeded. No compiler installation is currently required.

Live startup verification passed at .runtime/startup_20261106_103341/verification.json: actual settings, unpaused clock, landed state, no API owner, map/process binding, loaded DLL identity, and fresh dynamics receipt checked. No API ownership, arming or flight commands were issued. This is runtime readiness, not a controller performance test or reproducible source-to-binary build proof. The simulator was left running for the user.

GUI now displays report errors and worker exceptions in a selectable details field while retaining separate outcome/control/audit/landing verdicts. Launcher writes stdout/stderr to .runtime/gui_logs. All 41 geometric tests passed before the final status-text adjustment; focused Qt tests and entry check rerun afterward. New source conditions must remain separate from historical flights.

## Native viewport and full-attitude display

Added an explicit native-view show/hide button using the existing NativeScene worker and positive process/window binding. The geometry-specific adapter resolves the current process's abslog for actual reflection receipts, supports active-flight observation with an independent client, and waits for observer cleanup on close. No flight ownership or camera image requests occur in this adapter.

The geometric map now passes measured quaternion orientation to the existing four-rotor icon. Body/arms/rotor rings/nose follow full roll/pitch/yaw with NED-to-height basis conversion S R S. Old callers without the optional body rotation retain their yaw-only drawing. Measured Euler values are displayed numerically. Display scale and camera stay fixed as attitude changes.

Verification: 50 geometric tests and 56 UI/native/route regressions pass. The latter required copying the existing historical stop/pass frozen package directories from D:/foraisim/reports into this isolated checkout; those are regression inputs, not new experiments. Real native show/hide/show succeeded with reflection receipts and nonblank client-area pixel sampling. Qt desktop/small-window and 3D/XY/XZ renders use the user's existing local waypoint-flight samples; no new flights or AirSim camera screenshots. Receipt and source/input hashes: reports/geometric_display_revision_20261106_v1/verification.json. Future source fingerprints remain separate.

## Three planners completed locally

Latest user scope: finish Bezier, Hermite and B-spline first; minimum-snap is deferred. This supersedes the earlier unimplemented-continuous-task status above. Default Qt now selects the three frozen continuous-passing references. Old commissioning remains available with its original config. Original D:/foraisim launcher still forwards to this checkout.

New protocol geometric_planner_pass_v1 uses the same controller, physics and limits, with ordered measured via-point distance/speed checks. All three use local SLSQP reference time allocation, not an AirSim closed-loop fastest-feasible-time search. Bezier/Hermite are equivalent here; B-spline is a different interpolant. No global optimum or method superiority is claimed.

Reports: reports/geometric_three_planners_20261106_150754_212223/index.html. Three actual flights all passed normal completion, tracking, measured passages, independent audit and safe landing/disarm/API release. Actual times 11.946254848/11.946254848/12.510266880 s. Maximum position errors 0.13324/0.13348/0.07561 m; reference attitude errors 2.7273/2.7288/0.6802 deg. The preceding 150603 batch failed before connection due to a Windows snapshot path limit; it and its not-run entries remain. Windows filesystem paths now use an extended prefix for snapshots.

64 geometric tests pass; repeated independent audits pass for all three new logs. Derived comparison charts have source/input/output hashes. No camera input or screenshots, no obstacles, no gain/gate changes, no GitHub publishing. Source snapshots retain the exact flight versions; subsequent plot-only changes create future condition groups normally.

## Four new task environments

User confirmed straight acceleration/braking, planar S-turns, coupled 3D climbs/descents, and a 135-degree direction change. Presets are in configs/geometric_environments_v1.json; their 12 frozen references are in configs/geometric_environment_plans/. Original baseline frozen config remains in place. Qt now selects an environment and one planner; selection only decodes previews. Comparison lookup filters the current task/controller/gates instead of opening a different environment's latest page.

Shared development acceptance: 3D synchronous position norm <=0.5m; total SO(3) reference attitude error <=10deg. Terminal position <=0.15m, actual speed <=0.15m/s, attitude error <=5deg for 1s. These are tracking errors, not per-axis bounds or absolute tilt limits. Controller/physics/gates and the no-camera/no-obstacle rules are unchanged.

76 tests pass. reports/geometric_environments_20261106_v3/verification.json records offline checks for 15 references including the unchanged baseline and 30 Qt previews, with RPC/process launch blocked and no AirSim import. No new flights were performed for these environments. New source conditions must not be silently merged with the previous three-flight campaign. SLSQP now uses separate segment constraints to handle tied peaks; constant-axis Hermite coefficients are padded to the shared degree-7 format. Stale preset plans are rejected, and shared error gates are visible in the panel.

## E0-E4 actual AirSim suite collected

User explicitly authorized all five environments. New runner: python/run_geometric_environment_suite.py --airsim --launch. Default without flags only validates offline; --collect PATH re-collects without flight. All 15 requests and sources/configs were frozen before the first run. No retries, speed escalation, physics/gain/gate changes, obstacles or camera input. Controller cleanup verified after every run.

Suite: reports/geometric_environment_suite_20261106_160125_208546/index.html. Five sibling campaigns have the same timestamp and suffixes _E0 through _E4, so the GUI comparison lookup can find the correct environment. Fifteen actual flights, all normally completed and passed tracking, measured passages, independent re-audit and safe cleanup. All 39 intermediate waypoint checks passed; five within-environment fingerprint groups match. Actual completion seconds in Bezier/Hermite/B-spline order:

- E0: 11.955255 / 11.943255 / 12.501267.
- E1: 13.119280 / 13.110280 / 13.461287.
- E2: 17.907382 / 17.880381 / 25.206538.
- E3: 23.148494 / 23.151494 / 21.969469.
- E4: 19.644419 / 19.653419 / 19.902425.

Global recorded maxima: synchronous position error 0.133838m, reference SO(3) attitude error 2.729422deg, state interval 0.033000704s. Timing includes terminal hold and excludes preparation/landing. Still no closed-loop fastest-time search or superiority/reliability claim. Minimum-snap remains deferred.

Suite all_flights.csv and all_waypoint_passages.csv retain all requested results; each flight has states_export.csv plus original samples.jsonl, manifest/config/source, events/report/cleanup. Initial audits preserved separately. Derived CSV exports verify identical existing bytes and reject differences. Collection counts acceptance only after independent verdict agreement. Initial evidence verification: 155 hashes, zero mismatches. Read-only postflight check confirmed landed, API owner false, same UE PID 3536.

All 80 geometric tests passed before flights. After flights, only comparison.py was changed to make linear/planar 3D axes readable and physically scaled; its focused regression passed. Original generated plots and source preserved under each campaign's plot_initial/, revised charts have their own plot provenance; suite plot_revision.json records this postflight source difference. Flight source snapshots and raw reports stay unchanged. Future source conditions form a new group normally.

## Overlapping trajectory display fixed

User identified an apparently missing trajectory: the later solid Hermite line covered nearly equivalent Bezier data. All 15 flights have tracking samples. Measured curves now use distinct solid/dashed/dash-dot styles with different sample markers; references are dotted. Each environment page also includes by_method.png/svg with three independent rows and common scales. No numerical offsets, smoothing or rescaling of samples were introduced.

All five E0-E4 pages were redrawn, preserving previous plots/source under plot_before_overlap_fix/. Suite overlap_plot_revision.json records the change. Two targeted plot tests pass, including identical overlapping sample fixtures and source-data preservation. No extra flights or report changes; original flight source snapshots and source groups remain intact.

## Three-dimensional comparison shown directly on the suite page

The suite previously showed only a table and child-page links. It now opens with a dedicated trajectory_3d.png and named E0-E4 tabs; child pages also lead with that 3D figure. Separate PNG/SVG exports remain static scientific figures. render_suite_page refreshes only existing summary/display artifacts, without re-audit or AirSim calls.

Previous suite page/provenance are preserved in page_before_3d/, previous child plots/source in plot_before_3d_entry/. New page_provenance.json and three_d_entry_revision.json bind the display revision. Seven targeted checks pass. Playwright/Edge checks all five environment views at desktop 1280x900 and mobile 390x844, image load/color pixels, keyboard tabs and overflow; receipt under page_3d_browser_check/. No extra flights, camera captures or report changes.

## Display aspect adjusted at user request

Default standalone 3D view now uses independent display box lengths X:Y:H=1.55:1.10:0.80, clearly labelled. The suite offers Display/Metric modes and matching PNG/SVG links; both use identical original coordinates and axis ranges. Orthographic camera elevation is 43deg for planar routes, 28deg for height-changing routes, azimuth -58deg. Axis/camera/sample metadata: trajectory_3d_view.json.

Previous derived assets/source are preserved under plot_before_aspect/ and suite page_before_aspect/. Revision receipt: aspect_revision.json. Existing flight logs, numerical indicators, controller/limits and verdicts stay unchanged. Browser verification covers desktop/mobile, all five environments and both modes under aspect_browser_check/. No extra AirSim flight or camera capture.

## Interactive views and user-authorized Git publication

Result pages now embed offline Plotly 3D views with original tracking samples. Overlay and independent shared-range scenes, linked cameras, method/reference visibility, line width, display/metric proportions and camera presets are available. Mobile independent scenes stack vertically. Original input hashes (762 files) remain unchanged; derived artifacts and former source are preserved under plot_before_interactive/ and page_before_interactive/. All 84 geometric tests pass. Browser checks and source are stored under the suite's interactive_browser_check/.

The user explicitly requested uploading this work to Git. The current handoff is experiments/geometric_planners_20261106/HANDOFF.md, with a separate checksummed evidence archive and optional --package argument in restore-geometric-evidence.py. Historical commissioning evidence remains separate. AirSim local changes still match the existing tracked patch; no new submodule commit, UE distribution, environment or DLL is published. The original D:/foraisim worktree is preserved.
