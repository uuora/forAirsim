"""Explicit real-telemetry Qt commissioning check; default performs no flight.

--airsim runs the new runtime through its actual Qt start/stop/event-loop path.
The tracking image is grabbed in memory after reference release and saved only
after runtime cleanup. It is a Qt rendering, never an AirSim camera image.
Timeout requests stop and continues waiting for cleanup; it never kills flight.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "geometric_control_validation.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--airsim", action="store_true", help="Explicitly run one real AirSim case through Qt")
    mode.add_argument("--check", action="store_true", help="No-RPC check (also the default)")
    parser.add_argument("--launch", action="store_true", help="Allow runtime launch if configured port is absent")
    parser.add_argument("--case", choices=["hover", "line", "turn", "waypoints"], default="hover")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="New receipt directory; existing directory is refused")
    parser.add_argument("--timeout", type=float, default=180., help="Seconds before requesting stop; cleanup is still awaited")
    args = parser.parse_args(argv)
    if args.launch and not args.airsim:
        parser.error("--launch requires explicit --airsim")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    cfgpath = args.config.resolve()
    cfg = json.loads(cfgpath.read_text(encoding="utf-8-sig"))
    if args.case not in cfg["cases"]:
        parser.error("Selected case is absent from configuration")
    output = (args.output or ROOT / "reports" / ("geometric_live_qt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), cfgpath, ROOT / "python/geometric_validation/qt_panel.py",
             ROOT / "python/geometric_validation/reference.py", ROOT / "python/geometric_validation/runtime.py",
             ROOT / "python/geometric_validation/core.py", ROOT / "python/geometric_validation/audit.py",
             ROOT / "python/dual_target_control/route_view.py", ROOT / "python/dual_target_control/route_scene.py",
             ROOT / "python/dual_target_control/ui.py"]
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
    intake = dict(mode="airsim" if args.airsim else "check", case=args.case, launch_allowed=args.launch,
                  timeout_stop_request_s=args.timeout, config=str(cfgpath), sources_sha256=sources,
                  image_scope="Qt map rendering only; no AirSim camera, perception input, or full-attitude icon claim")
    write_json(output / "request.json", intake)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QFontDatabase
    from PyQt5.QtWidgets import QApplication
    from geometric_validation.qt_panel import GeometricControlWindow, height_vector, measured_yaw

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    if not QFontDatabase().families():
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont("Microsoft YaHei", 9))
    latest = None
    runner_calls = []

    def runner(config_path, case, output_root=None, telemetry=None, stop_event=None, launch=False):
        nonlocal latest
        if not args.airsim:
            raise RuntimeError("No-flight check must never invoke the runtime")
        runner_calls.append(dict(case=case, launch=launch, host_s=time.perf_counter()))
        from geometric_validation.runtime import run

        def observed(frame):
            nonlocal latest
            # Fixed-size immutable observation only. No Qt, I/O, long path
            # conversion, or waiting is performed on the control callback.
            latest = dict(phase=frame.get("phase"), t=frame.get("t"),
                          timestamp_ns=frame.get("timestamp_ns"),
                          position=tuple(frame["position"]) if frame.get("position") is not None else None,
                          quaternion_wxyz=tuple(frame["quaternion_wxyz"]) if frame.get("quaternion_wxyz") is not None else None)
            if telemetry is not None:
                telemetry(frame)

        return run(config_path, case, output_root=output / "runs", telemetry=observed,
                   stop_event=stop_event, launch=launch)

    window = GeometricControlWindow(cfgpath, runner=runner)
    index = window.case_combo.findData(args.case)
    window.case_combo.setCurrentIndex(index)
    window.launch_checkbox.setChecked(args.launch)
    window.show()
    app.processEvents()
    idle_check = dict(no_runtime_calls=not runner_calls, no_control_thread=not window.active,
                      runtime_imported="geometric_validation.runtime" in sys.modules,
                      airsim_imported="airsim" in sys.modules)
    write_json(output / "idle_check.json", idle_check)
    if not args.airsim:
        window.status.setText("离线 Qt 检查：未连接、未起飞；参考预览不是飞行证据。")
        for plane in ("3d", "xy", "xz", "3d"):
            window.view.set_plane(plane)
            app.processEvents()
        saved = window.grab().save(str(output / "idle_qt.png"))
        passed = bool(saved and idle_check["no_runtime_calls"] and idle_check["no_control_thread"]
                      and not idle_check["runtime_imported"] and not idle_check["airsim_imported"])
        result = dict(check_passed=passed, actual_flight=False, case=args.case, idle_check=idle_check,
                      reference_preview_only=True, capture_overview_enabled=False,
                      image="idle_qt.png", image_is_airsim_camera=False)
        write_json(output / "verification.json", result)
        window.close()
        print(json.dumps(dict(output=str(output), **result), ensure_ascii=False), flush=True)
        return 0 if passed else 1

    state = dict(started=False, timeout_stop_requested=False, pixmap=None, capture=None,
                 finished=False, gui_errors=[], result=None)
    durations = cfg["cases"][args.case].get("segment_durations", [6.])
    capture_after = min(1., sum(durations) / 2.)
    begin = time.perf_counter()

    def monitor():
        if state["finished"]:
            return
        try:
            window.refresh()
            frame = latest
            if (state["pixmap"] is None and frame and frame["phase"] == "TRACKING"
                    and frame["t"] is not None and frame["t"] >= capture_after
                    and window.view.snapshot.get("connected")):
                snapshot = window.view.snapshot
                expected_position = height_vector(frame["position"])
                expected_yaw = measured_yaw(frame["quaternion_wxyz"])
                position_match = all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(snapshot["position"], expected_position))
                yaw_difference = math.atan2(math.sin(snapshot["yaw"] - expected_yaw), math.cos(snapshot["yaw"] - expected_yaw))
                # A newly arriving telemetry sample can differ between the
                # observation and mailbox take. Retry next timer in that case.
                if position_match and abs(yaw_difference) < 1e-9:
                    state["pixmap"] = window.grab()
                    state["capture"] = dict(**frame, view_position=snapshot["position"], view_yaw_rad=snapshot["yaw"],
                        ned_to_height_position_match=True, measured_yaw_match=True, host_s=time.perf_counter(),
                        image_is_airsim_camera=False, icon_attitude_scope="measured yaw only",
                        evidence_scope="Qt map from real recorded telemetry after release; decimated display is not an evaluation input")
            if window.active and time.perf_counter() - begin > args.timeout and not state["timeout_stop_requested"]:
                state["timeout_stop_requested"] = True
                write_json(output / "timeout_stop_request.json", dict(elapsed_s=time.perf_counter() - begin,
                    action="stop event requested; continue event loop awaiting runtime cleanup; no forced kill"))
                window.stop()
            if state["started"] and not window.active and window._thread is None:
                finalize()
        except BaseException as exc:
            state["gui_errors"].append(repr(exc))
            if window.active:
                window.stop()
            else:
                finalize()

    def finalize():
        if state["finished"]:
            return
        state["finished"] = True
        timer.stop()
        result = window._result if isinstance(window._result, dict) else {}
        report = result.get("report", {})
        reaudit = None
        audit_error = None
        if result.get("run_dir"):
            try:
                # Recompute from raw logs only after the worker has completed
                # flight and cleanup; UI text is not the verdict source.
                from geometric_validation.audit import audit
                reaudit = audit(Path(result["run_dir"]))
                write_json(output / "independent_reaudit.json", reaudit)
            except BaseException as exc:
                audit_error = repr(exc)
        tracking_saved = False
        if state["pixmap"] is not None:
            # PNG encoding/disk I/O occurs after control and landing, not during
            # the sampled control loop. The pixels remain the tracking grab.
            tracking_saved = state["pixmap"].save(str(output / "tracking_qt.png"))
            write_json(output / "tracking_qt_capture.json", state["capture"])
        window.grab().save(str(output / "final_status_qt.png"))
        unchanged = all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
                        for path, value in sources.items())
        receipt = dict(actual_flight_requested=True, case=args.case, run_dir=result.get("run_dir"),
            runner_calls=runner_calls, runtime_error=window._error, gui_errors=state["gui_errors"],
            timeout_stop_requested=state["timeout_stop_requested"], normal_completion=report.get("normal_completion"),
            outcome=report.get("outcome"), control_constraints_passed=report.get("control_constraints_passed"),
            data_audit_passed=report.get("data_audit_passed"), cleanup=report.get("cleanup"),
            independent_reaudit_passed=reaudit.get("data_audit_passed") if reaudit else None,
            independent_control_constraints_passed=reaudit.get("control_constraints_passed") if reaudit else None,
            independent_audit_error=audit_error, tracking_qt_capture_verified=bool(tracking_saved and state["capture"]),
            source_files_unchanged=unchanged, idle_check=idle_check, image_is_airsim_camera=False,
            image_scope="Qt telemetry rendering; no perception loop or full-attitude icon evidence",
            actual_completion_time_s=report.get("actual_completion_time_s"),
            planning_wall_s=report.get("planning_wall_s"), elapsed_host_s=time.perf_counter() - begin)
        receipt["verification_passed"] = bool(
            receipt["normal_completion"] is True and receipt["control_constraints_passed"] is True
            and receipt["data_audit_passed"] is True and receipt["independent_reaudit_passed"] is True
            and receipt["independent_control_constraints_passed"] is True and report.get("cleanup", {}).get("landed") is True
            and receipt["tracking_qt_capture_verified"] and unchanged and not state["timeout_stop_requested"]
            and not state["gui_errors"] and not window._error and len(runner_calls) == 1)
        write_json(output / "verification.json", receipt)
        state["result"] = receipt
        print(json.dumps(dict(output=str(output), **receipt), ensure_ascii=False), flush=True)
        window.close()
        app.exit(0 if receipt["verification_passed"] else 1)

    timer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(monitor)
    timer.start()

    def start_once():
        state["started"] = True
        window.start()

    QTimer.singleShot(0, start_once)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
