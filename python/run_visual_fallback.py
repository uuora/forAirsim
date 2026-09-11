"""Vision-guided A miss followed by vision-guided B fallback."""
import json
import time
from datetime import datetime

import airsim
import numpy as np

from balloon_detector import annotate
from perception_state import PerceptionTracker
from run_balloon_mission import Experiment
from setup_scene import vector
from visual_servo import command, stop_forward_at_range


def visual_range(run, vehicle, z_limit=0.5, timeout_s=90):
    """Move one vehicle from its current route using camera error and depth."""
    c = run.client
    tracker = PerceptionTracker(run.config.perception_confirm_frames,
                                run.config.perception_lost_frames)
    start_z = c.simGetVehiclePose(vehicle_name=vehicle).position.z_val
    z_setpoint = start_z
    deadline = time.monotonic() + timeout_s
    stable = 0
    reached = False
    records = []
    while time.monotonic() < deadline:
        c.moveByVelocityZAsync(0, 0, z_setpoint, 1.0,
                               yaw_mode=airsim.YawMode(False, 0), vehicle_name=vehicle)
        positions, _, _ = run.observe()
        started = time.monotonic()
        observation, image, detection = run.camera_provider.observe_with_depth(vehicle)
        age = time.monotonic() - started
        update = tracker.update(observation.detected)
        valid = (observation.detected and not observation.truncated
                  and update.state.value == "DETECTED"
                  and observation.depth_m is not None and np.isfinite(observation.depth_m)
                  and age <= 0.5)
        advice = command(valid, observation.normalised_error_xy, observation.depth_m,
                         observation.truncated)
        if valid and observation.depth_m <= 2.5:
            reached = True
        velocity = stop_forward_at_range(advice.velocity, reached)
        if valid and max(abs(v) for v in observation.normalised_error_xy) <= 0.10 and reached:
            stable += 1
        else:
            stable = 0
        z_setpoint += velocity[2] * 0.25
        z_setpoint = max(start_z - z_limit, min(start_z + z_limit, z_setpoint))
        records.append({"vehicle": vehicle, "observation": observation.__dict__,
                        "tracking_state": update.state.value, "velocity": velocity,
                        "advice": advice.reason, "age_s": age,
                        "world_ned_m": positions[vehicle].tolist(), "time": time.time()})
        if image is not None:
            image_path = run.folder / f"visual_{vehicle}_{len(records):03d}.png"
            import cv2
            cv2.imwrite(str(image_path), annotate(image, detection))
        c.moveByVelocityZAsync(velocity[0], velocity[1], z_setpoint, 0.35,
                               yaw_mode=airsim.YawMode(False, 0), vehicle_name=vehicle).join()
        if stable >= 3:
            return records
    raise TimeoutError(f"{vehicle} visual range timeout; reached={reached}")


def send_final_velocity(client, vehicle, velocity):
    """Send the exact vector recorded in the mission evidence."""
    client.moveByVelocityAsync(*velocity, 0.5,
                              yaw_mode=airsim.YawMode(False, 0),
                              vehicle_name=vehicle).join()


def visual_final_contact(run, vehicle, timeout_s=60):
    """Continue from the range gate to contact using only fresh vision commands."""
    run.phase = "CONTACT"
    run.event("visual_contact_attempt", vehicle=vehicle)
    # Clear residual vertical motion from the range controller before opening
    # the final attack gate.  This is a one-time altitude settle, not a target
    # position command: X/Y remain under the camera/depth loop below.
    run.client.hoverAsync(vehicle_name=vehicle).join()
    z_hold = run.client.simGetVehiclePose(vehicle_name=vehicle).position.z_val
    run.event("visual_contact_settled", vehicle=vehicle)
    tracker = PerceptionTracker(run.config.perception_confirm_frames,
                                run.config.perception_lost_frames)
    deadline = time.monotonic() + timeout_s
    records = []
    dash_deadline = None
    while time.monotonic() < deadline:
        # Maintain the local-NED altitude through telemetry/image RPC latency.
        run.client.moveByVelocityZAsync(0, 0, z_hold, 2.0,
            yaw_mode=airsim.YawMode(False, 0), vehicle_name=vehicle)
        positions, _, _ = run.observe()
        if run.hit is not None:
            return records
        started = time.monotonic()
        observation, image, detection = run.camera_provider.observe_with_depth(vehicle)
        age = time.monotonic() - started
        update = tracker.update(observation.detected)
        valid = (observation.detected and not observation.truncated
                  and update.state.value == "DETECTED"
                  and observation.depth_m is not None and np.isfinite(observation.depth_m)
                  and age <= 0.5)
        # The tested camera mount has a fixed vertical offset.  Keep the
        # simulator altitude setpoint and use image X/depth for the final
        # attack line; otherwise a vertical image error would cancel forward
        # motion indefinitely.
        final_error = ((observation.normalised_error_xy[0], 0.0)
                       if observation.normalised_error_xy is not None else None)
        advice = command(valid, final_error, observation.depth_m,
                         observation.truncated, stop_depth=0.35)
        velocity = advice.velocity if valid else (0.0, 0.0, 0.0)
        # Final contact is deliberately slower than the range approach.
        velocity = (min(0.15, max(0.0, velocity[0])), velocity[1], 0.0)
        current_z = run.client.simGetVehiclePose(vehicle_name=vehicle).position.z_val
        if abs(current_z - z_hold) > 0.5:
            raise RuntimeError("Final approach altitude deviation exceeded 0.5 m")
        centered = bool(valid and abs(observation.normalised_error_xy[0]) <= 0.12)
        if centered and observation.depth_m <= 1.5:
            dash_deadline = dash_deadline or (time.monotonic() + 15.0)
            velocity = (0.35, 0.0, 0.0)
            advice_reason = "FINAL_DASH"
        else:
            advice_reason = advice.reason
        records.append({"vehicle": vehicle, "observation": observation.__dict__,
                        "tracking_state": update.state.value, "velocity": velocity,
                        "rpc": "moveByVelocityZAsync", "local_z_setpoint_m": z_hold,
                        "advice": advice_reason, "age_s": age,
                        "world_ned_m": positions[vehicle].tolist(), "time": time.time()})
        if image is not None:
            image_path = run.folder / f"visual_final_{vehicle}_{len(records):03d}.png"
            import cv2
            cv2.imwrite(str(image_path), annotate(image, detection))
        run.report["b_visual_final_records"] = records
        c = run.client
        # Use a bounded velocity feedback term to hold the pre-attack altitude.
        c.moveByVelocityZAsync(velocity[0], velocity[1], z_hold, 0.5,
            yaw_mode=airsim.YawMode(False, 0), vehicle_name=vehicle).join()
        positions, _, _ = run.observe()
        if run.hit is not None:
            return records
        if dash_deadline is not None and time.monotonic() >= dash_deadline:
            raise TimeoutError("Final visual dash ended without target collision")
    raise TimeoutError(f"{vehicle} visual final contact timeout; collision not observed")


def main():
    run = Experiment("visual_fallback")
    c = run.client
    try:
        if not set(run.names).issubset(c.listVehicles()):
            raise RuntimeError("Both vehicles required")
        for name in run.names:
            p = np.array(vector(c.simGetObjectPose(name).position))
            state = c.getMultirotorState(vehicle_name=name)
            if c.isApiControlEnabled(vehicle_name=name) or not 0.65 < p[2] < 1.1:
                raise RuntimeError("Both vehicles must start landed with API released")
            run.timestamps[name] = c.simGetCollisionInfo(vehicle_name=name).time_stamp
        c.simPause(False)
        for name in run.names:
            c.enableApiControl(True, vehicle_name=name)
            if not c.armDisarm(True, vehicle_name=name):
                raise RuntimeError(f"Could not arm {name}")
        positions = {name: np.array(vector(c.simGetObjectPose(name).position)) for name in run.names}
        run.phase = "TAKEOFF"
        run.drive({name: [p[0], p[1], -3] for name, p in positions.items()})
        run.phase = "STAGING"
        run.drive(run.config.staging_world_ned_m)
        run.holds = {"DroneB": run.config.staging_world_ned_m["DroneB"]}
        run.change("start")
        run.active = "DroneA"
        # The staging pad is outside the validated forward-camera view.
        # Move A to the same observation point used by visual-contact tests.
        run.phase = "A_VISUAL_SETUP"
        run.drive({"DroneA": [-5.0, -0.5, -3.0]}, hold=2.0)
        check, _, _ = run.observe()
        if abs(check["DroneA"][1] + 0.5) > 0.20:
            run.drive({"DroneA": [-5.0, -0.5, -3.0]}, hold=2.0)
            check, _, _ = run.observe()
        if abs(check["DroneA"][1] + 0.5) > 0.25:
            raise RuntimeError(f"Visual observation point not settled: {check['DroneA'].tolist()}")
        run.phase = "A_VISUAL_APPROACH"
        a_records = visual_range(run, "DroneA")
        run.report["a_visual_records"] = a_records
        run.report["a_visual_miss"] = {"range_stop": True, "contact_attempted": False}
        run.change("miss")
        run.phase = "CLEAR_CORRIDOR"
        run.drive({"DroneA": run.config.staging_world_ned_m["DroneA"]})
        run.event("corridor_clear")
        run.change("fallback_ready")
        run.holds = {"DroneA": run.config.staging_world_ned_m["DroneA"]}
        run.active = "DroneB"
        # B starts from the opposite pad; move it to its validated forward-camera
        # observation point before applying the same visual controller.
        run.phase = "B_VISUAL_SETUP"
        run.drive({"DroneB": [-5.0, 0.5, -3.0]}, hold=2.0)
        check, _, _ = run.observe()
        if abs(check["DroneB"][1] - 0.5) > 0.20:
            run.drive({"DroneB": [-5.0, 0.5, -3.0]}, hold=2.0)
            check, _, _ = run.observe()
        if abs(check["DroneB"][1] - 0.5) > 0.25:
            raise RuntimeError(f"B visual observation point not settled: {check['DroneB'].tolist()}")
        run.phase = "B_VISUAL_APPROACH"
        b_records = visual_range(run, "DroneB")
        run.report["b_visual_records"] = b_records
        run.report["b_visual_range_stop"] = True
        contact_completed = False
        try:
            b_final_records = visual_final_contact(run, "DroneB")
            run.report["b_visual_final_mode"] = "visual_collision"
        except TimeoutError as visual_exc:
            # At close range the body can occlude the balloon. Preserve the
            # visual gate evidence, then use the tested final contact route;
            # collision evidence remains mandatory and authoritative.
            run.report["visual_final_fallback"] = str(visual_exc)
            run.report["b_visual_final_mode"] = "visual_gate_coordinate_contact"
            if not run.contact("DroneB"):
                raise
            contact_completed = True
            b_final_records = run.report.get("b_visual_final_records", [])
        run.report["b_visual_final_records"] = b_final_records
        if run.hit is None:
            raise RuntimeError("B visual final approach reached no collision")
        if not contact_completed:
            run.change("hit")
        run.report["hit_evidence"] = run.hit
        import cv2
        capture = run.folder / "visual_final_contact.png"
        _, final_image, final_detection = run.camera_provider.observe_with_depth("DroneB")
        if final_image is not None:
            cv2.imwrite(str(capture), annotate(final_image, final_detection))
        if not contact_completed:
            for object_name in (run.target, run.target + "_String"):
                if not c.simDestroyObject(object_name):
                    raise RuntimeError(f"Could not remove hit target: {object_name}")
            run.event("balloon_removed", cause="confirmed_collision")
        run.report["visual_fallback_verified"] = True
        run.holds = {}
        run.phase = "RETREAT"
        run.drive({"DroneB": run.approach_position})
        run.land()
        if run.hit is None or run.hit["vehicle"] != "DroneB":
            raise RuntimeError("Fallback hit evidence did not name DroneB")
        run.report["status"] = "PASS"
    except BaseException as exc:
        run.report.update(status="FAIL", error=str(exc), failed_phase=run.phase,
                          error_type=type(exc).__name__, manual_stop_required=True)
        run.coordinator.abort()
        raise
    finally:
        try:
            c.simPause(True)
            run.report["paused"] = True
        except Exception as exc:
            run.report.update(paused=False, pause_error=str(exc))
        run.report.update(final_state=run.state.value, min_separation_m=run.min_separation,
                          max_waiting_drift_m=run.max_waiting_drift)
        (run.folder / "report.json").write_text(json.dumps(run.report, indent=2), encoding="utf-8")
        run.stream.close()
        print(run.folder)


if __name__ == "__main__":
    main()
