"""Vision-guided first attempt followed by an optional B fallback.

The default mode keeps the historical, deliberately injected A miss for
regression comparison.  ``--attempt-mode observation`` enables the new path:
the A contact attempt is closed from fresh visual state and authoritative
target-collision evidence rather than from a pre-written miss event.
"""
import argparse
import hashlib
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
from attempt_assessment import AttemptState, ContactAttemptAssessment


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


def committed_dash_velocity(origin, position, max_travel_m=1.35):
    """Continue a visually initiated dash through close-range occlusion."""
    travel_m = float(np.linalg.norm(np.asarray(position) - np.asarray(origin)))
    if travel_m >= max_travel_m:
        raise TimeoutError(
            f"Committed visual dash exceeded {max_travel_m:.2f} m without collision")
    return (0.35, 0.0, 0.0), travel_m


def _assessment_fields(assessment):
    """Convert an attempt assessment to JSON-safe per-frame evidence."""
    return {
        "attempt_state": assessment.state.value,
        "attempt_action": assessment.action.value,
        "attempt_reason": assessment.reason,
        "attempt_elapsed_s": assessment.elapsed_s,
        "attempt_tracking_state": assessment.tracking_state,
        "attempt_observation_fresh": assessment.observation_fresh,
        "attempt_collision_kind": assessment.collision_kind,
    }


def _record_attempt_assessment(run, vehicle, evaluator, assessment):
    """Persist the latest assessment and emit only actual state transitions."""
    run.report.setdefault("attempt_assessments", {})[vehicle] = evaluator.summary()
    if assessment.changed:
        run.event("contact_attempt_assessment", vehicle=vehicle,
                  attempt_state=assessment.state.value,
                  attempt_action=assessment.action.value,
                  reason=assessment.reason,
                  elapsed_s=assessment.elapsed_s,
                  tracking_state=assessment.tracking_state,
                  observation_fresh=assessment.observation_fresh,
                  collision_kind=assessment.collision_kind)


def _store_visual_final_records(run, vehicle, records):
    """Keep a per-vehicle record while preserving existing report keys."""
    run.report.setdefault("visual_final_records_by_vehicle", {})[vehicle] = records
    key = "a_visual_final_records" if vehicle == "DroneA" else "b_visual_final_records"
    run.report[key] = records


def visual_final_contact(run, vehicle, timeout_s=60, allow_handoff=False):
    """Continue from the range gate to contact using fresh vision commands.

    The evaluator is deliberately separate from the flight controller.  It
    labels temporary loss, an unrecovered track, and terminal failure, while
    only an exact target collision can produce ``CONTACT_CONFIRMED``.  The
    optional ``allow_handoff`` switch controls whether an unrecovered track
    immediately returns control to the caller; it is disabled by default so
    legacy diagnostic callers retain their timeout behaviour.
    """
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
    evaluator = ContactAttemptAssessment(timeout_s=timeout_s)
    started_monotonic = time.monotonic()
    assessment = evaluator.start()
    _record_attempt_assessment(run, vehicle, evaluator, assessment)
    deadline = time.monotonic() + timeout_s
    records = []
    dash_deadline = None
    dash_origin = None
    ever_tracked = False
    try:
        while time.monotonic() < deadline:
            # Maintain the local-NED altitude through telemetry/image RPC latency.
            run.client.moveByVelocityZAsync(0, 0, z_hold, 2.0,
                yaw_mode=airsim.YawMode(False, 0), vehicle_name=vehicle)
            positions, _, _ = run.observe()
            elapsed = time.monotonic() - started_monotonic
            if run.hit is not None:
                assessment = evaluator.update(
                    tracking_state=tracker.state.value,
                    observation_fresh=False,
                    collision_kind="TARGET", elapsed_s=elapsed)
                _record_attempt_assessment(run, vehicle, evaluator, assessment)
                records.append({"vehicle": vehicle, "collision_kind": "TARGET",
                                "world_ned_m": positions[vehicle].tolist(),
                                "time": time.time(), **_assessment_fields(assessment)})
                _store_visual_final_records(run, vehicle, records)
                return records
            started = time.monotonic()
            observation, image, detection = run.camera_provider.observe_with_depth(vehicle)
            age = time.monotonic() - started
            update = tracker.update(observation.detected)
            observation_fresh = age <= 0.5
            assessment = evaluator.update(
                tracking_state=update.state.value,
                observation_fresh=observation_fresh,
                elapsed_s=elapsed)
            _record_attempt_assessment(run, vehicle, evaluator, assessment)
            ever_tracked = ever_tracked or assessment.state == AttemptState.TRACKING
            if assessment.state == AttemptState.FAILED:
                raise TimeoutError(f"{vehicle} visual contact attempt timed out")
            handoff = (allow_handoff and ever_tracked
                       and assessment.state == AttemptState.UNCONFIRMED
                       and assessment.action.value == "HANDOFF_ELIGIBLE")
            valid = (observation.detected and not observation.truncated
                     and update.state.value == "DETECTED"
                     and observation.depth_m is not None and np.isfinite(observation.depth_m)
                     and observation_fresh)
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
            if handoff:
                velocity = (0.0, 0.0, 0.0)
                dash_travel_m = None
                advice_reason = "HANDOFF_ELIGIBLE"
            elif dash_origin is not None:
                velocity, dash_travel_m = committed_dash_velocity(
                    dash_origin, positions[vehicle])
                advice_reason = "FINAL_DASH" if valid else "FINAL_DASH_OCCLUDED"
            elif centered and observation.depth_m <= 1.5:
                dash_deadline = time.monotonic() + 15.0
                dash_origin = positions[vehicle].copy()
                velocity, dash_travel_m = committed_dash_velocity(
                    dash_origin, positions[vehicle])
                advice_reason = "FINAL_DASH"
            else:
                dash_travel_m = None
                advice_reason = advice.reason
            records.append({"vehicle": vehicle, "observation": observation.__dict__,
                            "tracking_state": update.state.value, "velocity": velocity,
                            "rpc": "moveByVelocityZAsync", "local_z_setpoint_m": z_hold,
                            "advice": advice_reason, "dash_travel_m": dash_travel_m,
                            "age_s": age,
                            "world_ned_m": positions[vehicle].tolist(), "time": time.time(),
                            **_assessment_fields(assessment)})
            if image is not None:
                image_path = run.folder / f"visual_final_{vehicle}_{len(records):03d}.png"
                import cv2
                cv2.imwrite(str(image_path), annotate(image, detection))
            _store_visual_final_records(run, vehicle, records)
            if handoff:
                raise TimeoutError(
                    f"{vehicle} visual target track lost; handoff eligible")
            c = run.client
            # Use a bounded velocity feedback term to hold the pre-attack altitude.
            c.moveByVelocityZAsync(velocity[0], velocity[1], z_hold, 0.5,
                yaw_mode=airsim.YawMode(False, 0), vehicle_name=vehicle).join()
            positions, _, _ = run.observe()
            elapsed = time.monotonic() - started_monotonic
            if run.hit is not None:
                assessment = evaluator.update(
                    tracking_state=tracker.state.value,
                    observation_fresh=False,
                    collision_kind="TARGET", elapsed_s=elapsed)
                _record_attempt_assessment(run, vehicle, evaluator, assessment)
                records.append({"vehicle": vehicle, "collision_kind": "TARGET",
                                "world_ned_m": positions[vehicle].tolist(),
                                "time": time.time(), **_assessment_fields(assessment)})
                _store_visual_final_records(run, vehicle, records)
                return records
            if dash_deadline is not None and time.monotonic() >= dash_deadline:
                raise TimeoutError("Final visual dash ended without target collision")
        raise TimeoutError(f"{vehicle} visual final contact timeout; collision not observed")
    except TimeoutError as exc:
        if evaluator.state not in (AttemptState.CONTACT_CONFIRMED, AttemptState.FAILED):
            message = str(exc)
            if "track lost" in message:
                reason = "track_lost_before_authoritative_contact"
            elif "Committed visual dash exceeded" in message:
                reason = "dash_bound_exceeded_without_target_collision"
            elif "visual dash ended" in message:
                reason = "visual_dash_timeout_without_target_collision"
            else:
                reason = "visual_contact_timeout_without_target_collision"
            assessment = evaluator.fail(reason, elapsed_s=time.monotonic() - started_monotonic,
                                        tracking_state=tracker.state.value,
                                        observation_fresh=False)
            _record_attempt_assessment(run, vehicle, evaluator, assessment)
        _store_visual_final_records(run, vehicle, records)
        raise
    except BaseException:
        if evaluator.state not in (AttemptState.CONTACT_CONFIRMED, AttemptState.FAILED):
            assessment = evaluator.fail(
                "controller_error_without_target_collision",
                elapsed_s=time.monotonic() - started_monotonic,
                tracking_state=tracker.state.value, observation_fresh=False)
            _record_attempt_assessment(run, vehicle, evaluator, assessment)
        _store_visual_final_records(run, vehicle, records)
        raise


def _complete_visual_hit(run, vehicle, capture_name, verified_key="visual_fallback_verified"):
    """Finish a collision-confirmed visual attempt and return both vehicles."""
    if run.hit is None:
        raise RuntimeError(f"{vehicle} visual contact completed without collision evidence")
    run.change("hit", source="collision_feedback",
              reason_code="TARGET_COLLISION", evaluation_role="completion")
    run.report["hit_evidence"] = run.hit
    import cv2
    _, final_image, final_detection = run.camera_provider.observe_with_depth(vehicle)
    if final_image is not None:
        cv2.imwrite(str(run.folder / capture_name), annotate(final_image, final_detection))
    for object_name in (run.target, run.target + "_String"):
        if not run.client.simDestroyObject(object_name):
            raise RuntimeError(f"Could not remove hit target: {object_name}")
    run.event("balloon_removed", cause="confirmed_collision")
    run.report[verified_key] = True
    run.holds = {}
    run.phase = "RETREAT"
    run.drive({vehicle: run.approach_position})
    run.land()
    if run.hit["vehicle"] != vehicle:
        raise RuntimeError(f"Visual hit evidence did not name {vehicle}")


def main(attempt_mode="fault_injection"):
    if attempt_mode not in ("fault_injection", "observation"):
        raise ValueError("attempt_mode must be 'fault_injection' or 'observation'")
    run = Experiment("visual_fallback")
    for source_name in ("run_visual_fallback.py", "attempt_assessment.py"):
        source_path = run.folder.parents[1] / "python" / source_name
        run.report["source_sha256"][source_name] = hashlib.sha256(
            source_path.read_bytes()).hexdigest()
    run.report["attempt_policy"] = {
        "mode": attempt_mode,
        "completion_authority": "new_target_collision",
        "fault_injection_allowed": attempt_mode == "fault_injection",
        "description": (
            "A miss is deliberately injected after the range gate"
            if attempt_mode == "fault_injection" else
            "A miss is emitted only after the visual contact attempt is assessed"
        ),
    }
    run.report["evaluation"].update(
        mode=("visual_fault_injection" if attempt_mode == "fault_injection"
              else "visual_observation_assessment"),
        policy_id="visual_a_first_fallback_v1",
        decision_source=("test_injection" if attempt_mode == "fault_injection"
                         else "attempt_assessment"),
    )
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
        if attempt_mode == "fault_injection":
            run.report["a_visual_miss"] = {
                "range_stop": True,
                "contact_attempted": False,
                "source": "planned_fault_injection",
                "reason_code": "INJECTED_MISS",
            }
            run.change("miss", source="test_injection",
                       reason_code="INJECTED_MISS", evaluation_role="fault_injection")
        else:
            try:
                a_final_records = visual_final_contact(
                    run, "DroneA", allow_handoff=True)
                run.report["a_visual_final_mode"] = "visual_collision"
            except TimeoutError as visual_exc:
                # A's failure is now an observed bounded-attempt outcome.  The
                # fallback route remains the same and is still collision-gated.
                run.report["a_visual_miss"] = {
                    "range_stop": True,
                    "contact_attempted": True,
                    "source": "observation_assessment",
                    "reason_code": "ATTEMPT_NOT_CONFIRMED",
                    "error": str(visual_exc),
                }
                run.report["a_visual_final_mode"] = "handoff_after_visual_failure"
                run.change("miss", source="observation_assessment",
                           reason_code="ATTEMPT_NOT_CONFIRMED",
                           evaluation_role="autonomous_candidate")
            else:
                run.report["a_visual_final_records"] = a_final_records
                _complete_visual_hit(run, "DroneA", "visual_final_contact_DroneA.png",
                                     verified_key="visual_first_attempt_verified")
                run.report["status"] = "PASS"
                return
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
            b_final_records = visual_final_contact(
                run, "DroneB", allow_handoff=(attempt_mode == "observation"))
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
            run.change("hit", source="collision_feedback",
                       reason_code="TARGET_COLLISION", evaluation_role="completion")
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
        run.report["failure_reason"] = str(exc)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-mode", choices=("fault_injection", "observation"),
                        default="fault_injection",
                        help="Use the historical injected A miss or assess A from observations")
    args = parser.parse_args()
    main(args.attempt_mode)
