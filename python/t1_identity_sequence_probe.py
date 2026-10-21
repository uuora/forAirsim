"""Grounded RGB-only continuous identity-development sequence.

Only the controlled actors and the virtual camera are repositioned. Vehicles
remain landed and released. Segmentation, depth, phase and actor information
are persisted for evaluation but are never passed to the observer callback.
The identical-balloon recovery condition deliberately has no visible identity
cue after complete target occlusion; it cannot prove successful reidentification.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import time

import airsim
import cv2
import numpy as np

from camera_geometry import scene_intrinsics
from setup_scene import rotate, vector
from t1_balloon_data_probe import ControlledMeshRegistry
from t1_ground_perception_probe import capture, digest, ground_states, save, visible_mask

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("initial_pair", "target_hidden_with_distractor_at_old_target", "return_target_right")
OBSERVABILITY = ("VISUAL_ATTRIBUTE", "UNRESOLVABLE_IDENTICAL")


def planned_slots(configuration):
    """Predeclare every interval, including intervals that cannot be captured."""
    period = float(configuration["minimum_period_s"])
    if not math.isfinite(period) or period < .2:
        raise ValueError("A finite period of at least .2 seconds is required")
    phases = configuration["phases"]
    if [p["phase_id"] for p in phases] != list(PHASES):
        raise ValueError("Exactly the three preregistered sequence phases are required")
    rows, start = [], 0.0
    for phase, expected in zip(phases, (5.0, 2.0, 5.0)):
        duration = float(phase["host_duration_s"])
        if duration != expected:
            raise ValueError("Phase durations must remain 5 / 2 / 5 seconds")
        count = round(duration / period)
        if count <= 0 or not math.isclose(count * period, duration, abs_tol=1e-9):
            raise ValueError("Phase duration must contain whole timing intervals")
        for number in range(count):
            rows.append({"frame_sequence": len(rows) + 1,
                "phase_id": phase["phase_id"],
                "planned_start_relative_s": start + number * period,
                "planned_end_relative_s": start + (number + 1) * period,
                "capture_status": "NOT_CAPTURED", "not_captured_reason": "SESSION_NOT_STARTED"})
        start += duration
    return rows


def validate_protocol(configuration):
    if configuration.get("recovery_observability") not in OBSERVABILITY:
        raise ValueError("Explicit visual observability condition required")
    if configuration.get("target_generation") != 1:
        raise ValueError("Target generation remains one across all sequence phases")
    if configuration.get("formal_efficacy_results") is not False:
        raise ValueError("This sequence is a development interface test")
    if configuration.get("ground_only") is not True:
        raise ValueError("Ground-only protocol required")
    description = configuration["description"]
    expected = ("带白色横条的红色气球" if configuration["recovery_observability"] == "VISUAL_ATTRIBUTE"
        else "持续跟踪开始时位于左侧的红色气球")
    if description != expected:
        raise ValueError("Description must not reveal the post-occlusion target position")
    planned_slots(configuration)
    return configuration


def scene_acceptance(phase_id, truth):
    """Evaluation-only acceptance; observer inference never consumes masks."""
    if phase_id not in PHASES:
        raise ValueError("Unknown phase")
    target = truth["target"]["visible_pixel_count"]
    distractor = truth["distractor"]["visible_pixel_count"]
    occluder = truth["occluder"]["visible_pixel_count"]
    failures = []
    if distractor < 500:
        failures.append("VISIBLE_DISTRACTOR_SUPPORT_BELOW_500")
    if phase_id == PHASES[1]:
        if target != 0:
            failures.append("TARGET_NOT_COMPLETELY_OCCLUDED")
    else:
        if target < 500:
            failures.append("VISIBLE_TARGET_SUPPORT_BELOW_500")
        if occluder != 0:
            failures.append("INACTIVE_OCCLUDER_VISIBLE")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures,
        "scope": "controlled visible instance masks; not identity inference"}


def observer_observation(frame):
    """Closed RGB/self-camera allowlist, copied to isolate callback mutation."""
    pose = frame["response_camera_poses"][0]
    result = {"session_id": frame["session_id"], "target_generation": 1,
        "frame_id": frame["frame_id"], "frame_sequence": frame["frame_sequence"],
        "description": frame["description"], "rgb_path": frame["media"]["rgb"]["path"],
        "rgb_sha256": frame["media"]["rgb"]["sha256"],
        "image_size_wh": frame["image_size_wh"],
        "readback_receipt_monotonic_s": frame["capture_received_host_s"],
        "capture_requested_monotonic_s": frame["capture_started_host_s"],
        "capture_completed_monotonic_s": frame["capture_received_host_s"],
        "camera": {"scene_intrinsics": frame["scene_intrinsics"],
            "pose_local_ned_m": pose["position_local_ned_m"],
            "orientation_xyzw": pose["orientation_xyzw"],
            "world_from_local_translation_ned_m": frame["world_from_local_translation_ned_m"]},
        "capture_metadata": {"rgb_render_readback_timestamp_ns": pose["render_readback_timestamp_ns"],
            "timestamp_scope": "render readback completion, not exposure timestamp"}}
    return copy.deepcopy(result)


def wait_until(deadline, *, clock=time.monotonic, sleeper=time.sleep):
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return
        sleeper(min(.2, remaining))


def _set_phase(client, registry, assets, configuration, phase):
    for name, asset, scale, key in (
        ("T1SequenceTarget", "sphere", configuration["target_scale_xyz"], "target_world_ned_m"),
        ("T1SequenceDistractor", "sphere", configuration["target_scale_xyz"], "distractor_world_ned_m"),
        ("T1SequenceOccluder", "1m_cube_chamfer", configuration["occluder_scale_xyz"], "occluder_world_ned_m")):
        registry.ensure(client, name, assets[asset], phase[key], scale)


def run(output, protocol, observer):
    output, protocol = Path(output).resolve(), Path(protocol).resolve()
    configuration = validate_protocol(json.loads(protocol.read_text(encoding="utf-8-sig")))
    output.mkdir(parents=True, exist_ok=False)
    slots = planned_slots(configuration)
    session = configuration["protocol_id"] + "_" + output.parent.name
    report = {"status": "RUNNING", "kind": "T1_GROUND_CONTINUOUS_IDENTITY_DEVELOPMENT",
        "started_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "protocol": str(protocol), "protocol_sha256": digest(protocol), "session_id": session,
        "target_generation": 1, "formal_efficacy_results": False,
        "flight_commands_issued": False, "arm_commands_issued": False,
        "api_control_acquired": False, "new_flights": 0, "independent_UE_sessions": 1,
        "scheduled_slots": slots, "frames": [], "phase_transition_records": [],
        "truth_access": "constructor/scenario setup and separate evaluation only",
        "observer_input_contract": "closed RGB/self-camera allowlist; no phase, masks, depth or actor coordinates",
        "timestamp_scope": "Host monotonic readback receipts; render timestamps do not establish exposure time",
        "source_sha256": {name: digest(ROOT/"python"/name) for name in (
            "t1_identity_sequence_probe.py", "t1_balloon_data_probe.py", "t1_ground_perception_probe.py",
            "camera_geometry.py", "setup_scene.py")},
        "limits": ["Frames nested within one owned UE session; not independent efficacy trials",
            "No flight, vehicle attitude, trajectory or physical collision validation",
            "Stripe is a controlled visible identity attribute; no cue condition is intentionally unresolvable after complete occlusion",
            "Shared Blocks map/sphere meshes; not appearance or map generalization",
            "Slot misses and late receipts remain in the denominator; phase timing is host time"],
        "ai_assistance": "AI assisted implementation and evidence preparation"}
    save(output/"report.json", report)
    client = airsim.MultirotorClient(timeout_value=15)
    registry, original_camera = ControlledMeshRegistry(), None
    try:
        if not client.ping() or client.simIsPause():
            raise ValueError("Fresh unpaused simulation required")
        report["ground_before"] = ground_states(client)
        vehicle = client.simGetVehiclePose(vehicle_name="DroneA")
        world = np.asarray(vector(client.simGetObjectPose("DroneA").position), float)
        local = np.asarray(vector(vehicle.position), float)
        translation = world-local
        report["world_from_local_translation_ned_m"] = translation.tolist()
        camera = client.simGetCameraInfo("front_center", vehicle_name="DroneA").pose
        inverse = vehicle.orientation.inverse()
        original_camera = airsim.Pose(rotate(camera.position-vehicle.position, inverse), inverse*camera.orientation)
        if not client.simSetSegmentationObjectID(".*", 0, True):
            raise ValueError("Background segmentation reset failed")
        assets = {name.casefold(): name for name in client.simListAssets()}
        if "sphere" not in assets or "1m_cube_chamfer" not in assets:
            raise ValueError("Required controlled mesh assets unavailable")
        _set_phase(client, registry, assets, configuration, configuration["phases"][0])
        red = np.full((64,64,3), [55,35,235], np.uint8)
        cue = red.copy()
        cue[24:40,:,:] = 245
        texture_paths = {"plain": output/"plain_red_texture.png", "stripe": output/"white_stripe_red_texture.png"}
        for key, pixels in (("plain", red), ("stripe", cue)):
            if not cv2.imwrite(str(texture_paths[key]), pixels):
                raise ValueError("Controlled texture persistence failed")
        for name, segid in (("T1SequenceTarget",42), ("T1SequenceDistractor",43), ("T1SequenceOccluder",44)):
            if segid in (42,43):
                key = "stripe" if segid==42 and configuration["recovery_observability"]=="VISUAL_ATTRIBUTE" else "plain"
                if not client.simSetObjectMaterialFromTexture(name, str(texture_paths[key])):
                    raise ValueError("Controlled texture assignment failed")
            if not client.simSetSegmentationObjectID(name, segid) or client.simGetSegmentationObjectID(name.lower()) != segid:
                raise ValueError("Controlled instance segmentation ID mismatch")
        palette_path = ROOT/"external/AirSim/docs/seg_rgbs.txt"
        palette = {int(line.split("\t")[0]): json.loads(line.split("\t")[1])
            for line in palette_path.read_text(encoding="utf-8").splitlines()}
        report["segmentation_palette_sha256"] = digest(palette_path)
        report["segmentation_palette_rgb"] = {str(i): palette[i] for i in (42,43,44)}
        report["texture_sha256"] = {key: digest(path) for key,path in texture_paths.items()}
        pitch, roll, yaw = np.radians(configuration["camera_pitch_roll_yaw_deg"])
        orientation = airsim.to_quaternion(float(pitch),float(roll),float(yaw))
        eye = airsim.Vector3r(*configuration["camera_world_ned_m"])
        camera_relative = airsim.Pose(rotate(eye-airsim.Vector3r(*world), inverse), inverse*orientation)
        client.simSetCameraPose("front_center",camera_relative,vehicle_name="DroneA")
        for _ in range(3):
            capture(client)
        warm_image, _, warm_seg, warm_meta = capture(client)
        warm_truth = {key: visible_mask(warm_seg,palette[segid]) for key,segid in (("target",42),("distractor",43),("occluder",44))}
        report["warmup_scene_acceptance"] = scene_acceptance(PHASES[0],warm_truth)
        if report["warmup_scene_acceptance"]["status"] != "PASS":
            raise ValueError("Initial sequence scene acceptance failed before timing")
        warmup_path = output/"warmup_rgb.png"
        if not cv2.imwrite(str(warmup_path),warm_image):
            raise ValueError("Warmup media persistence failed")
        began = time.monotonic()
        observer.begin(session,configuration["description"],str(warmup_path),
            recovery_observability=configuration["recovery_observability"])
        report["observer_begin"] = {"wall_seconds":time.monotonic()-began,"rgb_sha256":digest(warmup_path),
            "not_a_scheduled_observation":True}
        origin = time.monotonic()
        report["schedule_begin_host_monotonic_s"] = origin
        report["schedule_end_host_monotonic_s"] = origin+slots[-1]["planned_end_relative_s"]
        phase_by_id = {row["phase_id"]:row for row in configuration["phases"]}
        current_phase = PHASES[0]
        report["phase_transition_records"].append({"phase_id":current_phase,"setup_completed_host_monotonic_s":origin,
            "setup_before_timed_schedule":True,"transition_capture_count":4})
        save(output/"report.json",report)
        for slot in slots:
            start = origin+slot["planned_start_relative_s"]
            end = origin+slot["planned_end_relative_s"]
            slot.update(planned_start_host_monotonic_s=start,planned_end_host_monotonic_s=end,
                not_captured_reason="NOT_YET_REACHED")
            wait_until(start)
            if time.monotonic()>=end:
                slot["not_captured_reason"]="SCHEDULE_OVERRUN"
                save(output/"report.json",report)
                continue
            if current_phase != slot["phase_id"]:
                transition_started=time.monotonic()
                _set_phase(client,registry,assets,configuration,phase_by_id[slot["phase_id"]])
                for _ in range(3):
                    capture(client)
                current_phase=slot["phase_id"]
                report["phase_transition_records"].append({"phase_id":current_phase,
                    "setup_started_host_monotonic_s":transition_started,
                    "setup_completed_host_monotonic_s":time.monotonic(),"transition_capture_count":3})
                if time.monotonic()>=end:
                    slot["not_captured_reason"]="PHASE_SETUP_OVERRAN_SLOT"
                    save(output/"report.json",report)
                    continue
            image,ranges,seg,meta=capture(client)
            frame_id=f"sequence_{slot['frame_sequence']:04d}"
            paths={"rgb":output/f"{frame_id}_rgb.png","depth":output/f"{frame_id}_depth.npy",
                "segmentation":output/f"{frame_id}_seg.png"}
            if not cv2.imwrite(str(paths["rgb"]),image) or not cv2.imwrite(str(paths["segmentation"]),seg):
                raise ValueError("Sequence media persistence failed")
            np.save(paths["depth"],ranges,allow_pickle=False)
            info=client.simGetCameraInfo("front_center",vehicle_name="DroneA")
            intrinsic=scene_intrinsics(info,image.shape[1],image.shape[0])
            camera_world=np.asarray(meta["response_camera_poses"][0]["position_local_ned_m"])+translation
            if np.linalg.norm(camera_world-np.asarray(configuration["camera_world_ned_m"]))>.02:
                raise ValueError("Observed virtual camera position disagrees")
            truth={key:visible_mask(seg,palette[segid]) for key,segid in (("target",42),("distractor",43),("occluder",44))}
            truth["target_world_ned_m"]=vector(client.simGetObjectPose("T1SequenceTarget").position)
            truth["distractor_world_ned_m"]=vector(client.simGetObjectPose("T1SequenceDistractor").position)
            truth["truth_kind"]="Controlled visible instance-ID masks; no human or real-world annotations"
            acceptance=scene_acceptance(current_phase,truth)
            meta.update(frame_id=frame_id,frame_sequence=slot["frame_sequence"],session_id=session,target_generation=1,
                description=configuration["description"],phase_id=current_phase,
                image_size_wh=[image.shape[1],image.shape[0]],
                media={key:{"path":str(path),"sha256":digest(path)} for key,path in paths.items()},
                scene_intrinsics=intrinsic,camera_world_ned_m=camera_world.tolist(),
                world_from_local_translation_ned_m=translation.tolist(),evaluation_only=truth,
                scene_acceptance=acceptance,ground_state=ground_states(client),
                planned_start_host_monotonic_s=start,planned_end_host_monotonic_s=end,
                receipt_late_for_planned_interval=meta["capture_received_host_s"]>=end)
            report["frames"].append(meta)
            slot.update(capture_status="CAPTURED",not_captured_reason=None,frame_id=frame_id,
                capture_received_host_s=meta["capture_received_host_s"],
                receipt_late_for_planned_interval=meta["receipt_late_for_planned_interval"])
            save(output/"report.json",report)
            if acceptance["status"] != "PASS":
                slot["observer_status"]="NOT_CALLED_SCENE_ACCEPTANCE_FAILED"
                raise ValueError("Sequence intended scene acceptance failed")
            process_started=time.monotonic()
            try:
                decision=observer.process(observer_observation(meta))
                if not isinstance(decision,dict):
                    raise ValueError("Observer must return a dictionary")
                json.dumps(decision,ensure_ascii=False,allow_nan=False)
                meta["observer_record"]={"status":"RETURNED","process_started_host_monotonic_s":process_started,
                    "process_completed_host_monotonic_s":time.monotonic(),"decision":decision}
                slot["observer_status"]="RETURNED"
            except BaseException as error:
                meta["observer_record"]={"status":"ERROR","error_type":type(error).__name__,
                    "process_started_host_monotonic_s":process_started,"process_completed_host_monotonic_s":time.monotonic()}
                slot["observer_status"]="ERROR"
                raise
            meta["observer_record"]["wall_seconds"]=meta["observer_record"]["process_completed_host_monotonic_s"]-process_started
            save(output/"report.json",report)
        wait_until(report["schedule_end_host_monotonic_s"])
        report["schedule_actual_completion_host_monotonic_s"]=time.monotonic()
        report["status"]="PASS"
    except BaseException as error:
        report.update(status="FAIL",error_type=type(error).__name__)
        for slot in slots:
            if slot["capture_status"]=="NOT_CAPTURED" and slot["not_captured_reason"] in ("NOT_YET_REACHED","SESSION_NOT_STARTED"):
                slot["not_captured_reason"]="SESSION_FAILED_BEFORE_SLOT"
        raise
    finally:
        try:
            observer.close()
            report["observer_closed"]=True
            close_receipt=getattr(observer,"close_receipt",None)
            if close_receipt is not None:
                json.dumps(close_receipt,ensure_ascii=False,allow_nan=False)
                report["observer_close_receipt"]=copy.deepcopy(close_receipt)
        except BaseException as error:
            report.update(status="FAIL",observer_closed=False,observer_close_error_type=type(error).__name__)
        try:
            if original_camera is not None:
                client.simSetCameraPose("front_center",original_camera,vehicle_name="DroneA")
            report["ground_after"]=ground_states(client)
            client.simPause(True)
            report["paused_after"]=bool(client.simIsPause())
            report["safe_ground_closure"]=report["paused_after"] and all(
                row["landed"] and not row["api_control_enabled"] for row in report["ground_after"].values())
        except BaseException as error:
            report.update(status="FAIL",safe_ground_closure=False,closure_error_type=type(error).__name__)
        report["actor_registry_audit"]={"lookup":"direct object pose; no full actor-name inventory",
            "created_names":sorted(registry.created),"operations":registry.audit}
        report["scheduled_slot_count"]=len(slots)
        report["captured_slot_count"]=sum(row["capture_status"]=="CAPTURED" for row in slots)
        report["not_captured_slot_count"]=sum(row["capture_status"]=="NOT_CAPTURED" for row in slots)
        transitions=report["phase_transition_records"]
        ended=report.get("schedule_actual_completion_host_monotonic_s",time.monotonic())
        report["phase_timing_summary"]=[]
        for phase in configuration["phases"]:
            phase_id=phase["phase_id"]
            phase_slots=[row for row in slots if row["phase_id"]==phase_id]
            phase_frames=[row for row in report["frames"] if row["phase_id"]==phase_id]
            transition=next((row for row in transitions if row["phase_id"]==phase_id),None)
            subsequent=None
            if transition is not None:
                index=transitions.index(transition)
                if index+1<len(transitions):
                    subsequent=transitions[index+1]["setup_started_host_monotonic_s"]
            receipts=[row["capture_received_host_s"] for row in phase_frames]
            report["phase_timing_summary"].append({"phase_id":phase_id,
                "planned_host_duration_s":phase["host_duration_s"],
                "planned_slots":len(phase_slots),"captured_slots":len(phase_frames),
                "not_captured_slots":sum(row["capture_status"]=="NOT_CAPTURED" for row in phase_slots),
                "observer_process_wall_seconds":sum(row.get("observer_record",{}).get("wall_seconds",0) for row in phase_frames),
                "phase_setup_wall_seconds":None if transition is None or "setup_started_host_monotonic_s" not in transition else
                    transition["setup_completed_host_monotonic_s"]-transition["setup_started_host_monotonic_s"],
                "stable_scene_host_span_proxy_s":None if transition is None else max(0,
                    (subsequent if subsequent is not None else ended)-transition["setup_completed_host_monotonic_s"]),
                "first_readback_receipt_host_s":min(receipts) if receipts else None,
                "last_readback_receipt_host_s":max(receipts) if receipts else None,
                "timing_scope":"host setup/readback durations, not optical exposure or simulation-clock intervals"})
        report["finished_utc"]=datetime.now(timezone.utc).isoformat()
        save(output/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--observer-module",default="t1_continuous_semantic_observer")
    parser.add_argument("--weights-config",type=Path,default=ROOT/"configs/t1_balloon_detector_v1p2.json")
    arguments=parser.parse_args()
    module=importlib.import_module(arguments.observer_module)
    sequence_observer=module.create_observer(arguments.weights_config)
    run(arguments.output,arguments.protocol,sequence_observer)
