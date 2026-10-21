"""Reconcile T1 development gates; never promote them to formal efficacy."""
import json
from pathlib import Path
import time

from evaluate_t1_ground_perception import digest, save
from setup_scene import ROOT


def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def scene_acceptance(probe):
    if not probe["frames"]:
        return {"status":"NOT_STARTED_NO_SAVED_FRAMES","all_base_frames":0,"failed_frame_count":0,"rows":[]}
    cases={case["case_id"]:case for case in probe["configuration"]["cases"]}
    rows=[]
    for frame in probe["frames"]:
        case=cases[frame["case_id"]];truth=frame["evaluation_only"];reasons=[]
        if "distractor_world_ned_m" not in case and truth["distractor"]["visible_pixel_count"]!=0:reasons.append("INACTIVE_DISTRACTOR_VISIBLE")
        if "occluder_world_ned_m" not in case and truth["occluder"]["visible_pixel_count"]!=0:reasons.append("INACTIVE_OCCLUDER_VISIBLE")
        if "occluder_world_ned_m" in case and truth["target"]["visible_pixel_count"]!=0:reasons.append("FULL_TARGET_OCCLUSION_NOT_ACHIEVED")
        if "occluder_world_ned_m" not in case and truth["target"]["visible_pixel_count"]<500:reasons.append("VISIBLE_TARGET_MISSING")
        if "distractor_world_ned_m" in case and truth["distractor"]["visible_pixel_count"]<500:reasons.append("VISIBLE_DISTRACTOR_MISSING")
        rows.append({"frame_id":frame["frame_id"],"reasons":reasons})
    complete = len(rows) == len(cases)*probe["configuration"]["frames_per_case"]
    return {"status":"PASS" if complete and all(not row["reasons"] for row in rows) else "FAILED_INTENDED_SCENE_ACCEPTANCE",
            "all_base_frames":len(rows),"failed_frame_count":sum(bool(row["reasons"]) for row in rows),"rows":rows}


def archive_sources(folder, hashes, prefix=ROOT/"python"):
    destination=folder/"source_snapshot";destination.mkdir(exist_ok=True)
    for name,expected in hashes.items():
        path=prefix/name
        if digest(path)!=expected:raise ValueError("Source changed before archival: "+name)
        target=destination/name
        if target.exists() and digest(target)!=expected:raise ValueError("Preserve earlier source snapshot")
        if not target.exists():target.write_bytes(path.read_bytes())


def run():
    output=ROOT/"logs/t1_perception_closeout_v1p1_20261004";output.mkdir(exist_ok=False)
    campaigns=[ROOT/"logs"/name for name in (
        "t1_ground_camera_campaign_20261004_000437_5286608",
        "t1_ground_camera_campaign_20261004_000731_5735269",
        "t1_ground_camera_campaign_20261004_003218_3495496",
        "t1_ground_camera_campaign_20261004_003840_1015193")]
    attempts=[]
    for folder in campaigns:
        manifest=load(folder/"manifest.json");probe=load(folder/"probe/report.json")
        if probe["flight_commands_issued"] or probe["api_control_acquired"] or not probe["safe_ground_closure"]:
            raise ValueError("Ground closure mismatch")
        closed=manifest["engine_closed_after_safe_completion"] or (folder/"failure_closure_receipt.json").is_file()
        if not closed:raise ValueError("Owned engine closure unverified")
        scene=scene_acceptance(probe)
        attempts.append({"manifest":str(folder/"manifest.json"),"manifest_sha256":digest(folder/"manifest.json"),
            "report":str(folder/"probe/report.json"),"report_sha256":digest(folder/"probe/report.json"),
            "execution_status":probe["status"],"safe_ground_closure":True,"engine_closed":True,
            "scene_acceptance":scene,"new_flights":0})
    latest=attempts[-1]
    if latest["scene_acceptance"]["status"]!="PASS":raise ValueError("Latest intended scene gate failed")
    paths={
        "live":ROOT/"logs/t1_live_semantic_validation_v1p1_20261004/summary.json",
        "pose":ROOT/"logs/t1_pose_probe_validation_v1p1_20261004/summary.json",
        "detector":ROOT/"logs/t1_ground_perception_replay_v1p1_20261004/report.json",
        "dataset":ROOT/"datasets/t1_balloon_development_v1p1_20261004/manifest.json",
        "surface":ROOT/"logs/t1_yolo_surface_replay_v1p1_20261004/summary.json",
        "semantic_surface":ROOT/"logs/t1_live_surface_validation_v1p1_20261004/summary.json",
        "updated_cloud_replay":ROOT/"logs/t1_vlx_cloud_after_update_20261004/report.json"}
    data={key:load(path) for key,path in paths.items()}
    if data["dataset"]["visible_instance_labels"]!=18 or data["dataset"]["negative_frame_count"]!=2:
        raise ValueError("Latest dataset intended instance counts differ")
    originals=load(ROOT/"configs/trajectory_timed_pilot_v6p2/center.json")["flight_source_sha256"]
    if any(digest(ROOT/"python"/name)!=expected for name,expected in originals.items()):raise ValueError("Frozen V6.2 original changed")
    archive_sources(paths["detector"].parent,data["detector"]["source_sha256"])
    archive_sources(paths["updated_cloud_replay"].parent,data["updated_cloud_replay"]["source_sha256"])
    archive_sources(paths["dataset"].parent,data["dataset"]["source_sha256"])
    inventory={name:digest(ROOT/"python"/name) for name in (
        "t1_ground_perception_probe.py","t1_live_ground_semantic.py","t1_semantic_perception.py",
        "t1_pose_geometry.py","replay_t1_pose_geometry.py","t1_candidate_surface.py","t1_identity_metrics.py",
        "analyze_t1_live_semantic.py","evaluate_t1_ground_perception.py","retry_t1_vlx_replay.py",
        "export_t1_balloon_dataset.py","replay_t1_live_surfaces.py","closeout_t1_perception.py")}
    archive_sources(output,inventory)
    receipt={"status":"T1_PERCEPTION_DEVELOPMENT_GATES_CLOSED_FORMAL_T1_PENDING",
        "attempts":attempts,"owned_UE_attempts":4,"execution_pass_sessions":3,
        "intended_scene_pass_sessions":sum(row["execution_status"]=="PASS" and row["scene_acceptance"]["status"]=="PASS" for row in attempts),
        "supersedes_closeout":{"path":"logs/t1_perception_closeout_20261004/receipt.json",
            "sha256":digest(ROOT/"logs/t1_perception_closeout_20261004/receipt.json"),
            "correction":"Empty pre-capture failure is NOT_STARTED scene evidence, not vacuous PASS; previous receipt retained"},
        "no_flight_failure_sessions":1,"new_flights":0,"unknown_engine_closures":0,
        "latest":{key:{"path":str(path),"sha256":digest(path)} for key,path in paths.items()},
        "latest_clean_session":{"base_frames":18,"live_frames":data["live"]["actual_live_frames"],
            "valid_semantic_cases":7,"uncertain_cases":2,"visible_instance_labels":18,"global_no_balloon_negative_frames":2,
            "YOLO_surface_estimates_all18":14,"semantic_selected_surface_estimates_all9":7},
        "frozen_original_source_count":len(originals),"frozen_original_sources_unchanged":True,
        "runtime_monotonic_clock_resolution_s":time.get_clock_info("monotonic").resolution,
        "timing_note":"Small color replay durations quantized to clock resolution; zero is not zero-cost inference",
        "source_inventory":inventory,"archive_scope":"New development sources; launcher snapshot before latest launch; bounded inventory, not full library/build lock",
        "formal_method_effect_complete":False,"ai_assistance":"AI assisted implementation, audits and summary"}
    registry_path=ROOT/"configs/thesis_simulation_tasks.json"
    registry=load(registry_path);save(output/"task_registry_before.json",registry)
    task=next(row for row in registry["tasks"] if row["id"]=="T1")
    task["perception_integration_20261004"]={"status":"CLEAN_GROUND_YOLO_VLX_POSE_SURFACE_AND_EVALUATION_GATES_COMPLETED",
        "receipt":"logs/t1_perception_closeout_v1p1_20261004/receipt.json","new_flights":0,
        "intended_scene_pass_sessions":1,"base_frames":18,"additional_live_frames":data["live"]["actual_live_frames"],
        "valid_semantic_cases":7,"uncertain_cases":2,"vlx_API_restored_and_verified":True,
        "specialist_YOLO_training_completed":False,"persistent_instance_tracking_completed":False,
        "formal_method_effect_complete":False}
    task["next_gate"]="episode_grouped_balloon_dataset_and_specialist_YOLO_then_temporal_identity_and_new_trajectory_execution"
    task["status"]="PERCEPTION_INTERFACE_AND_FIXED_HEIGHT_DEVELOPMENT_COMPLETED_FORMAL_T1_PENDING"
    registry["active_stage"]="task_1_episode_grouped_balloon_data_and_identity_trajectory_integration"
    registry["updated"]="2026-10-04"
    save(registry_path,registry);receipt["task_registry_sha256"]=digest(registry_path)
    save(output/"receipt.json",receipt)
    return receipt


if __name__=="__main__":run()
