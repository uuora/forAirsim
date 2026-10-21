"""Recompute live semantic-selected surface geometry with no truth input."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_t1_ground_perception import digest, save
from t1_candidate_surface import candidate_surface_estimate
from t1_pose_geometry import SceneCameraModel, CameraPose
from t1_semantic_perception import Candidate, SemanticFrame


def run(source_path, output):
    probe=json.loads(source_path.read_text(encoding="utf-8-sig"))
    if probe["status"]!="PASS":raise ValueError("Completed probe required")
    by_case={row["case_id"]:row for row in probe["frames"]}
    rows=[]
    for record in probe["semantic_case_records"]:
        if digest(record["path"])!=record["sha256"]:raise ValueError("Semantic receipt changed")
        case=json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        decision=case["final_decision"]
        row={"case_id":case["case_id"],"status":"NO_ESTIMATE_UNCERTAIN", "reason":decision["reason"]}
        if decision["status"]=="SEMANTIC_LOCAL_ASSOCIATION_VALID":
            raw=next(row for row in case["frames"] if row["frame"]["frame_id"]==decision["current_frame_id"])
            f=raw["frame"]
            for media in raw["media"].values():
                if digest(media["path"])!=media["sha256"]:raise ValueError("Selected frame media changed")
            candidates=tuple(Candidate(c["candidate_id"],tuple(c["bbox_xyxy_pixels"]),c["detector_score"],
                c["source_class_name"],c["truncated"]) for c in f["candidates"])
            frame=SemanticFrame(f["session_id"],f["target_generation"],f["frame_id"],f["frame_sequence"],
                f["image_sha256"],*f["image_size_wh"],f["readback_receipt_monotonic_s"],f["candidates_ready_monotonic_s"],
                candidates,Path(raw["media"]["rgb"]["path"]).read_bytes(),json.dumps(f["detector_report"]))
            # Geometry originates in saved camera/depth metadata; target truth is not read.
            camera=SceneCameraModel.from_k(by_case[case["case_id"]]["scene_intrinsics"]["K_nominal"],*f["image_size_wh"])
            captures=raw["capture"]["response_camera_poses"]
            poses=[CameraPose(p["position_local_ned_m"],p["orientation_xyzw"],
                              probe["world_from_local_translation_ned_m"]) for p in captures[:2]]
            pairing={"frame_id":frame.frame_id,"image_sha256":frame.image_sha256,"depth_kind":"DepthPerspective",
                "rgb_readback_timestamp_ns":captures[0]["render_readback_timestamp_ns"],
                "depth_readback_timestamp_ns":captures[1]["render_readback_timestamp_ns"],
                "intrinsics_pairing_scope":"caller_declared_equal_RGB_depth_Scene_intrinsics"}
            estimate=candidate_surface_estimate(frame,depth=np.load(raw["media"]["depth"]["path"],allow_pickle=False),
                camera=camera,rgb_pose=poses[0],depth_pose=poses[1],selection_certificate=decision,pairing_contract=pairing)
            row.update(status="VALID_SURFACE_GEOMETRY_ONLY",reason=None,estimate=estimate)
        rows.append(row)
    valid=[row["estimate"] for row in rows if "estimate" in row]
    summary={"status":"VERIFIED_SELECTED_SURFACE_GEOMETRY_ONLY","source_report":str(source_path.resolve()),
        "source_report_sha256":digest(source_path),"case_count_all":len(rows),"surface_estimates":len(valid),
        "no_estimate_uncertain":len(rows)-len(valid),"target_truth_used":False,"new_flights":0,
        "max_depth_pixel_reprojection_error_px":max(e["depth_pixel_reprojection_max_axis_error_px"] for e in valid) if valid else None,
        "max_rgb_reference_reprojection_error_px":max(e["rgb_reference_reprojection_max_axis_error_px"] for e in valid) if valid else None,
        "source_sha256":{Path(__file__).name:digest(__file__),"t1_candidate_surface.py":digest(Path(__file__).with_name("t1_candidate_surface.py"))},
        "rows":rows,"limits":["Geometry consistency only; no independent surface localization accuracy",
            "RGB/depth equal-intrinsics is a caller declaration based on the capture settings",
            "No physical target center, calibrated identity probability, or flight execution authorization"]}
    output.mkdir(parents=True,exist_ok=False);save(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
