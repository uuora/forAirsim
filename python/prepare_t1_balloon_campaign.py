"""Freeze a small, episode-grouped camera-only balloon adaptation campaign."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def prepare(output, campaign):
    output = output.resolve(); campaign = campaign.resolve()
    if output.exists() or campaign.exists():
        raise ValueError("Plan and campaign directories must be new")
    output.mkdir(parents=True)
    settings = output/"settings.json"
    shutil.copyfile(ROOT/"configs/t1_perception_ground_probe_v1p1/settings.json", settings)
    splits = ["train"]*4+["validation"]*2+["test"]*2
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_COLLECTION_AND_TRAINING",
        "kind": "T1_CONTROLLED_BALLOON_ADAPTATION_PILOT", "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign), "settings_path": str(settings),
        "settings_sha256": hashlib.sha256(settings.read_bytes()).hexdigest(),
        "ground_only": True, "new_flight_requests": 0, "cloud_requests": 0,
        "cases_per_episode": 24, "frames_per_case": 1,
        "seed_rule": "6100400 + one-based episode index; RNG used only for scene generation",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_source_sha256": {name: hashlib.sha256((ROOT/"python"/name).read_bytes()).hexdigest()
            for name in ("t1_ground_perception_probe.py", "setup_scene.py", "camera_geometry.py", "t1_balloon_data_probe.py")},
        "split_unit": "whole independently launched UE process/session",
        "target_and_distractor_class": "balloon", "label_kind": "visible-mask half-open bounding boxes",
        "model_initial_weights": str(ROOT/"models/yolo11n.pt"),
        "training_protocol": {"epochs": 20, "imgsz": 640, "batch": 8, "workers": 0,
            "device": 0, "seed": 6100420, "deterministic": True, "optimizer": "AdamW",
            "lr0": 0.001, "patience": 20, "amp": False, "pretrained": True,
            "hsv_h": 0.0, "hsv_s": 0.2, "hsv_v": 0.2, "mosaic": 0.5,
            "degrees": 5.0, "translate": 0.1, "scale": 0.2, "fliplr": 0.5,
            "close_mosaic": 5, "plots": False},
        "evaluation_protocol": {"fixed_operating_confidence": 0.25, "nms_iou": 0.5,
            "matching_iou": 0.5, "test_use": "one evaluation after validation-selected best checkpoint",
            "baselines": ["original_COCO_sports_ball", "specialist_balloon"],
            "timing_clock": "perf_counter_ns", "scope": "same-map same-red-material held-out scene seeds only"},
        "episodes": [],
        "limits": ["Small controlled adaptation pilot; two validation and two test UE sessions",
            "All episodes share Blocks map, sphere mesh and red material; no map or appearance generalization claim",
            "Scene labels supervise detector training only; instance identity is a separate downstream task",
            "Virtual camera changes do not validate vehicle flight/trajectory or formal thesis efficacy"],
        "ai_assistance": "AI assisted configuration, implementation and bounded reporting"}
    weights = Path(plan["model_initial_weights"])
    if not weights.is_file():
        # The workspace keeps explicit original checkpoints; never auto-download.
        matches = sorted((ROOT/"models").rglob("yolo11n.pt"))
        if len(matches) != 1:
            raise ValueError("An unambiguous local original YOLO11n checkpoint is required")
        weights = matches[0]
        plan["model_initial_weights"] = str(weights)
    plan["model_initial_weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    for index, split in enumerate(splits, 1):
        episode_id = f"ep{index:02d}_{split}"
        rng = random.Random(6100400+index)
        scale = [round(rng.uniform(1.05,1.30),4), round(rng.uniform(1.0,1.2),4), round(rng.uniform(1.30,1.55),4)]
        cases = []
        kinds = ["single"]*12+["similar_pair"]*8+["occluded_negative"]*4
        rng.shuffle(kinds)
        for case_index, kind in enumerate(kinds, 1):
            camera = [round(-rng.uniform(3.8,6.0),4), round(rng.uniform(-.2,.2),4), round(-3+rng.uniform(-.2,.2),4)]
            target = [0, round(rng.uniform(-.40,.40),4), round(-3+rng.uniform(-.10,.10),4)]
            case = {"case_id": f"{case_index:02d}_{kind}", "condition": kind,
                "target_world_ned_m": target, "camera_world_ned_m": camera,
                "camera_pitch_roll_yaw_deg": [round(rng.uniform(-4,4),4) for _ in range(3)],
                "description": "红色气球"}
            if kind == "similar_pair":
                target[1] = round(-rng.uniform(.75,1.0),4)
                case["distractor_world_ned_m"] = [0,round(rng.uniform(.75,1.0),4),target[2]]
                case["description"] = "最左侧的红色气球"
            elif kind == "occluded_negative":
                case["occluder_world_ned_m"] = [round((a+b)/2,4) for a,b in zip(camera,target)]
            cases.append(case)
        protocol = output/(episode_id+".json")
        save(protocol, {"protocol_id": "t1_balloon_adaptation_"+episode_id+"_20261004",
            "research_stage": "episode_grouped_detector_adaptation_pilot", "formal_efficacy_results": False,
            "seed": 6100400+index, "acquisition_seed": 6100400+index, "split": split, "frames_per_case": 1,
            "target_scale_xyz": scale, "inactive_object_world_ned_m": [30,100,-3],
            "scene_acceptance": "Inactive masks zero; full occlusion target zero; active visible instances >=500 pixels",
            "cases": cases})
        plan["episodes"].append({"episode_id": episode_id, "split": split, "seed": 6100400+index,
            "acquisition_seed": 6100400+index,
            "protocol_path": str(protocol), "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "manifest_path": str(campaign/episode_id/"manifest.json"),
            "report_path": str(campaign/episode_id/"probe/report.json")})
    save(output/"plan.json", plan)
    return plan


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--campaign",type=Path,required=True)
    a=p.parse_args();result=prepare(a.output,a.campaign)
    print(json.dumps({"status":result["status"],"episodes":len(result["episodes"]),"plan":str(a.output/"plan.json")}))
