"""Freeze four grounded continuous-identity sessions after modules are ready.

This only prepares a development protocol and a bounded source/model inventory.
It does not launch UE, invoke inference, train a detector or read API secrets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil

ROOT=Path(__file__).resolve().parents[1]
PHASES=("initial_pair","target_hidden_with_distractor_at_old_target","return_target_right")
SOURCE_FILES=(
    "python/t1_identity_sequence_probe.py","python/prepare_t1_identity_sequence.py",
    "scripts/run-t1-identity-sequence.ps1","python/test_t1_identity_sequence.py",
    "python/t1_balloon_data_probe.py","python/t1_ground_perception_probe.py",
    "python/camera_geometry.py","python/setup_scene.py","python/braking_startup_readiness.py",
    "python/t1_continuous_semantic_observer.py","python/t1_temporal_identity.py",
    "python/t1_semantic_perception.py","python/specialist_balloon_detector.py",
    "python/train_t1_balloon_yolo.py","python/vlx_api_client.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def build_protocol(episode_id,seed,observability,*,hidden_distractor_x=-2.5):
    """Both seeds vary camera range slightly; masks still decide acceptance."""
    if observability not in ("VISUAL_ATTRIBUTE","UNRESOLVABLE_IDENTICAL"):
        raise ValueError("Unknown recovery observability")
    camera_x=-4.0-(seed%2)*.08
    target_y=-.8
    camera=[camera_x,0,-3]
    occluder_x=-1.0
    occluder_y=target_y*((occluder_x-camera_x)/(0-camera_x))
    distractor_y=target_y*((hidden_distractor_x-camera_x)/(0-camera_x))
    if hidden_distractor_x>=occluder_x:
        raise ValueError("Hidden-phase distractor must be in front of the occluder")
    positions=(
        ([0,-.8,-3],[0,.8,-3],[30,100,-3]),
        ([0,-.8,-3],[hidden_distractor_x,distractor_y,-3],[occluder_x,occluder_y,-3]),
        ([0,.8,-3],[0,-.8,-3],[30,100,-3]))
    return {"schema_version":1,"protocol_id":"t1_continuous_identity_"+episode_id+"_20261004",
        "episode_id":episode_id,"acquisition_seed":seed,"split":"DEVELOPMENT_UNPARTITIONED",
        "recovery_observability":observability,"ground_only":True,"formal_efficacy_results":False,
        "description":"带白色横条的红色气球" if observability=="VISUAL_ATTRIBUTE" else "持续跟踪开始时位于左侧的红色气球",
        "target_generation":1,"minimum_period_s":.2,"target_scale_xyz":[1.2,1.2,1.5],
        "occluder_scale_xyz":[.2,1.6,2.0],"camera_world_ned_m":camera,
        "camera_pitch_roll_yaw_deg":[0,0,0],"controlled_target_motion_amplitude_m":0.0,
        "target_texture":"red_64x64_white_horizontal_rows_24_to_39" if observability=="VISUAL_ATTRIBUTE" else "plain_red_64x64",
        "distractor_texture":"plain_red_64x64",
        "texture_Bgr_red":[55,35,235],"texture_Bgr_white":[245,245,245],
        "observer_configuration":{
            "review_period_s":3.0,"current_frame_max_age_s":.5,"semantic_max_age_s":2.0,
            "response_timeout_s":2.0,"min_request_interval_s":2.0,
            "track_max_unreviewed_s":8.0,"track_max_gap_s":.5,"confirmation_frames":2},
        "scene_acceptance":"target>=500 and distractor>=500 in initial/return; target=0 and distractor>=500 while hidden; inactive occluder=0",
        "timing_contract":"60 predeclared .2-second intervals over 12 host seconds; missed slots remain NOT_CAPTURED; actual readback receipts stored separately",
        "truth_access":"scenario constructor and independent evaluation only; RGB/self-camera observer does not receive phase/seg/depth/actor poses",
        "phases":[{"phase_id":phase,"host_duration_s":duration,"target_world_ned_m":target,
            "distractor_world_ned_m":distractor,"occluder_world_ned_m":occluder}
            for phase,duration,(target,distractor,occluder) in zip(PHASES,(5.0,2.0,5.0),positions)],
        "limits":["Development scene semantics are preregistered; no post-occlusion location hint is given to observer",
            "Identical red instances after complete target occlusion lack a visible identity cue; successful reacquisition cannot be inferred from initial-left instruction",
            "Controlled stripe condition is an attribute-cued recovery test, not unconstrained reidentification",
            "Actors and virtual camera move; neither drone flies"]}


def prepare(output,campaign,*,weights_config=None,hidden_distractor_x=-2.5):
    output,campaign=Path(output).resolve(),Path(campaign).resolve()
    weights_config=Path(weights_config or ROOT/"configs/t1_balloon_detector_v1p2.json").resolve()
    if output.exists() or campaign.exists():
        raise ValueError("Plan and campaign destinations must be new")
    # Validate every ready source before creating a plan. Never snapshot .env.
    source_paths=[ROOT/name for name in SOURCE_FILES]
    missing=[str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise ValueError("Declared module inventory is not ready: "+", ".join(missing))
    detector=json.loads(weights_config.read_text(encoding="utf-8-sig"))
    weights=Path(detector["weights"])
    if not weights.is_absolute():
        weights=ROOT/weights
    weights=weights.resolve()
    if not weights.is_file() or digest(weights)!=detector["weights_sha256"]:
        raise ValueError("Explicit trained checkpoint missing or digest changed")
    output.mkdir(parents=True)
    settings=output/"settings.json"
    shutil.copyfile(ROOT/"configs/t1_perception_ground_probe_v1p1/settings.json",settings)
    frozen_config=output/"detector_config.json"
    shutil.copyfile(weights_config,frozen_config)
    packages={}
    for package in ("airsim","numpy","opencv-python","Pillow","torch","ultralytics","requests"):
        try:
            packages[package]=metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package]="NOT_DISTRIBUTION_METADATA_AVAILABLE"
    plan={"schema_version":1,"status":"FROZEN_BEFORE_SEQUENCE_COLLECTION",
        "kind":"T1_GROUND_CONTINUOUS_IDENTITY_CAMPAIGN","created_utc":datetime.now(timezone.utc).isoformat(),
        "campaign_root":str(campaign),"ground_only":True,"formal_efficacy_results":False,
        "new_flights":0,"planned_UE_sessions":4,"planned_slots_per_session":60,"target_generation":1,
        "settings_path":str(settings),"settings_sha256":digest(settings),
        "observer_module":"t1_continuous_semantic_observer","weights_config_path":str(frozen_config),
        "weights_config_sha256":digest(frozen_config),"weights_path":str(weights),"weights_sha256":digest(weights),
        "source_inventory":[{"path":str(path.resolve()),"relative_path":str(path.relative_to(ROOT)).replace("\\","/"),
            "sha256":digest(path)} for path in source_paths],
        "runtime":{"python":platform.python_version(),"python_implementation":platform.python_implementation(),
            "platform":platform.platform(),"packages":packages},
        "episodes":[],"evaluation_unit":"whole independently launched UE session; frames and phases nested",
        "scope":"development mechanism and observability test, not formal thesis method efficacy",
        "source_lock_scope":"declared project modules and checkpoint only; not full transitive package/source/build lock",
        "api_secret_snapshot":"NONE; no .env/credential/endpoint inventory",
        "seed_use":"Two preregistered camera ranges per condition; no stochastic policy RNG claim",
        "limits":["Four sessions, two per visual observability condition, same map and meshes",
            "No persistent train/validation/test split or detector training in this campaign",
            "All started/failed sessions and every scheduled slot retained",
            "No flight or formal algorithm effectiveness"],"ai_assistance":"AI assisted protocol implementation and evidence preparation"}
    for condition,observability in (("visible_identity_cue","VISUAL_ATTRIBUTE"),("identical_no_recovery_cue","UNRESOLVABLE_IDENTICAL")):
        for seed in (6100410,6100411):
            episode_id=f"{condition}_seed{seed}"
            protocol=output/(episode_id+".json")
            save(protocol,build_protocol(episode_id,seed,observability,hidden_distractor_x=hidden_distractor_x))
            plan["episodes"].append({"episode_id":episode_id,"condition":condition,"acquisition_seed":seed,
                "recovery_observability":observability,"protocol_path":str(protocol),"protocol_sha256":digest(protocol),
                "manifest_path":str(campaign/episode_id/"manifest.json"),"report_path":str(campaign/episode_id/"probe/report.json")})
    save(output/"plan.json",plan)
    return plan


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--weights-config",type=Path)
    parser.add_argument("--hidden-distractor-x",type=float,default=-2.5)
    arguments=parser.parse_args()
    plan=prepare(arguments.output,arguments.campaign,weights_config=arguments.weights_config,
        hidden_distractor_x=arguments.hidden_distractor_x)
    print(json.dumps({"status":plan["status"],"episodes":len(plan["episodes"]),"planned_slots":240,
        "plan":str(arguments.output.resolve()/"plan.json")},ensure_ascii=False))
