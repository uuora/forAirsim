"""Close development evidence across timing repairs without merging trial units."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSIONS=("v1","v1p1a","v1p2","v1p3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def run(output):
    versions,identities,session_labels,problems=[],set(),[],[]
    frozen=read(ROOT/"configs/trajectory_timed_pilot_v6p2/center.json")["flight_source_sha256"]
    for name,expected in frozen.items():
        if digest(ROOT/"python"/name)!=expected:
            problems.append("FROZEN_FLIGHT_SOURCE_CHANGED:"+name)
    for version in VERSIONS:
        plan_path=ROOT/f"configs/t1_continuous_identity_{version}_20261004/plan.json"
        analysis_path=ROOT/f"logs/t1_continuous_identity_analysis_{version}_final_20261004/report.json"
        plan,analysis=read(plan_path),read(analysis_path)
        campaign_path=Path(plan["campaign_root"])/"campaign_receipt.json"
        campaign=read(campaign_path)
        if campaign["plan_sha256"]!=digest(plan_path) or campaign["status"]!="PASS":
            problems.append("CAMPAIGN_OR_PLAN_FAILED:"+version)
        media_count=runtime_sources=0
        for source in plan["source_inventory"]:
            if digest(source["path"])!=source["sha256"]:
                problems.append("RUNTIME_DECLARED_SOURCE_CHANGED:"+source["relative_path"])
        for episode in plan["episodes"]:
            manifest=read(episode["manifest_path"])
            raw=read(episode["report_path"])
            session_labels.append(raw["session_id"])
            identity=(manifest["engine_process_id"],manifest["engine_started_utc"])
            if identity in identities:
                problems.append("DUPLICATE_OWNED_ENGINE_IDENTITY")
            identities.add(identity)
            if not (manifest["engine_closed_after_safe_completion"] and raw["safe_ground_closure"]
                    and raw["observer_closed"] and raw["new_flights"]==0):
                problems.append("SESSION_CLOSURE_FAILED:"+episode["episode_id"])
            snapshot=Path(episode["manifest_path"]).parent/"source_snapshot"
            for source in plan["source_inventory"]:
                if digest(snapshot/source["relative_path"])!=source["sha256"]:
                    problems.append("RUNTIME_SNAPSHOT_CHANGED:"+source["relative_path"])
                runtime_sources+=1
            if digest(snapshot/"specialist_best.pt")!=plan["weights_sha256"]:
                problems.append("WEIGHT_SNAPSHOT_CHANGED")
            for frame in raw["frames"]:
                for record in frame["media"].values():
                    if digest(record["path"])!=record["sha256"]:
                        problems.append("MEDIA_CHANGED")
                    media_count+=1
        versions.append({"version":version,"plan":str(plan_path),"plan_sha256":digest(plan_path),
            "campaign_receipt":str(campaign_path),"campaign_sha256":digest(campaign_path),
            "analysis":str(analysis_path),"analysis_sha256":digest(analysis_path),
            "planned_sessions":analysis["planned_UE_sessions"],
            "planned_slots":analysis["planned_slots_all"],"captured_slots":analysis["captured_slots_all"],
            "missed_slots":analysis["planned_slots_all"]-analysis["captured_slots_all"],
            "on_time_observer_results":analysis["captured_slots_all"]-analysis["late_observer_completions_all"],
            "late_observer_results":analysis["late_observer_completions_all"],
            "late_readback_receipts":analysis["late_readback_receipts_all"],
            "observed_identity_point_counts":analysis["identity_point_counts_all"],
            "scheduled_identity":analysis["scheduled_identity"],
            "request_submissions":analysis["request_submissions_all"],
            "scene_verified_frames":analysis["valid_scene_slots_all"],
            "media_hashes_reverified":media_count,"source_snapshots_reverified":runtime_sources,
            "cue_recovery_observed_sessions":sum(ep["receipt_events"]["attribute_cued_recovery_observed"] is True
                for ep in analysis["episodes"]),
            "identical_return_refusal_captured_frames":sum((ep["receipt_events"]["identical_post_loss_refusal"] or {}).get(
                "refusal_flag_slots",0) for ep in analysis["episodes"]),
            "all_safe_closed":analysis["all_sessions_safely_closed"],"new_flights":0})
    duplicates={label:count for label,count in Counter(session_labels).items() if count>1}
    result={"status":"PASS" if not problems else "FAIL","kind":"T1_CONTINUOUS_IDENTITY_DEVELOPMENT_CLOSEOUT",
        "created_utc":datetime.now(timezone.utc).isoformat(),"versions":versions,"problems":problems,
        "independently_owned_UE_launches":len(identities),"new_flights":0,
        "request_submissions":sum(row["request_submissions"] for row in versions),
        "media_hashes_reverified":sum(row["media_hashes_reverified"] for row in versions),
        "flight_source_sha256":frozen,"flight_source_freeze_unchanged":not any(
            item.startswith("FROZEN_FLIGHT") for item in problems),
        "historical_session_labels_duplicated_across_protocol_versions":duplicates,
        "historical_session_label_interpretation":"V1/V1.1/V1.2 reused per-condition label in separately fresh observer processes; process identities and protocol hashes distinguish those runs. V1.3 adds campaign and version to the label.",
        "unstarted_preparation":"configs/t1_continuous_identity_v1p1_20261004 contains an earlier preflight freeze followed by a test fix; no campaign or UE launched. Not an experiment failure or started trial.",
        "formal_method_efficacy_completed":False,"trajectory_integration_completed":False,
        "persistent_identity_scope":"Current RGB/YOLO geometry continuity with periodic client-bound semantic review; appearance-cued recovery; identical post-loss rejection is declared task observability rule",
        "limits":["Four serial development protocols, four UE launches each; repeated seed/geometry family, not a randomized formal method comparison",
            "All frames/slots nested within sessions; total frame counts do not supply independent trial repetitions",
            ".2s failures retained; .4s budget changed only in a newly frozen protocol, with .2s phase-end intervals",
            "Spatial replacement with identical same-size pixels may defeat geometry continuity; no persistent instance guarantee",
            "Ground camera/actors only; no trajectory effectiveness or new motion validation",
            "Missing media/late capture truth yields bounds; correct at saved receipts is separate from output by registered deadline",
            "Source/model snapshots are bounded project inventory, not a full transitive engine/build lock"],
        "source_sha256":digest(Path(__file__)),"ai_assistance":"AI assisted implementation and deterministic evidence closeout"}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/"receipt.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status","independently_owned_UE_launches",
        "request_submissions","media_hashes_reverified","new_flights")},ensure_ascii=False))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();run(args.output)
