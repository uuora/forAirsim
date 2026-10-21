"""Independent readonly audit of grounded new-frame semantic association.

No model inference, simulator API, credentials, or policy objects are invoked.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_t1_ground_perception import digest, save, identity_score, local_association
from t1_ground_perception_probe import visible_mask


def verify_case(case):
    if case["status"]!="COMPLETE" or case["truth_used_by_policy"] or case["flight_commands_issued"]:
        raise ValueError("Incomplete or inappropriate ground interface record")
    frames=case["frames"]
    indexed={row["frame"]["frame_id"]:row for row in frames}
    if len(indexed)!=len(frames) or not frames:
        raise ValueError("Duplicate or absent live frame IDs")
    previous_seq=-1;previous_time=-1;media_count=0
    for row in frames:
        frame=row["frame"]
        if frame["session_id"]!=case["session_id"] or frame["target_generation"]!=case["target_generation"]:
            raise ValueError("Frame context mismatch")
        if frame["frame_sequence"]<=previous_seq or frame["readback_receipt_monotonic_s"]<=previous_time:
            raise ValueError("Live capture sequence/receipt did not advance")
        previous_seq=frame["frame_sequence"];previous_time=frame["readback_receipt_monotonic_s"]
        if frame["readback_receipt_monotonic_s"]!=row["capture"]["capture_received_host_s"]:
            raise ValueError("Semantic clock was not the actual image response receipt")
        for media in row["media"].values():
            if digest(media["path"])!=media["sha256"]:raise ValueError("Live media changed")
            media_count+=1
        if frame["image_sha256"]!=row["media"]["rgb"]["sha256"]:raise ValueError("RGB binding mismatch")
        seg=cv2.imdecode(np.fromfile(row["media"]["segmentation"]["path"],np.uint8),cv2.IMREAD_COLOR)
        truth={"target":visible_mask(seg,[92,31,106]),"distractor":visible_mask(seg,[207,229,90])}
        if truth!=row["evaluation_only"]:raise ValueError("Stored independent masks do not reproduce")
    final=case["final_decision"]
    box=None;current_age=None;source_age=None
    submitted=[row for row in case["decisions"] if row["status"]=="SUBMITTED"]
    if len(submitted)>1:raise ValueError("More than one submitted transport in one case")
    if final["transport_slot_occupied"]:raise ValueError("Network worker not drained")
    if final["status"]=="SEMANTIC_LOCAL_ASSOCIATION_VALID":
        source=indexed[final["request_binding"]["frame_id"]]["frame"]
        current=indexed[final["current_frame_id"]]["frame"]
        binding=final["request_binding"]
        if not submitted or binding!=submitted[0]["request_binding"]:
            raise ValueError("Final result belongs to another request")
        if (binding["image_sha256"]!=source["image_sha256"] or binding["frame_sequence"]!=source["frame_sequence"]
                or final["current_image_sha256"]!=current["image_sha256"]
                or final["current_frame_sequence"]!=current["frame_sequence"]
                or current["frame_sequence"]<=source["frame_sequence"]
                or current["readback_receipt_monotonic_s"]<=source["readback_receipt_monotonic_s"]):
            raise ValueError("Source/current frame association mismatch")
        source_age=final["semantic_source_age_proxy_s"];current_age=final["current_frame_age_proxy_s"]
        now=current["readback_receipt_monotonic_s"]+current_age
        if (current_age<0 or current_age>case["policy_config"]["current_frame_max_age_s"]
                or source_age<0 or source_age>case["policy_config"]["semantic_max_age_s"]
                or abs(now-source["readback_receipt_monotonic_s"]-source_age)>1e-8
                or now<current["candidates_ready_monotonic_s"]):
            raise ValueError("Freshness reconstruction failed")
        source_candidate=next(item for item in source["candidates"] if item["candidate_id"]==final["source_candidate_id"])
        selected=next(item for item in current["candidates"] if item["candidate_id"]==final["selected_candidate_id"])
        if selected!=final["selected_candidate"]:raise ValueError("Selected candidate payload changed")
        config=case["policy_config"]
        source_assoc=local_association([final["semantic_box_xyxy_pixels"]],
            [item["bbox_xyxy_pixels"] for item in source["candidates"]],
            config["source_association_min_iou"],config["association_margin"])
        current_assoc=local_association([source_candidate["bbox_xyxy_pixels"]],
            [item["bbox_xyxy_pixels"] for item in current["candidates"]],
            config["current_association_min_iou"],config["association_margin"])
        if source_assoc["selected_box"]!=source_candidate["bbox_xyxy_pixels"] or current_assoc["selected_box"]!=selected["bbox_xyxy_pixels"]:
            raise ValueError("Local unique box association did not reproduce")
        if final["server_proposal_ranking"] or final["proposals_transmitted"] or final["request_mode"]!="image_text":
            raise ValueError("Unsupported cloud capability assertion")
        box=selected["bbox_xyxy_pixels"]
        evaluation=identity_score(box,indexed[final["current_frame_id"]]["evaluation_only"])
    else:
        if final["status"]!="UNCERTAIN":raise ValueError("Unrecognized terminal interface state")
        evaluation=identity_score(None,frames[-1]["evaluation_only"])
    if evaluation!=case["evaluation"]:raise ValueError("Independent identity score mismatch")
    return {"case_id":case["case_id"],"final_status":final["status"],"reason":final["reason"],
        "correct_visible_instance":evaluation["all_attempts_correct"],"live_frames":len(frames),"media_hashes":media_count,
        "semantic_source_age_proxy_s":source_age,"current_frame_age_proxy_s":current_age,
        "submitted_requests":len(submitted)}


def run(source_path, output):
    source=json.loads(source_path.read_text(encoding="utf-8-sig"))
    if source["status"]!="PASS" or not source["safe_ground_closure"] or source["flight_commands_issued"]:
        raise ValueError("Completed grounded probe required")
    records=[];submitted_times=[]
    for record in source["semantic_case_records"]:
        path=Path(record["path"])
        if digest(path)!=record["sha256"]:raise ValueError("Case receipt changed")
        case=json.loads(path.read_text(encoding="utf-8"))
        row=verify_case(case)
        for decision in case["decisions"]:
            if decision["status"]=="SUBMITTED":submitted_times.append(decision["request_binding"]["submitted_monotonic_s"])
        if row["final_status"]!=record["final_status"] or row["correct_visible_instance"]!=record["correct_visible_instance"]:
            raise ValueError("Aggregate case differs from raw receipt")
        records.append(row)
    expected=[case["case_id"] for case in source["configuration"]["cases"]]
    if [row["case_id"] for row in records]!=expected:raise ValueError("Planned cases missing or reordered")
    intervals=[right-left for left,right in zip(submitted_times,submitted_times[1:])]
    if any(value<2 for value in intervals):raise ValueError("Global semantic request interval violated")
    summary={"status":"VERIFIED_LIVE_GROUND_SEMANTIC_INTERFACE","source_report":str(source_path.resolve()),
        "source_report_sha256":digest(source_path),"case_count_all":len(records),
        "valid_fresh_current_associations":sum(row["final_status"]=="SEMANTIC_LOCAL_ASSOCIATION_VALID" for row in records),
        "uncertain_cases":sum(row["final_status"]=="UNCERTAIN" for row in records),
        "correct_visible_case_outputs_all_cases":sum(row["correct_visible_instance"] for row in records),
        "actual_live_frames":sum(row["live_frames"] for row in records),
        "live_media_hashes_verified":sum(row["media_hashes"] for row in records),
        "cloud_requests_submitted":len(submitted_times),"minimum_request_interval_s":min(intervals) if intervals else None,
        "maximum_current_frame_age_proxy_s":max(row["current_frame_age_proxy_s"] or 0 for row in records),
        "maximum_semantic_source_age_proxy_s":max(row["semantic_source_age_proxy_s"] or 0 for row in records),
        "independent_UE_sessions":1,"new_flights":0,"formal_method_effect":False,"records":records,
        "source_sha256":{Path(__file__).name:digest(__file__)},
        "limits":["Grounded virtual camera and static cases; no online aircraft trajectory validation",
                  "Case/frame observations are nested in one session; not independent efficacy repeats",
                  "Freshness uses readback receipt proxy, not exposure age or full RGB acquisition latency",
                  "Local geometric association is not persistent identity tracking or calibrated probability"]}
    output.mkdir(parents=True,exist_ok=False);save(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
