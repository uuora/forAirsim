"""Retained first cloud failure + separately retried saved RGB, no duplicate YOLO trials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from evaluate_t1_ground_perception import digest, save, identity_score, local_association
from benchmark_phase1_perception import cloud_contract
from vlx_api_client import VLXSeekAPIClient
from setup_scene import ROOT


def run(source_path, output):
    previous = json.loads(source_path.read_text(encoding="utf-8-sig"))
    scene_path = Path(previous["source_report"])
    if digest(scene_path) != previous["source_report_sha256"]:
        raise ValueError("Ground data changed")
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    by_id = {frame["frame_id"]:frame for frame in scene["frames"]}
    attempts = [row for row in previous["frames"] if "vlx_image_text" in row["methods"]]
    if any(row["methods"]["vlx_image_text"]["status"]!="ERROR" for row in attempts):
        raise ValueError("This recovery is only for the retained failed cloud attempts")
    output.mkdir(parents=True,exist_ok=False)
    load_dotenv(ROOT/".env")  # No credential serialization.
    worker = VLXSeekAPIClient(timeout=3,max_retries=1)
    report = {"status":"RUNNING","kind":"RETRY_CLOUD_RGB_REPLAY_SEPARATE_FROM_INITIAL_FAILURES",
        "started_utc":datetime.now(timezone.utc).isoformat(),"prior_report":str(source_path.resolve()),
        "prior_report_sha256":digest(source_path),"source_report":str(scene_path),"source_report_sha256":digest(scene_path),
        "independent_UE_sessions":1,"new_independent_sessions":0,"new_flights":0,
        "sampling":"Same nine first-of-case RGBs; initial errors remain in original report",
        "request_mode":worker.request_mode,"proposals_transmitted":worker.proposals_transmitted,
        "model":worker.model,"timeout_s":3,"max_retries":1,"formal_efficacy_results":False,
        "source_sha256":{name:digest(ROOT/"python"/name) for name in (
            "retry_t1_vlx_replay.py","evaluate_t1_ground_perception.py","vlx_api_client.py",
            "benchmark_phase1_perception.py","t1_semantic_perception.py")},"frames":[]}
    def persist(): save(output/"report.json",report)
    persist()
    try:
        for row in attempts:
            source = by_id[row["frame_id"]]
            media = source["media"]["rgb"]
            if digest(media["path"]) != media["sha256"]:
                raise ValueError("Bound PNG hash changed")
            image = cv2.imdecode(np.fromfile(media["path"],np.uint8),cv2.IMREAD_COLOR)
            yolo_boxes = [item["bbox_xyxy_pixels"] for item in row["methods"]["yolo"].get("result",{}).get("candidates",[])
                          if item["class_name"]=="balloon"]
            result_row = {"frame_id":source["frame_id"],"case_id":source["case_id"],"image_sha256":media["sha256"],"methods":{}}
            report["frames"].append(result_row)
            started = time.monotonic()
            try:
                raw = worker.ground(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),[],source["target_description"],
                                   lang="zh",temperature=0.,max_new_tokens=128)
                elapsed = time.monotonic()-started
                result = cloud_contract(raw,image.shape[1],image.shape[0])
                box = result["boxes"][0] if len(result["boxes"])==1 else None
                assoc = local_association(result["boxes"],yolo_boxes)
                result_row["methods"]["vlx_image_text"] = {"status":"RETURNED","wall_s":elapsed,"result":result,
                    "selected_box":box,"evaluation":identity_score(box,source["evaluation_only"])}
                result_row["methods"]["yolo_vlx_source_association"] = {"status":"RETURNED","wall_s":elapsed,
                    "association":assoc,"selected_box":assoc["selected_box"],
                    "evaluation":identity_score(assoc["selected_box"],source["evaluation_only"])}
            except Exception as exc:
                for name in ("vlx_image_text","yolo_vlx_source_association"):
                    result_row["methods"][name] = {"status":"ERROR","error_type":type(exc).__name__,
                        "wall_s":time.monotonic()-started,"selected_box":None,
                        "evaluation":identity_score(None,source["evaluation_only"])}
            persist()
            print("Cloud retry finished: "+source["frame_id"],flush=True)
        report["summaries"] = {}
        for name in ("vlx_image_text","yolo_vlx_source_association"):
            rows = [row["methods"][name] for row in report["frames"]]
            returned = [row["wall_s"] for row in rows if row["status"]=="RETURNED"]
            report["summaries"][name] = {"attempted":len(rows),"returned":len(returned),
                "error_count":len(rows)-len(returned),
                "correct_visible_instance_all_attempts":sum(row["evaluation"]["all_attempts_correct"] for row in rows),
                "no_output":sum(row["selected_box"] is None for row in rows),
                "returned_wall_p50_s":float(np.median(returned)) if returned else None,
                "returned_wall_p95_s":float(np.percentile(returned,95)) if returned else None,
                "returned_within_half_second":sum(x<=.5 for x in returned)}
        report["usage"] = worker.get_usage_summary()
        report["status"] = "PASS" if report["summaries"]["vlx_image_text"]["error_count"]==0 else "COMPLETE_WITH_ERRORS"
        report["limits"] = ["Source-frame local association only; no fresh current-frame proof",
            "Retry of same data; no additional independent samples", "No calibrated probability, online identity continuity or planning effect"]
    except BaseException as exc:
        report.update(status="FAIL",error_type=type(exc).__name__)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.output)
