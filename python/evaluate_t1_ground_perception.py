"""RGB-only detector/cloud replay scored against separate visible instance masks.

Static saved-frame replay, NOT online tracking or trajectory efficacy. Every
attempted frame stays in its method's denominator, including no box/errors.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from balloon_detector import detect_balloon
from benchmark_phase1_perception import cloud_contract
from t1_semantic_perception import box_iou
from setup_scene import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+"\n", encoding="utf-8")


def identity_score(selected_box, evaluation_only, min_iou=.5):
    """Boxes matched to separately saved visible-instance rectangles only."""
    target = evaluation_only["target"]["bbox_xyxy_pixels"]
    distractor = evaluation_only["distractor"]["bbox_xyxy_pixels"]
    target_iou = box_iou(selected_box, target) if selected_box is not None and target else 0.
    distractor_iou = box_iou(selected_box, distractor) if selected_box is not None and distractor else 0.
    if selected_box is None:
        status = "NO_OUTPUT"
    elif target_iou >= min_iou and target_iou-distractor_iou >= .05:
        status = "CORRECT_VISIBLE_INSTANCE"
    elif distractor_iou >= min_iou and distractor_iou-target_iou >= .05:
        status = "WRONG_DISTRACTOR"
    else:
        status = "UNMATCHED_OR_AMBIGUOUS_OUTPUT"
    return {"status": status, "target_visible": target is not None,
            "target_visible_bbox_iou": target_iou, "distractor_visible_bbox_iou": distractor_iou,
            "all_attempts_correct": status == "CORRECT_VISIBLE_INSTANCE",
            "criterion": "Development visible-mask bbox IoU>=.5 and identity margin>=.05",
            "interpretation": "Rendered visible-instance rectangle association; NOT mAP or identity probability"}


def local_association(cloud_boxes, yolo_boxes, min_iou=.5, margin=.15):
    if len(cloud_boxes) != 1:
        return {"status": "NO_SINGLE_SEMANTIC_BOX", "selected_box": None}
    ranked = sorted([(box_iou(cloud_boxes[0], box), index) for index, box in enumerate(yolo_boxes)], reverse=True)
    if not ranked or ranked[0][0] < min_iou:
        return {"status": "NO_LOCAL_YOLO_ASSOCIATION", "selected_box": None}
    if len(ranked) > 1 and ranked[1][0] >= min_iou:
        return {"status": "AMBIGUOUS_LOCAL_YOLO_ASSOCIATION", "selected_box": None}
    if len(ranked) > 1 and ranked[0][0]-ranked[1][0] < margin:
        return {"status": "AMBIGUOUS_LOCAL_YOLO_ASSOCIATION", "selected_box": None}
    return {"status": "SOURCE_FRAME_LOCAL_ASSOCIATION_ONLY", "selected_box": yolo_boxes[ranked[0][1]],
            "source_iou": ranked[0][0], "fresh_current_frame_verified": False,
            "server_proposal_ranking": False}


def run(source_path, output, weights, run_vlx=False):
    from yolo_detector import YoloTargetDetector
    import torch
    import ultralytics
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    if source["status"] != "PASS" or not source["safe_ground_closure"]:
        raise ValueError("Completed ground probe required; preserve failures separately")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "T1_SAVED_RGB_DEVELOPMENT_PERCEPTION_REPLAY", "source_report": str(source_path.resolve()),
        "source_report_sha256": digest(source_path), "independent_UE_sessions": 1,
        "formal_efficacy_results": False, "flight_commands_issued": False,
        "truth_is_not_passed_to_policy": True, "weights_sha256": digest(weights),
        "device": "cpu", "imgsz": 640, "confidence_threshold": .25,
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "yolo_class_mapping": "pretrained COCO sports ball -> balloon, not specialist trained",
        "vlx_attempt_selection": "First saved frame of each preregistered camera case",
        "source_sha256": {name: digest(ROOT/"python"/name) for name in (
            "evaluate_t1_ground_perception.py", "balloon_detector.py", "benchmark_phase1_perception.py",
            "yolo_detector.py", "t1_semantic_perception.py", "vlx_api_client.py")},
        "frames": [], "ai_assistance": "AI assisted scripts and bounded interpretation"}
    def persist(): save(output/"report.json", report)
    persist()
    worker = None
    previews = []
    try:
        started = time.monotonic()
        detector = YoloTargetDetector(weights, confidence=.25, imgsz=640, device="cpu")
        report["model_load_wall_s"] = time.monotonic()-started
        first = source["frames"][0]["media"]["rgb"]
        if digest(first["path"]) != first["sha256"]:
            raise ValueError("Warmup image hash mismatch")
        initial = cv2.imdecode(np.fromfile(first["path"],np.uint8), cv2.IMREAD_COLOR)
        started = time.monotonic()
        detector.detect(initial)
        report["yolo_warmup_wall_s"] = time.monotonic()-started
        report["warmup_trial_count"] = 0
        if run_vlx:
            from dotenv import load_dotenv
            from vlx_api_client import VLXSeekAPIClient
            load_dotenv(ROOT/".env")  # Internal credential loading only; never serialize env or client.
            worker = VLXSeekAPIClient(timeout=3, max_retries=1)
            report.update(vlx_model=worker.model, vlx_request_mode=worker.request_mode,
                vlx_proposals_transmitted=worker.proposals_transmitted, vlx_timeout_s=3, vlx_retries=1)
        seen = set()
        for frame in source["frames"]:
            media = frame["media"]["rgb"]
            if digest(media["path"]) != media["sha256"]:
                raise ValueError("Raw RGB changed")
            image = cv2.imdecode(np.fromfile(media["path"],np.uint8), cv2.IMREAD_COLOR)
            row = {"frame_id": frame["frame_id"], "case_id": frame["case_id"],
                   "rgb_sha256": media["sha256"], "methods": {}}
            report["frames"].append(row)
            preview = image.copy()
            yolo_boxes = []
            for name, operation in (("color",lambda: detect_balloon(image)),("yolo",lambda: detector.detect(image))):
                started = time.monotonic()
                try:
                    result = operation()
                    elapsed = time.monotonic()-started
                    if name == "color":
                        selected = result["selected"]
                        box = None if selected is None else [selected["bbox_xywh"][0], selected["bbox_xywh"][1],
                            selected["bbox_xywh"][0]+selected["bbox_xywh"][2], selected["bbox_xywh"][1]+selected["bbox_xywh"][3]]
                    else:
                        candidates = [item for item in result["candidates"] if item["class_name"] == "balloon"]
                        yolo_boxes = [item["bbox_xyxy_pixels"] for item in candidates]
                        box = yolo_boxes[0] if yolo_boxes else None  # Declared highest detector score baseline.
                    row["methods"][name] = {"status":"RETURNED", "wall_s":elapsed,
                        "selected_box":box, "result":result, "evaluation":identity_score(box,frame["evaluation_only"])}
                except Exception as exc:
                    row["methods"][name] = {"status":"ERROR", "error_type":type(exc).__name__,
                        "wall_s":time.monotonic()-started, "selected_box":None,
                        "evaluation":identity_score(None,frame["evaluation_only"])}
            if worker is not None and frame["case_id"] not in seen:
                started = time.monotonic()
                try:
                    # Only saved RGB and task description enter cloud inference.
                    raw = worker.ground(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)), [],
                                       frame["target_description"],lang="zh",temperature=0.,max_new_tokens=128)
                    elapsed = time.monotonic()-started
                    result = cloud_contract(raw,image.shape[1],image.shape[0])
                    box = result["boxes"][0] if len(result["boxes"]) == 1 else None
                    row["methods"]["vlx_image_text"] = {"status":"RETURNED", "wall_s":elapsed,
                        "selected_box":box,"result":result,"evaluation":identity_score(box,frame["evaluation_only"])}
                    assoc = local_association(result["boxes"],yolo_boxes)
                    row["methods"]["yolo_vlx_source_association"] = {"status":"RETURNED", "wall_s":elapsed,
                        "association":assoc,"selected_box":assoc["selected_box"],
                        "evaluation":identity_score(assoc["selected_box"],frame["evaluation_only"])}
                except Exception as exc:
                    for name in ("vlx_image_text","yolo_vlx_source_association"):
                        row["methods"][name] = {"status":"ERROR", "error_type":type(exc).__name__,
                            "wall_s":time.monotonic()-started, "selected_box":None,
                            "evaluation":identity_score(None,frame["evaluation_only"])}
                seen.add(frame["case_id"])
            for name, result in row["methods"].items():
                box = result["selected_box"]
                if box is not None:
                    color = (0,220,255) if name == "yolo" else (255,160,0)
                    cv2.rectangle(preview,tuple(int(x) for x in box[:2]),tuple(int(x) for x in box[2:]),color,1)
            truth = frame["evaluation_only"]["target"]["bbox_xyxy_pixels"]
            if truth:
                cv2.rectangle(preview,tuple(truth[:2]),tuple(truth[2:]),(0,255,0),2)
            cv2.putText(preview,frame["case_id"]+" | green=eval target",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
            if len(previews)<9 and frame["frame_id"].endswith("_01"):
                previews.append(cv2.resize(preview,(320,240)))
            persist()
            print(f"Evaluated {frame['frame_id']}",flush=True)
        report["summaries"] = {}
        for name in ("color","yolo","vlx_image_text","yolo_vlx_source_association"):
            rows = [row["methods"][name] for row in report["frames"] if name in row["methods"]]
            if not rows: continue
            latencies = [row["wall_s"] for row in rows]
            report["summaries"][name] = {"attempted":len(rows), "returned":sum(row["status"]=="RETURNED" for row in rows),
                "correct_visible_instance_all_attempts":sum(row["evaluation"]["all_attempts_correct"] for row in rows),
                "visible_target_attempts":sum(row["evaluation"]["target_visible"] for row in rows),
                "no_output":sum(row["selected_box"] is None for row in rows),
                "wrong_distractor":sum(row["evaluation"]["status"]=="WRONG_DISTRACTOR" for row in rows),
                "wall_p50_s":float(np.median(latencies)),"wall_p95_s":float(np.percentile(latencies,95)),
                "returned_within_half_second":sum(row["status"]=="RETURNED" and row["wall_s"]<=.5 for row in rows),
                "wall_latency_scope":"All attempted durations include errors; only returned operations count as budget met"}
        if len(previews)==9:
            cv2.imwrite(str(output/"contact_sheet.png"),np.vstack([np.hstack(previews[i:i+3]) for i in (0,3,6)]))
        report["vlx_usage"] = worker.get_usage_summary() if worker else None
        report["status"] = "COMPLETE_WITH_ERRORS" if any(row["status"]=="ERROR" for frame in report["frames"] for row in frame["methods"].values()) else "PASS"
        report["limits"] = ["Saved frames: no online freshness/current-frame proof", "One simulator session, no independent statistical efficacy inference",
            "No tracking-through-occlusion identity claim", "Bbox association only; no calibrated identity probability or mAP",
            "No policy/output selection accesses segmentation; truth only in evaluation"]
    except BaseException as exc:
        report.update(status="FAIL",error_type=type(exc).__name__)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights",type=Path,default=ROOT/"models/yolo11n.pt")
    parser.add_argument("--run-vlx",action="store_true")
    args=parser.parse_args()
    run(args.source,args.output,args.weights,args.run_vlx)
