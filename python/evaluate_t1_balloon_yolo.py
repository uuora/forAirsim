"""Single frozen test evaluation of COCO sports-ball and specialist balloon.

RGB-only detector calls precede independent label scoring. Every test frame and
every prediction is retained. This evaluates class detection on one simulated
map/material; it does not establish designated identity or trajectory benefit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from train_t1_balloon_yolo import (configure_offline_runtime, dataset_rows,
    disable_integrations, forbid_network, read_json, runtime_versions, save_json,
    sha256, source_snapshot, utc_now, validate_binding, validate_label_file,
    validate_split_files, write_data_yaml)


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2])-max(a[0], b[0]))*max(
        0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


def match_boxes(predictions, truth_boxes, matching_iou=0.5):
    """Maximum cardinality one-to-one IoU matching, deterministic edge order.

    This keeps duplicate detections as false positives and avoids the greedy
    pairing error where one ambiguous proposal consumes a second object's only
    match. It does not compare scores with class/instance truth.
    """
    if not 0 < matching_iou <= 1:
        raise ValueError("Invalid matching IoU")
    for box in list(predictions)+list(truth_boxes):
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid matching box")
    edges = {p: sorted((t for t, box in enumerate(truth_boxes)
        if box_iou(predictions[p], box) >= matching_iou),
        key=lambda t: (-box_iou(predictions[p], truth_boxes[t]), t))
        for p in range(len(predictions))}
    owner = {}
    def augment(p, visited):
        for t in edges[p]:
            if t in visited:
                continue
            visited.add(t)
            if t not in owner or augment(owner[t], visited):
                owner[t] = p
                return True
        return False
    for p in range(len(predictions)):
        augment(p, set())
    matches = [{"prediction_index": p, "truth_index": t,
        "iou": box_iou(predictions[p], truth_boxes[t])}
        for t, p in sorted(owner.items())]
    tp = len(matches)
    return {"matches": matches, "true_positive": tp,
        "false_positive": len(predictions)-tp, "false_negative": len(truth_boxes)-tp,
        "ground_truth_count": len(truth_boxes), "prediction_count": len(predictions),
        "all_visible_objects_detected": len(truth_boxes) > 0 and tp == len(truth_boxes),
        "all_visible_objects_detected_without_extra_boxes": len(truth_boxes) > 0 and tp == len(truth_boxes) == len(predictions),
        "negative_frame": len(truth_boxes) == 0,
        "negative_frame_false_positive": len(truth_boxes) == 0 and len(predictions) > 0}


def aggregate_frames(rows):
    if not rows:
        raise ValueError("Cannot aggregate no frames")
    tp = sum(row["score"]["true_positive"] for row in rows)
    fp = sum(row["score"]["false_positive"] for row in rows)
    fn = sum(row["score"]["false_negative"] for row in rows)
    negative = [row for row in rows if row["score"]["negative_frame"]]
    two = [row for row in rows if row["score"]["ground_truth_count"] == 2]
    times = [row["elapsed_ns"]/1e6 for row in rows if row["status"] == "RETURNED"]
    return {"frames": len(rows), "returned_frames": len(times),
        "error_frames": sum(row["status"] != "RETURNED" for row in rows),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/(tp+fn) if tp+fn else None,
        "negative_frames": len(negative),
        "negative_frames_with_false_positive": sum(row["score"]["negative_frame_false_positive"] for row in negative),
        "two_visible_balloon_frames": len(two),
        "both_balloons_detected_frames": sum(row["score"]["all_visible_objects_detected"] for row in two),
        "both_balloons_detected_without_extra_boxes_frames": sum(row["score"]["all_visible_objects_detected_without_extra_boxes"] for row in two),
        "latency_ms_median": float(np.median(times)) if times else None,
        "latency_ms_p95": float(np.percentile(times, 95)) if times else None,
        "latency_ms_min": min(times) if times else None,
        "latency_ms_max": max(times) if times else None,
        "denominator_unit": "frames nested within independently launched UE episodes"}


def comparison_visuals(output, manifest, frame_results):
    """First preregistered zero/one/two-visible-object test case, without cherry picking."""
    import cv2
    chosen = {}
    by_key = {(row["episode_id"], row["frame_id"], row["model"]): row for row in frame_results}
    for row in dataset_rows(manifest):
        count = row["visible_balloon_count"]
        if row["split"] != "test" or count not in (0, 1, 2) or count in chosen:
            continue
        image = cv2.imdecode(np.fromfile(row["image"], np.uint8), cv2.IMREAD_COLOR)
        panels = []
        for name in ("original_COCO_sports_ball", "specialist_balloon"):
            panel = image.copy()
            result = by_key[(row["episode_id"], row["frame_id"], name)]
            for truth in result["truth_boxes_evaluation_only"]:
                cv2.rectangle(panel, tuple(int(round(x)) for x in truth[:2]),
                    tuple(int(round(x)) for x in truth[2:]), (0, 200, 0), 1)
            for candidate in result["detection"]["candidates"]:
                box = candidate["bbox_xyxy_pixels"]
                cv2.rectangle(panel, tuple(int(round(x)) for x in box[:2]),
                    tuple(int(round(x)) for x in box[2:]), (0, 160, 255), 2)
            cv2.putText(panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            panels.append(panel)
        path = output/f"first_test_{count}_visible_comparison.png"
        ok, encoded = cv2.imencode(".png", np.concatenate(panels, axis=1))
        if not ok:
            raise ValueError("Comparison image encoding failed")
        encoded.tofile(path)
        chosen[count] = {"path": str(path.resolve()), "sha256": sha256(path),
            "episode_id": row["episode_id"], "frame_id": row["frame_id"],
            "selection_rule": "first in frozen manifest order with this visible-object count",
            "green_boxes": "independent visible-mask GT; evaluation only",
            "orange_boxes": "RGB-only detector predictions"}
    return chosen


def run(manifest_path: Path, training_receipt_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "PREFLIGHT", "kind": "T1_BALLOON_FIXED_TEST_EVALUATION",
        "created_utc": utc_now(), "training_receipt": str(training_receipt_path.resolve()),
        "training_receipt_sha256": sha256(training_receipt_path),
        "dataset_manifest": str(manifest_path.resolve()), "dataset_manifest_sha256": sha256(manifest_path),
        "clock": "perf_counter_ns", "timing_input": "decoded BGR; CPU end-to-end detector call after five warmups",
        "new_flight_requests": 0, "cloud_requests": 0, "frames": [], "failure_attempts": [],
        "test_use": "single invocation; fixed operating point plus preregistered low-confidence specialist mAP",
        "score_interpretation": "detector score; NOT calibrated identity probability",
        "designated_identity_evaluated": False, "formal_trajectory_benefit_evaluated": False}
    save_json(output/"report.json", report)
    start_ns = time.perf_counter_ns()
    try:
        receipt = read_json(training_receipt_path)
        if receipt.get("status") != "PASS" or receipt.get("checkpoint_selection_data") != "validation":
            raise ValueError("Completed validation-selected training required")
        if receipt["dataset_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Training and evaluation dataset binding differs")
        plan_path = Path(receipt["plan"])
        manifest, plan = validate_binding(manifest_path, plan_path)
        if receipt["plan_sha256"] != sha256(plan_path):
            raise ValueError("Training plan changed")
        checked = validate_split_files(manifest, {"test"})
        best, baseline = Path(receipt["best_weights"]), Path(receipt["initial_weights"])
        if sha256(best) != receipt["best_weights_sha256"] or sha256(baseline) != plan["model_initial_weights_sha256"]:
            raise ValueError("Checkpoint weights changed")
        marker = training_receipt_path.parent/"test_consumption.json"
        if marker.exists():
            raise RuntimeError("This training receipt already has a consumed test evaluation")
        source_hashes = source_snapshot(output)
        if source_hashes != receipt["source_sha256"]:
            raise ValueError("Evaluation source differs from pre-training source snapshot")
        protocol = plan["evaluation_protocol"]
        if (protocol["fixed_operating_confidence"], protocol["nms_iou"], protocol["matching_iou"]) != (.25, .5, .5):
            raise ValueError("Unsupported frozen operating point")
        report.update(source_sha256=source_hashes, runtime_versions=runtime_versions(),
            offline_runtime=configure_offline_runtime(), plan_sha256=sha256(plan_path),
            checked_test_files=checked, operating_point=protocol,
            model_weights={"original_COCO_sports_ball": {"path": str(baseline), "sha256": sha256(baseline)},
                "specialist_balloon": {"path": str(best), "sha256": sha256(best)}})
        with forbid_network():
            import cv2
            import torch
            from ultralytics import YOLO
            from specialist_balloon_detector import SpecialistBalloonDetector
            from yolo_detector import YoloTargetDetector
            disable_integrations()
            detectors = {
                "original_COCO_sports_ball": YoloTargetDetector(baseline, confidence=.25, imgsz=640, device="cpu"),
                "specialist_balloon": SpecialistBalloonDetector(best, confidence=.25, imgsz=640,
                    device="cpu", nms_iou=.5)}
            report["cpu_runtime"] = {"device": "cpu", "imgsz": 640,
                "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}
            warmup_row = next(row for row in dataset_rows(manifest) if row["split"] == "train")
            if sha256(warmup_row["image"]) != warmup_row["rgb_sha256"]:
                raise ValueError("Warmup training RGB hash mismatch")
            warmup_image = cv2.imdecode(np.fromfile(warmup_row["image"], np.uint8), cv2.IMREAD_COLOR)
            report["warmup"] = {}
            for name, detector in detectors.items():
                before = time.perf_counter_ns()
                for unused in range(5):
                    detector.detect(warmup_image)
                report["warmup"][name] = {"calls": 5, "elapsed_ns": time.perf_counter_ns()-before,
                    "split": "train", "image_sha256": warmup_row["rgb_sha256"], "included_in_test_latency": False}
            with marker.open("x", encoding="utf-8") as stream:
                json.dump({"status": "TEST_CONSUMED", "created_utc": utc_now(),
                    "training_receipt_sha256": sha256(training_receipt_path),
                    "evaluation_output": str(output.resolve()), "test_images": len(checked),
                    "tuning_after_test_authorized": False}, stream, indent=2)
            report.update(status="EVALUATING", test_consumption_marker=str(marker.resolve()),
                          test_started_utc=utc_now())
            save_json(output/"report.json", report)
            for row in dataset_rows(manifest):
                if row["split"] != "test":
                    continue
                if sha256(row["image"]) != row["rgb_sha256"]:
                    raise ValueError("Test RGB changed during evaluation")
                image = cv2.imdecode(np.fromfile(row["image"], np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Test RGB failed to decode")
                # No mask, label, target instance, or scene actor is passed here.
                for name, detector in detectors.items():
                    before = time.perf_counter_ns()
                    error_type, status = None, "RETURNED"
                    try:
                        detection = detector.detect(image)
                        if name == "original_COCO_sports_ball":
                            detection["candidates"] = [candidate for candidate in detection["candidates"]
                                if candidate["source_class_name"] == "sports ball"]
                            detection["candidate_count"] = len(detection["candidates"])
                    except Exception as error:
                        error_type, status = type(error).__name__, "ERROR"
                        detection = {"candidates": [], "candidate_count": 0}
                    elapsed_ns = time.perf_counter_ns()-before
                    height, width = image.shape[:2]
                    truth = validate_label_file(row, width, height)
                    frame = {"model": name, "episode_id": row["episode_id"],
                        "frame_id": row["frame_id"], "case_id": row["case_id"],
                        "image": row["image"], "image_sha256": row["rgb_sha256"],
                        "label_sha256": row["label_sha256"], "status": status,
                        "error_type": error_type, "elapsed_ns": elapsed_ns,
                        "detection": detection, "truth_boxes_evaluation_only": truth,
                        "score": match_boxes([candidate["bbox_xyxy_pixels"] for candidate in detection["candidates"]], truth, .5)}
                    report["frames"].append(frame)
                    save_json(output/"report.json", report)
            # mAP is a separate low-score sweep of the same frozen checkpoint.
            # An 80-class COCO model is never validated against class-zero labels.
            map_data = write_data_yaml(output, manifest, include_test=True)
            map_args = {"data": str(map_data), "split": "test", "imgsz": 640,
                "batch": 8, "workers": 0, "device": "cpu", "half": False,
                "conf": .001, "iou": .5, "max_det": 300, "plots": False,
                "save_json": False, "save_txt": True, "save_conf": True,
                "project": str(output/"map"), "name": "specialist_test_once",
                "exist_ok": False, "verbose": False}
            report["specialist_map_arguments"] = map_args
            save_json(output/"report.json", report)
            before = time.perf_counter_ns()
            metrics = YOLO(str(best)).val(**map_args)
            map_report = {"status": "PASS", "elapsed_ns": time.perf_counter_ns()-before,
                "model_weights_sha256": sha256(best), "arguments": map_args,
                "results": {key: float(value) for key, value in metrics.results_dict.items()},
                "library_speed_ms": {key: float(value) for key, value in metrics.speed.items()},
                "baseline_map": None, "baseline_map_reason": "80-class labels not mapped to one-class test labels",
                "predictions_folder": str((output/"map"/"specialist_test_once"/"labels").resolve())}
            save_json(output/"specialist_map.json", map_report)
            report["specialist_map"] = map_report
        report["summary_by_model"] = {}
        report["summary_by_episode_and_model"] = {}
        for name in detectors:
            rows = [row for row in report["frames"] if row["model"] == name]
            report["summary_by_model"][name] = aggregate_frames(rows)
            groups = defaultdict(list)
            for row in rows:
                groups[row["episode_id"]].append(row)
            report["summary_by_episode_and_model"][name] = {
                episode: aggregate_frames(group) for episode, group in groups.items()}
        report["fixed_selection_visuals"] = comparison_visuals(output, manifest, report["frames"])
        validate_split_files(manifest, {"test"})
        if sha256(manifest_path) != report["dataset_manifest_sha256"] or sha256(best) != receipt["best_weights_sha256"]:
            raise ValueError("Data or selected checkpoint changed during evaluation")
        report.update(status="PASS" if all(row["status"] == "RETURNED" for row in report["frames"]) else "COMPLETED_WITH_DETECTOR_ERRORS",
            completed_utc=utc_now(), independent_test_episodes=len({row["episode_id"] for row in checked}),
            unique_test_frames=len(checked), detection_calls=len(report["frames"]),
            limits=plan["limits"]+[
                "Visible bounding boxes only, no designated-target identity score",
                "Two independent test sessions with nested frames; no cross-map or material generalization",
                "Test inspected once after validation selection; no post-test training or threshold changes"])
    except Exception as error:
        previous_status = report["status"]
        report.update(status="FAIL", completed_utc=utc_now())
        report["failure_attempts"].append({"stage": previous_status,
            "error_type": type(error).__name__, "exception_text_retained": False})
        report["elapsed_ns"] = time.perf_counter_ns()-start_ns
        save_json(output/"report.json", report)
        raise
    report["elapsed_ns"] = time.perf_counter_ns()-start_ns
    save_json(output/"report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_manifest, args.training_receipt, args.output)
    print(json.dumps({"status": result["status"], "report": str((args.output/"report.json").resolve())}))
