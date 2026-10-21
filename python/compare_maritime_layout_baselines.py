"""Exploratory instance-selection comparison on three held-together AirSim sessions.

AI-reviewed boxes were suggested by the color detector, so localization IoU
against them is not an independent color-detector score. No VLX score is made
unless a separate inference report contains real model outputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2

from balloon_detector import detect_balloon
from validate_maritime_layout_review import validate
from vlx_seek_probe import box_iou, red_region_proposals


def side_choice(boxes, side):
    if not boxes:
        return None
    key = lambda box: (box[0] + box[2]) / 2
    return (min if side == "left" else max)(boxes, key=key)


def identity(box, target, distractor):
    if box is None:
        return "NO_SELECTION"
    target_iou = box_iou(box, target)
    distractor_iou = box_iou(box, distractor)
    if max(target_iou, distractor_iou) < 0.1:
        return "OTHER"
    return "TARGET" if target_iou > distractor_iou else "DISTRACTOR"


def red_fraction(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [round(value) for value in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 100, 65), (12, 255, 255)) | cv2.inRange(
        hsv, (168, 100, 65), (179, 255, 255))
    return float(cv2.countNonZero(red)) / ((x2 - x1) * (y2 - y1))


def compare(manifest_path, proposal_paths, output_path):
    gate = validate(manifest_path, allow_ai_reviewed=True)
    if gate["status"] != "READY_FOR_EXPLORATORY_EVALUATION":
        raise ValueError(f"Requires reviewed exploratory labels: {gate}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    with Path(manifest["review_queue"]).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    sources = []
    expected_reports = {str(session["report_sha256"]): str(session["session_index"])
                        for session in manifest["sessions"]}
    for path in proposal_paths:
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "PASS" or data.get("vlx_executed"):
            raise ValueError(f"Expected completed YOLO-only report: {path}")
        session = expected_reports.get(data.get("source_report_sha256"))
        if session is None or session in {item["session_index"] for item in sources}:
            raise ValueError(f"Proposal source is unknown or duplicated: {path}")
        for entry in data["frames"]:
            key = (session, entry["profile"], entry["frame"])
            if key in indexed:
                raise ValueError(f"Duplicate proposal frame: {key}")
            indexed[key] = entry
        sources.append({"path": str(path), "session_index": session,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(indexed) != len(rows):
        raise ValueError("Proposal frame count differs from reviewed collection")
    results = []
    for row in rows:
        key = (row["session_index"], row["profile"], row["frame"])
        entry = indexed[key]
        if entry["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"Proposal/image hash mismatch: {key}")
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        target = json.loads(row["target_visible_bbox_xyxy"])
        distractor = json.loads(row["distractor_visible_bbox_xyxy"])
        side = row["expected_target_side"]
        yolo_boxes = [candidate["bbox_xyxy_pixels"] for candidate in entry["yolo_candidates"]
                      if candidate["class_name"] == "balloon"]
        red_yolo = [box for box in yolo_boxes if red_fraction(image, box) >= 0.2]
        color = detect_balloon(image)["candidates"]
        color_boxes = [[x, y, x + w, y + h]
                       for x, y, w, h in (candidate["bbox_xywh"] for candidate in color)]
        red_regions = red_region_proposals(image)
        selected = {
            "yolo_highest_confidence_balloon": yolo_boxes[0] if yolo_boxes else None,
            "yolo_side_all_balloon": side_choice(yolo_boxes, side),
            "yolo_red_filtered_side": side_choice(red_yolo, side),
            "color_largest_area": color_boxes[0] if color_boxes else None,
            "color_prompt_side": side_choice(color_boxes, side),
            "red_region_prompt_side": side_choice(red_regions, side),
        }
        results.append({
            "session_index": row["session_index"], "profile": row["profile"],
            "frame": row["frame"], "task_description": row["task_description"],
            "yolo_balloon_count": len(yolo_boxes), "yolo_red_count": len(red_yolo),
            "color_count": len(color_boxes), "red_region_count": len(red_regions),
            "yolo_target_iou_ge_0_5": any(box_iou(box, target) >= 0.5 for box in yolo_boxes),
            "yolo_distractor_iou_ge_0_5": any(box_iou(box, distractor) >= 0.5 for box in yolo_boxes),
            "identity": {name: identity(box, target, distractor)
                         for name, box in selected.items()},
            "selected_boxes": selected,
            "yolo_wall_ms": entry["yolo_wall_ms"],
        })
    summary = {}
    for group_name, members in ((f"session_{session}", [r for r in results if r["session_index"] == session])
                                for session in sorted({r["session_index"] for r in results})):
        summary[group_name] = {
            "frames": len(members),
            "yolo_target_proposal_recall_iou_0_5": sum(r["yolo_target_iou_ge_0_5"] for r in members),
            "yolo_distractor_proposal_recall_iou_0_5": sum(r["yolo_distractor_iou_ge_0_5"] for r in members),
            "selection_target_counts": {name: sum(r["identity"][name] == "TARGET" for r in members)
                                        for name in results[0]["identity"]},
        }
    by_profile = {}
    for profile in sorted({r["profile"] for r in results}):
        members = [r for r in results if r["profile"] == profile]
        by_profile[profile] = {"frames": len(members),
                               "selection_target_counts": {
                                   name: sum(r["identity"][name] == "TARGET" for r in members)
                                   for name in results[0]["identity"]}}
    output = {
        "status": "PASS_EXPLORATORY", "frame_count": len(results),
        "session_count": len(summary), "vlx_executed": False,
        "label_provenance": "Codex AI visual review, color-detector-assisted; no independent human labels",
        "limitation": "Color detector supplied label suggestions; color localization IoU is circular. "
                      "Frames within a session are correlated, and sessions use the same scripted scene.",
        "review_manifest": str(Path(manifest_path).resolve()),
        "review_queue_sha256": hashlib.sha256(Path(manifest["review_queue"]).read_bytes()).hexdigest(),
        "proposal_reports": sources, "summary_by_session": summary,
        "summary_by_profile": by_profile, "frames": results,
    }
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.manifest, args.proposals, args.output)
    print(json.dumps({"status": result["status"], "frames": result["frame_count"],
                      "by_session": result["summary_by_session"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
