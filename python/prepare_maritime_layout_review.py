"""Build a session-grouped review queue without treating simulator boxes as labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


FIELDS = [
    "session_index", "engine_process_id", "profile", "frame", "image_path",
    "image_sha256", "task_description", "expected_target_side",
    "projected_target_box_reference", "projected_distractor_box_reference",
    "target_visible_bbox_xyxy", "distractor_visible_bbox_xyxy",
    "target_visibility", "distractor_visibility", "reviewer", "reviewed_at",
    "review_status",
]


def prepare(report_paths, output_dir):
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "collection_manifest.json"
    queue_path = output_dir / "review_queue.csv"
    if manifest_path.exists() or queue_path.exists():
        raise FileExistsError("Review outputs already exist; preserve any human edits")
    reports = []
    rows = []
    processes = set()
    config_hashes = set()
    for report_path in report_paths:
        report_path = Path(report_path).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        meta = json.loads((report_path.parent / "session_meta.json").read_text(
            encoding="utf-8-sig"))
        if (report["status"] != "PASS" or not report["paused"]
                or not report["scene_restored_to_baseline"]
                or not report["landing_evidence"]["landed_flag_after_settle"]):
            raise ValueError(f"Unsafe or incomplete session: {report_path}")
        if meta["capture_status"] != "PASS" or meta["config_sha256"] != report["config_sha256"]:
            raise ValueError(f"Session metadata mismatch: {report_path}")
        process_key = (meta["engine_process_id"], meta["engine_started"])
        if process_key in processes:
            raise ValueError("Two claimed sessions share one Unreal process")
        processes.add(process_key)
        config_hashes.add(report["config_sha256"])
        target_id = report["config"]["target_id"]
        for condition in report["conditions"]:
            profile = condition["profile"]
            task = report["config"]["tasks"][profile]
            for frame in condition["frames"]:
                if (frame["task_description"] != task["description"]
                        or frame["expected_target_side"] != task["expected_target_side"]
                        or frame["task_relation_verified"] is not True):
                    raise ValueError(f"Preregistered task relation mismatch: {frame['frame']}")
                image_path = report_path.parent / frame["frame"]
                with Image.open(image_path) as image:
                    if image.size != (1280, 720):
                        raise ValueError(f"Unexpected image dimensions: {image_path}")
                truth = frame["simulator_boxes_xyxy"]
                if target_id not in truth or "balloon_003" not in truth:
                    raise ValueError(f"Missing projected actor reference: {frame['frame']}")
                rows.append({
                    "session_index": meta["session_index"],
                    "engine_process_id": meta["engine_process_id"],
                    "profile": profile,
                    "frame": frame["frame"],
                    "image_path": str(image_path),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "task_description": frame["task_description"],
                    "expected_target_side": frame["expected_target_side"],
                    "projected_target_box_reference": json.dumps(truth[target_id]),
                    "projected_distractor_box_reference": json.dumps(truth["balloon_003"]),
                    "target_visible_bbox_xyxy": "",
                    "distractor_visible_bbox_xyxy": "",
                    "target_visibility": "",
                    "distractor_visibility": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "review_status": "PENDING_HUMAN",
                })
        reports.append({"session_index": meta["session_index"],
                        "engine_process_id": meta["engine_process_id"],
                        "engine_started": meta["engine_started"],
                        "report_path": str(report_path),
                        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        "frame_count": sum(len(c["frames"]) for c in report["conditions"])})
    if len(config_hashes) != 1:
        raise ValueError("Session configurations differ")
    if len({report["session_index"] for report in reports}) != len(reports):
        raise ValueError("Duplicate session index")
    output_dir.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "PENDING_HUMAN_REVIEW",
        "session_count": len(reports),
        "frame_count": len(rows),
        "config_sha256": next(iter(config_hashes)),
        "sessions": sorted(reports, key=lambda item: item["session_index"]),
        "review_queue": str(queue_path),
        "reference_box_warning": "AirSim projected actor boxes are not visible-pixel annotations",
        "partition_policy": "Keep all frames from one Unreal process in one analysis group; no random frame split",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.reports, args.output_dir)
    print(json.dumps({"status": result["status"], "sessions": result["session_count"],
                      "frames": result["frame_count"], "queue": result["review_queue"]},
                     ensure_ascii=False))
