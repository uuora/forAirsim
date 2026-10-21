"""Validate provenance-aware maritime visible boxes before scoring."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


BOX_FIELDS = ("target_visible_bbox_xyxy", "distractor_visible_bbox_xyxy")
VISIBILITY_FIELDS = ("target_visibility", "distractor_visibility")
VISIBILITY = {"VISIBLE", "PARTIAL", "NOT_VISIBLE", "UNCERTAIN"}


def parse_box(value, image_size=(1280, 720)):
    box = json.loads(value)
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Visible box must be a four-number JSON array")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in box):
        raise ValueError("Visible box coordinates must be numbers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image_size[0] and 0 <= y1 < y2 <= image_size[1]):
        raise ValueError("Visible box is empty or outside the image")
    return box


def validate(manifest_path, allow_ai_reviewed=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue_path = Path(manifest["review_queue"])
    with queue_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != manifest["frame_count"]:
        raise ValueError("Review queue row count differs from collection manifest")
    seen = set()
    for row in rows:
        key = (row["session_index"], row["profile"], row["frame"])
        if key in seen:
            raise ValueError(f"Duplicate review row: {key}")
        seen.add(key)
        image_path = Path(row["image_path"])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed after queue creation: {image_path}")
        status = row["review_status"]
        if status == "PENDING_HUMAN":
            if any(row[field].strip() for field in (*BOX_FIELDS, *VISIBILITY_FIELDS,
                                                   "reviewer", "reviewed_at")):
                raise ValueError(f"Pending row contains a partial review: {key}")
        elif status in ("VERIFIED", "AI_REVIEWED"):
            if not row["reviewer"].strip() or not row["reviewed_at"].strip():
                raise ValueError(f"Verified row needs reviewer and time: {key}")
            if status == "AI_REVIEWED" and "AI" not in row["reviewer"].upper():
                raise ValueError(f"AI review must be identified in reviewer metadata: {key}")
            if status == "VERIFIED" and row["reviewer"].strip().lower().startswith("codex ai"):
                raise ValueError(f"AI reviewer cannot certify human verification: {key}")
            for box_field, visibility_field in zip(BOX_FIELDS, VISIBILITY_FIELDS):
                visibility = row[visibility_field]
                if visibility not in VISIBILITY:
                    raise ValueError(f"Invalid visibility for {key}: {visibility}")
                if visibility in ("VISIBLE", "PARTIAL"):
                    if not row[box_field].strip():
                        raise ValueError(f"Visible object lacks a reviewed box: {key}")
                    parse_box(row[box_field])
                elif row[box_field].strip():
                    raise ValueError(f"Absent or uncertain object has a box: {key}")
        else:
            raise ValueError(f"Unknown review status for {key}: {status}")
    counts = Counter(row["review_status"] for row in rows)
    if counts["VERIFIED"] == len(rows):
        status = "READY_FOR_EVALUATION"
    elif allow_ai_reviewed and counts["VERIFIED"] + counts["AI_REVIEWED"] == len(rows):
        status = "READY_FOR_EXPLORATORY_EVALUATION"
    else:
        status = "PENDING_HUMAN_REVIEW"
    return {"status": status, "frames": len(rows),
            "verified": counts["VERIFIED"], "ai_reviewed": counts["AI_REVIEWED"],
            "pending": counts["PENDING_HUMAN"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-ai-reviewed", action="store_true",
                        help="Allow exploratory scoring; never certifies independent human review")
    args = parser.parse_args()
    result = validate(args.manifest, allow_ai_reviewed=args.allow_ai_reviewed)
    print(json.dumps(result, ensure_ascii=False))
