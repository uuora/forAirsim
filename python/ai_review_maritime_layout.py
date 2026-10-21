"""Prepare image-derived box suggestions and visual QA sheets for an AI review.

The suggestions are not independent annotations: they come from the color
detector that is also evaluated in this project. Commit only after viewing
the generated sheets, and retain AI_REVIEWED provenance in the queue.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from balloon_detector import detect_balloon


def proposed_rows(queue_path):
    with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty review queue")
    for row in rows:
        if row["review_status"] not in ("PENDING_HUMAN", "AI_REVIEWED"):
            raise ValueError(f"Refusing to replace review status: {row['review_status']}")
        path = Path(row["image_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image hash changed: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable frame: {path}")
        candidates = sorted(detect_balloon(image)["candidates"],
                            key=lambda item: item["centre_px"][0])
        if len(candidates) != 2:
            raise ValueError(f"Expected two red objects in {path}, found {len(candidates)}")
        left, right = candidates
        target, distractor = (left, right) if row["expected_target_side"] == "left" else (right, left)
        if row["expected_target_side"] not in ("left", "right"):
            raise ValueError(f"Unknown expected side: {row['expected_target_side']}")
        for prefix, candidate in (("target", target), ("distractor", distractor)):
            x, y, w, h = candidate["bbox_xywh"]
            row[f"{prefix}_visible_bbox_xyxy"] = json.dumps([x, y, x + w, y + h])
            row[f"{prefix}_visibility"] = "VISIBLE"
    return rows


def sheets(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted({row["session_index"] for row in rows}, key=int)
    outputs = []
    for session in sessions:
        group = [row for row in rows if row["session_index"] == session]
        tile_w, tile_h = 580, 230
        canvas = Image.new("RGB", (tile_w * 2, tile_h * ((len(group) + 1) // 2)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(group):
            image = Image.open(row["image_path"]).convert("RGB")
            # Fixed crop shows both red balloons and enough maritime context.
            crop = image.crop((800, 450, 1080, 590)).resize((560, 280))
            for label, color in (("target", "#00e000"), ("distractor", "#00c8ff")):
                x1, y1, x2, y2 = json.loads(row[f"{label}_visible_bbox_xyxy"])
                box = [(x1 - 800) * 2, (y1 - 450) * 2,
                       (x2 - 800) * 2, (y2 - 450) * 2]
                ImageDraw.Draw(crop).rectangle(box, outline=color, width=3)
            crop = crop.resize((560, 170))
            ox = index % 2 * tile_w + 10
            oy = index // 2 * tile_h + 25
            canvas.paste(crop, (ox, oy))
            draw.text((ox, oy - 20),
                      f"{row['profile']} / {row['frame']} / target {row['expected_target_side']}",
                      fill="black")
            draw.text((ox, oy + 175), "green = target; cyan = distractor", fill="black")
        path = output_dir / f"session_{session}_ai_review_draft.jpg"
        canvas.save(path, quality=92)
        outputs.append(str(path))
    return outputs


def commit(queue_path, rows):
    queue_path = Path(queue_path)
    with queue_path.open(newline="", encoding="utf-8-sig") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != len(rows) or any(
            a["image_sha256"] != b["image_sha256"] for a, b in zip(original, rows)):
        raise ValueError("Queue changed since draft; regenerate and inspect it")
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["reviewer"] = "Codex AI visual review; color-detector-assisted"
        row["reviewed_at"] = timestamp
        row["review_status"] = "AI_REVIEWED"
    temporary = queue_path.with_suffix(".ai-review-tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=original[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(queue_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--commit-reviewed", action="store_true")
    args = parser.parse_args()
    rows = proposed_rows(args.queue)
    if args.commit_reviewed:
        commit(args.queue, rows)
        print(json.dumps({"status": "AI_REVIEWED", "frames": len(rows)}))
    else:
        output = sheets(rows, args.queue.parent)
        print(json.dumps({"status": "DRAFT_NEEDS_VISUAL_INSPECTION",
                          "frames": len(rows), "sheets": output}))


if __name__ == "__main__":
    main()
