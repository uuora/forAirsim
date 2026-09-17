"""Materialize visually reviewed boxes without changing collection evidence."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


VALID_DECISIONS = {
    "ACCEPT",
    "REDRAW",
    "REJECT_ABSENT",
    "REJECT_CROPPED",
    "REJECT_AMBIGUOUS",
}


def target_class(target_id):
    if target_id.startswith("vessel_"):
        return "vessel"
    if target_id.startswith("balloon_"):
        return "balloon"
    raise ValueError(f"Unknown target class: {target_id}")


def load_corrections(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["key"]: item for item in payload["corrections"]}


def as_box(row):
    return [float(row[name]) for name in ("suggested_x1", "suggested_y1", "suggested_x2", "suggested_y2")]


def validate_box(box, width, height, key):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid box for {key}: {box} outside {width}x{height}")
    if x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height:
        raise ValueError(f"Included box touches an image boundary for {key}: {box}")


def make_contact_sheet(root, preview_paths, output, waypoint_order):
    by_waypoint = defaultdict(dict)
    for waypoint, vehicle, path in preview_paths:
        by_waypoint[waypoint][vehicle] = path
    rows = []
    for waypoint in waypoint_order:
        panels = []
        for vehicle in ("DroneA", "DroneB"):
            frame = cv2.imread(str(by_waypoint[waypoint][vehicle]))
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (410, 38), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{waypoint} | {vehicle} | FINAL REVIEW",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
        rows.append(np.hstack(panels))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def finalize(root):
    reviewed_path = root / "manual_box_review_reviewed.csv"
    corrections_path = root / "manual_redraws.json"
    corrections = load_corrections(corrections_path)
    rows = list(csv.DictReader(reviewed_path.open(encoding="utf-8-sig")))
    if len(rows) != 40:
        raise ValueError(f"Expected 40 review rows, found {len(rows)}")
    waypoint_order = list(dict.fromkeys(row["waypoint_id"] for row in rows))

    final_rows = []
    used_corrections = set()
    decisions = Counter()
    image_rows = defaultdict(list)
    for row in rows:
        decision = row["review_decision"]
        if decision not in VALID_DECISIONS or not row["review_notes"]:
            raise ValueError(f"Incomplete review row: {row}")
        decisions[decision] += 1
        if decision.startswith("REJECT_"):
            continue
        key = f"{row['waypoint_id']}|{row['vehicle']}|{row['target_id']}"
        if decision == "REDRAW":
            if key not in corrections:
                raise ValueError(f"Missing correction for {key}")
            correction = corrections[key]
            box = [float(value) for value in correction["bbox_xyxy_pixels"]]
            source = "manual_redraw"
            used_corrections.add(key)
        else:
            box = as_box(row)
            source = "accepted_simulator_suggestion"
        image_path = root / Path(row["rgb_image"])
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = frame.shape[:2]
        validate_box(box, width, height, key)
        item = {
            "waypoint_id": row["waypoint_id"],
            "vehicle": row["vehicle"],
            "rgb_image": row["rgb_image"],
            "class_name": target_class(row["target_id"]),
            "target_id": row["target_id"],
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
            "annotation_source": source,
            "review_decision": decision,
            "review_notes": row["review_notes"],
        }
        final_rows.append(item)
        image_rows[row["rgb_image"]].append(item)

    unused = set(corrections) - used_corrections
    if unused:
        raise ValueError(f"Unused redraw corrections: {sorted(unused)}")

    final_path = root / "final_box_annotations.csv"
    with final_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)

    preview_dir = root / "final_review_previews"
    preview_dir.mkdir(exist_ok=True)
    preview_paths = []
    for rgb_image in sorted({row["rgb_image"] for row in rows}):
        frame = cv2.imread(str(root / Path(rgb_image)))
        for item in image_rows.get(rgb_image, []):
            box = [int(round(item[name])) for name in ("x1", "y1", "x2", "y2")]
            color = (0, 255, 0) if item["annotation_source"].startswith("accepted") else (255, 255, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(
                frame,
                item["class_name"],
                (box[0], max(20, box[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
            )
        waypoint = Path(rgb_image).parts[0]
        vehicle = Path(rgb_image).name.split("_")[0]
        output = preview_dir / f"{waypoint}_{vehicle}_final.jpg"
        cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        preview_paths.append((waypoint, vehicle, output))

    contact_sheet = root / "final_annotation_contact_sheet.jpg"
    make_contact_sheet(root, preview_paths, contact_sheet, waypoint_order)
    summary = {
        "status": "PASS",
        "review_scope": "AI-assisted full-resolution visual review; not independent human verification",
        "source_review_csv": reviewed_path.name,
        "source_corrections": corrections_path.name,
        "review_rows": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "included_annotations": len(final_rows),
        "included_by_class": dict(sorted(Counter(row["class_name"] for row in final_rows).items())),
        "included_by_source": dict(
            sorted(Counter(row["annotation_source"] for row in final_rows).items())
        ),
        "excluded_annotations": len(rows) - len(final_rows),
        "final_annotations": final_path.name,
        "contact_sheet": contact_sheet.name,
        "training_ready": False,
        "next_gate": "human confirmation of the reviewed boxes before YOLO conversion",
    }
    (root / "box_review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
