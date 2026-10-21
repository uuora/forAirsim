"""Interactive human review of visible balloon boxes in saved maritime images.

Keys: 1 target, 2 distractor, drag a rectangle, p partial, x not visible,
u uncertain, c clear active box, s save reviewed row, n skip, q quit.
"""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from validate_maritime_layout_review import parse_box, validate


WINDOW = "Maritime visible-box review"
OBJECTS = ("target", "distractor")
COLORS = {"target": (0, 255, 0), "distractor": (255, 255, 0)}


def save_rows(queue_path, fields, rows):
    temp = queue_path.with_suffix(queue_path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, queue_path)


def review(queue_path, reviewer):
    import cv2

    queue_path = Path(queue_path).resolve()
    with queue_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    if not reviewer.strip():
        raise ValueError("A human reviewer name is required")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 800)
    try:
        for row in rows:
            if row["review_status"] == "VERIFIED":
                continue
            if row["review_status"] not in ("PENDING_HUMAN", "AI_REVIEWED"):
                raise ValueError(f"Unknown review status: {row['review_status']}")
            image_path = Path(row["image_path"])
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
                raise ValueError(f"Image changed since collection: {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (720, 1280):
                raise ValueError(f"Cannot display 1280x720 image: {image_path}")
            boxes = {name: None for name in OBJECTS}
            visibility = {name: "VISIBLE" for name in OBJECTS}
            state = {"active": "target", "start": None, "drag": None}

            def mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN:
                    state["start"] = (x, y)
                    state["drag"] = (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
                    state["drag"] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
                    x0, y0 = state["start"]
                    box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
                    try:
                        parse_box(json.dumps(box))
                    except ValueError:
                        pass
                    else:
                        boxes[state["active"]] = box
                        if visibility[state["active"]] in ("NOT_VISIBLE", "UNCERTAIN"):
                            visibility[state["active"]] = "VISIBLE"
                    state["start"] = None
                    state["drag"] = None

            cv2.setMouseCallback(WINDOW, mouse)
            while True:
                canvas = image.copy()
                for name in OBJECTS:
                    box = boxes[name]
                    if box is not None:
                        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]),
                                      COLORS[name], 2)
                        cv2.putText(canvas, name, (box[0], max(20, box[1] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2)
                if state["start"] is not None and state["drag"] is not None:
                    cv2.rectangle(canvas, state["start"], state["drag"],
                                  COLORS[state["active"]], 1)
                lines = [
                    f"S{row['session_index']} {row['profile']} {row['frame']}",
                    f"TASK: {row['task_description']}",
                    f"ACTIVE: {state['active']} | target {visibility['target']} | distractor {visibility['distractor']}",
                    "1 target | 2 distractor | drag box | p partial | x absent | u uncertain",
                    "c clear | s save verified | n skip | q quit",
                ]
                cv2.rectangle(canvas, (0, 0), (1279, 118), (30, 30, 30), -1)
                for line_index, line in enumerate(lines):
                    cv2.putText(canvas, line, (8, 20 + 23 * line_index),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("1"):
                    state["active"] = "target"
                elif key == ord("2"):
                    state["active"] = "distractor"
                elif key == ord("p"):
                    if boxes[state["active"]] is not None:
                        visibility[state["active"]] = "PARTIAL"
                elif key == ord("x"):
                    boxes[state["active"]] = None
                    visibility[state["active"]] = "NOT_VISIBLE"
                elif key == ord("u"):
                    boxes[state["active"]] = None
                    visibility[state["active"]] = "UNCERTAIN"
                elif key == ord("c"):
                    boxes[state["active"]] = None
                    visibility[state["active"]] = "VISIBLE"
                elif key == ord("n"):
                    break
                elif key == ord("q"):
                    return validate(queue_path.parent / "collection_manifest.json")
                elif key == ord("s"):
                    for name in OBJECTS:
                        if visibility[name] in ("VISIBLE", "PARTIAL") and boxes[name] is None:
                            print(f"Cannot save: {name} needs a visible box")
                            break
                    else:
                        for name in OBJECTS:
                            row[f"{name}_visible_bbox_xyxy"] = (json.dumps(boxes[name])
                                if boxes[name] is not None else "")
                            row[f"{name}_visibility"] = visibility[name]
                        row["reviewer"] = reviewer.strip()
                        row["reviewed_at"] = datetime.now().astimezone().isoformat()
                        row["review_status"] = "VERIFIED"
                        save_rows(queue_path, fields, rows)
                        break
    finally:
        cv2.destroyWindow(WINDOW)
    return validate(queue_path.parent / "collection_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.queue, args.reviewer), ensure_ascii=False))
