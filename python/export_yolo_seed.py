"""Export reviewed multi-view boxes as a provenance-preserving YOLO seed set."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_map():
    registry = json.loads((ROOT / "configs" / "maritime_targets.json").read_text(encoding="utf-8"))
    return {item["name"]: int(item["id"]) for item in registry["categories"]}


def yolo_box(row, width, height):
    x1, y1, x2, y2 = [float(row[name]) for name in ("x1", "y1", "x2", "y2")]
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    values = (cx, cy, box_width, box_height)
    if not all(0 < value < 1 for value in values):
        raise ValueError(f"Invalid normalized YOLO box: {values}")
    return values


def export(source, output):
    annotations_path = source / "final_box_annotations.csv"
    summary = json.loads((source / "box_review_summary.json").read_text(encoding="utf-8"))
    expected_annotations = int(summary.get("included_annotations", 0))
    if summary.get("status") != "PASS" or expected_annotations <= 0:
        raise ValueError("Reviewed annotation summary is not a passing non-empty seed")
    categories = class_map()
    if categories != {"vessel": 0, "balloon": 1}:
        raise ValueError(f"Unexpected category map: {categories}")
    rows = list(csv.DictReader(annotations_path.open(encoding="utf-8-sig")))
    if len(rows) != expected_annotations:
        raise ValueError(
            f"Expected {expected_annotations} final annotations from review summary, found {len(rows)}"
        )
    by_image = defaultdict(list)
    for row in rows:
        if row["class_name"] not in categories:
            raise ValueError(f"Unknown class in reviewed annotations: {row['class_name']}")
        by_image[row["rgb_image"]].append(row)
    if not by_image:
        raise ValueError("Reviewed annotations contain no images")
    review_rows = list(
        csv.DictReader((source / "manual_box_review_reviewed.csv").open(encoding="utf-8-sig"))
    )
    excluded_images = defaultdict(list)
    for row in review_rows:
        if row["review_decision"] in {"REJECT_CROPPED", "REJECT_AMBIGUOUS"}:
            excluded_images[row["rgb_image"]].append(
                {
                    "target_id": row["target_id"],
                    "review_decision": row["review_decision"],
                    "review_notes": row["review_notes"],
                }
            )
    eligible_images = sorted(set(by_image) - set(excluded_images))
    expected_eligible = len(by_image) - len(excluded_images)
    if len(eligible_images) != expected_eligible or not eligible_images:
        raise ValueError(
            f"Expected {expected_eligible} clean images after crop exclusion, "
            f"found {len(eligible_images)}"
        )

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_images = []
    for relative_image in eligible_images:
        image_rows = by_image[relative_image]
        source_image = source / Path(relative_image)
        frame = cv2.imread(str(source_image))
        if frame is None:
            raise ValueError(f"Cannot read source image: {source_image}")
        height, width = frame.shape[:2]
        waypoint = image_rows[0]["waypoint_id"]
        vehicle = image_rows[0]["vehicle"]
        stem = f"{waypoint}_{vehicle}"
        image_output = images_dir / f"{stem}.png"
        label_output = labels_dir / f"{stem}.txt"
        shutil.copy2(source_image, image_output)
        label_lines = []
        manifest_annotations = []
        for row in sorted(image_rows, key=lambda item: (categories[item["class_name"]], item["target_id"])):
            values = yolo_box(row, width, height)
            class_id = categories[row["class_name"]]
            label_lines.append(
                f"{class_id} " + " ".join(f"{value:.8f}" for value in values)
            )
            manifest_annotations.append(
                {
                    "class_id": class_id,
                    "class_name": row["class_name"],
                    "target_id": row["target_id"],
                    "bbox_xyxy_pixels": [
                        float(row[name]) for name in ("x1", "y1", "x2", "y2")
                    ],
                    "bbox_yolo_normalized": list(values),
                    "annotation_source": row["annotation_source"],
                }
            )
        label_output.write_text("\n".join(label_lines) + "\n", encoding="ascii")
        manifest_images.append(
            {
                "image": str(image_output.relative_to(output)).replace("\\", "/"),
                "label": str(label_output.relative_to(output)).replace("\\", "/"),
                "source_image": str(source_image),
                "source_sha256": sha256(source_image),
                "export_sha256": sha256(image_output),
                "label_sha256": sha256(label_output),
                "width": width,
                "height": height,
                "waypoint_id": waypoint,
                "vehicle": vehicle,
                "annotations": manifest_annotations,
            }
        )
    if any(item["source_sha256"] != item["export_sha256"] for item in manifest_images):
        raise ValueError("An exported image differs from its source")

    manifest = {
        "status": "REVIEWED_CLEAN_SEED_NOT_TRAINING_SPLIT",
        "source_collection": str(source),
        "source_collection_report": str(source / "report.json"),
        "source_review_summary": str(source / "box_review_summary.json"),
        "class_map": categories,
        "source_reviewed_image_count": len(by_image),
        "source_reviewed_annotation_count": len(rows),
        "image_count": len(manifest_images),
        "annotation_count": sum(len(item["annotations"]) for item in manifest_images),
        "excluded_image_count": len(excluded_images),
        "excluded_images": [
            {"source_image": image, "reasons": reasons}
            for image, reasons in sorted(excluded_images.items())
        ],
        "review_provenance": "AI-assisted full-resolution visual review; not independent human verification",
        "split_policy": "No train/validation/test split: all images belong to one collection session.",
        "training_ready": False,
        "limitations": [
            "One simulator scene and one collection session cannot support model evaluation.",
            "DroneA and DroneB images are sequential RPC captures; consult the source report for timing.",
            "Images containing a boundary-cropped or ambiguous target are excluded to avoid false-negative supervision.",
            "Rejected absent targets do not exclude an image because the reviewed target is not visible.",
            "A human should confirm the review sheet and final overlay before training use.",
        ],
        "images": manifest_images,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "classes.json").write_text(
        json.dumps(
            {"names": {str(value): key for key, value in sorted(categories.items(), key=lambda item: item[1])}},
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(source / "final_annotation_contact_sheet.jpg", output / "review_contact_sheet.jpg")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = export(args.source.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output.resolve()),
                "image_count": manifest["image_count"],
                "annotation_count": manifest["annotation_count"],
                "training_ready": manifest["training_ready"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
