"""Run a pretrained YOLO baseline on the reviewed airborne image set."""
import argparse
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
APPLICATION_CLASS_MAP = {"boat": "vessel", "sports ball": "balloon"}
DEMO_THRESHOLD = 0.25


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def clean_ground_truth(dataset_root):
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "REVIEWED_CLEAN_SEED_NOT_TRAINING_SPLIT":
        raise ValueError("Expected the reviewed clean seed manifest")
    result = {}
    for item in manifest["images"]:
        result[Path(item["image"]).name] = [
            {
                "class_name": annotation["class_name"],
                "bbox_xyxy_pixels": annotation["bbox_xyxy_pixels"],
            }
            for annotation in item["annotations"]
        ]
    return result


def evaluate(predictions, ground_truth, threshold=0.5):
    totals = Counter()
    by_class = defaultdict(Counter)
    details = []
    for image_name, truth in sorted(ground_truth.items()):
        candidates = [
            item
            for item in predictions.get(image_name, [])
            if item["application_candidate_class"] is not None
        ]
        used = set()
        image_matches = []
        for gt in truth:
            options = [
                (iou(gt["bbox_xyxy_pixels"], pred["bbox_xyxy_pixels"]), index, pred)
                for index, pred in enumerate(candidates)
                if index not in used and pred["application_candidate_class"] == gt["class_name"]
            ]
            best = max(options, default=(0.0, None, None), key=lambda item: item[0])
            if best[0] >= threshold:
                used.add(best[1])
                totals["tp"] += 1
                by_class[gt["class_name"]]["tp"] += 1
                image_matches.append(
                    {
                        "ground_truth_class": gt["class_name"],
                        "matched": True,
                        "iou": best[0],
                        "prediction": best[2],
                    }
                )
            else:
                totals["fn"] += 1
                by_class[gt["class_name"]]["fn"] += 1
                image_matches.append(
                    {"ground_truth_class": gt["class_name"], "matched": False, "best_iou": best[0]}
                )
        for index, pred in enumerate(candidates):
            if index not in used:
                totals["fp"] += 1
                by_class[pred["application_candidate_class"]]["fp"] += 1
        details.append(
            {
                "image": image_name,
                "ground_truth_count": len(truth),
                "candidate_prediction_count": len(candidates),
                "matches": image_matches,
            }
        )
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {
        "scope": "descriptive smoke evaluation on three images from one reviewed collection session",
        "not_a_model_performance_claim": True,
        "iou_threshold": threshold,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "by_class": {name: dict(counts) for name, counts in sorted(by_class.items())},
        "details": details,
    }


def collection_waypoints(collection_root):
    report_path = collection_root / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        waypoints = [item["waypoint_id"] for item in report.get("waypoints", [])]
        if waypoints:
            return waypoints
    return ["center_mid", "left_mid", "right_mid", "center_high", "center_low"]


def collect_images(collection_root, waypoints):
    images = []
    for waypoint in waypoints:
        for vehicle in ("DroneA", "DroneB"):
            path = collection_root / waypoint / "cameras" / f"{vehicle}_00_rgb.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(path)
    return images


def make_contact_sheet(annotated_dir, output, suffix, title, waypoints):
    rows = []
    for waypoint in waypoints:
        panels = []
        for vehicle in ("DroneA", "DroneB"):
            path = annotated_dir / f"{waypoint}_{vehicle}_{suffix}.jpg"
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Cannot read annotated image: {path}")
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (390, 38), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{waypoint} | {vehicle} | {title}",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            panels.append(frame)
        rows.append(np.hstack(panels))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        type=Path,
        default=ROOT / "logs" / "air_multiview_20260916_152557_316743",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "air_multiview_seed_20260916",
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="run inference without comparing against the reviewed seed labels",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this baseline run")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Download official weights first: {args.weights}")
    waypoints = collection_waypoints(args.collection.resolve())
    images = collect_images(args.collection.resolve(), waypoints)
    ground_truth = None if args.no_evaluation else clean_ground_truth(args.dataset.resolve())
    output = ROOT / "logs" / datetime.now().strftime("yolo_baseline_%Y%m%d_%H%M%S_%f")
    annotated_dir = output / "annotated"
    demo_dir = output / "demo_mapped_025"
    annotated_dir.mkdir(parents=True)
    demo_dir.mkdir()

    model = YOLO(str(args.weights.resolve()))
    # Warm-up is separated from the measured batch.
    model.predict(
        source=str(images[0]),
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=0.5,
        device=0,
        verbose=False,
    )
    started = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=0.5,
        device=0,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    prediction_index = {}
    records = []
    speed_rows = []
    for source_path, result in zip(images, results):
        waypoint = source_path.parents[1].name
        vehicle = source_path.stem.split("_")[0]
        canonical_name = f"{waypoint}_{vehicle}.png"
        boxes = []
        if result.boxes is not None:
            for xyxy, class_id, confidence in zip(
                result.boxes.xyxy.detach().cpu().numpy(),
                result.boxes.cls.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy(),
            ):
                class_id = int(class_id)
                class_name = str(result.names[class_id])
                boxes.append(
                    {
                        "coco_class_id": class_id,
                        "coco_class_name": class_name,
                        "confidence": float(confidence),
                        "bbox_xyxy_pixels": [float(value) for value in xyxy],
                        "application_candidate_class": APPLICATION_CLASS_MAP.get(class_name),
                    }
                )
        prediction_index[canonical_name] = boxes
        speed = {key: float(value) for key, value in result.speed.items()}
        speed_rows.append(speed)
        record = {
            "image": str(source_path),
            "canonical_name": canonical_name,
            "waypoint_id": waypoint,
            "vehicle": vehicle,
            "model": args.weights.name,
            "imgsz": args.imgsz,
            "confidence_threshold": args.confidence,
            "speed_ms": speed,
            "predictions": boxes,
        }
        records.append(record)
        annotated = result.plot()
        cv2.imwrite(str(annotated_dir / f"{waypoint}_{vehicle}_predictions.jpg"), annotated)
        demo = cv2.imread(str(source_path))
        for item in boxes:
            mapped = item["application_candidate_class"]
            if mapped is None or item["confidence"] < DEMO_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in item["bbox_xyxy_pixels"]]
            color = (0, 255, 0) if mapped == "vessel" else (255, 255, 0)
            cv2.rectangle(demo, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                demo,
                f"{mapped} {item['confidence']:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )
        cv2.imwrite(str(demo_dir / f"{waypoint}_{vehicle}_demo.jpg"), demo)

    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    mean_speed = {
        key: float(np.mean([row[key] for row in speed_rows])) for key in speed_rows[0]
    }
    make_contact_sheet(
        annotated_dir,
        output / "prediction_contact_sheet.jpg",
        "predictions",
        "YOLO11n RAW 0.10",
        waypoints,
    )
    make_contact_sheet(
        demo_dir,
        output / "demo_mapped_025_contact_sheet.jpg",
        "demo",
        "MAPPED 0.25",
        waypoints,
    )
    threshold_sweep = {}
    evaluation = None
    if ground_truth is not None:
        for threshold in (0.10, 0.25, 0.50):
            filtered = {
                image: [item for item in items if item["confidence"] >= threshold]
                for image, items in prediction_index.items()
            }
            item = evaluate(filtered, ground_truth)
            item["all_image_candidate_prediction_count"] = sum(
                prediction["application_candidate_class"] is not None
                for predictions in filtered.values()
                for prediction in predictions
            )
            item["all_image_unmapped_prediction_count"] = sum(
                prediction["application_candidate_class"] is None
                for predictions in filtered.values()
                for prediction in predictions
            )
            threshold_sweep[f"{threshold:.2f}"] = item
        evaluation = threshold_sweep[f"{args.confidence:.2f}"]
    report = {
        "status": "PASS",
        "mode": "official pretrained COCO baseline; no project fine-tuning",
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights.resolve()),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "image_count": len(images),
        "imgsz": args.imgsz,
        "confidence_threshold": args.confidence,
        "evaluation_enabled": ground_truth is not None,
        "application_class_map": APPLICATION_CLASS_MAP,
        "wall_time_ms": elapsed_ms,
        "wall_time_per_image_ms": elapsed_ms / len(images),
        "mean_ultralytics_speed_ms": mean_speed,
        "raw_prediction_count": sum(len(row["predictions"]) for row in records),
        "candidate_prediction_count": sum(
            item["application_candidate_class"] is not None
            for row in records
            for item in row["predictions"]
        ),
        "smoke_evaluation": evaluation,
        "threshold_sweep": threshold_sweep,
        "provisional_demo_threshold": DEMO_THRESHOLD,
        "provisional_demo_threshold_scope": (
            "display threshold inherited from the prior three-image smoke run; "
            "not validated on this collection"
            if ground_truth is None
            else "chosen only for the saved-image demo because it retains all nine clean-seed "
            "matches while removing the three unmapped predictions; not a validated operating point"
        ),
        "prediction_contact_sheet": "prediction_contact_sheet.jpg",
        "mapped_demo_contact_sheet": "demo_mapped_025_contact_sheet.jpg",
        "limitations": [
            "COCO boat and sports ball are only candidate mappings for vessel and balloon.",
            "No project-specific training or validation was performed.",
            "The three-image clean seed is one session and cannot support a performance claim.",
            "This collection has no reviewed ground truth; inference output is not scored."
            if ground_truth is None
            else "The reviewed seed evaluation is descriptive only.",
            "Annotated images are previews; predictions.jsonl is the machine-readable output.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "logs" / "yolo_baseline_latest.json").write_text(
        json.dumps({"output": str(output)}, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "raw_prediction_count": report["raw_prediction_count"],
                "candidate_prediction_count": report["candidate_prediction_count"],
                "wall_time_per_image_ms": report["wall_time_per_image_ms"],
                "smoke_evaluation": None
                if evaluation is None
                else {
                    key: evaluation[key]
                    for key in ("tp", "fp", "fn", "precision", "recall")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
