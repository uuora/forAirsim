"""Offline timing and output-contract probe on unannotated single-target RGB.

This does not command AirSim, evaluate human-labelled accuracy, or enable an
online controller. A cloud image/text call and a YOLO call are separate methods.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from balloon_detector import detect_balloon
from setup_scene import ROOT


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_frames(sweep_path):
    sweep = load(sweep_path)
    if sweep["status"] != "PASS":
        raise ValueError("Timing probe requires a completed sweep; preserve partial runs separately")
    frames = []
    for entry in sweep["pilots"]:
        if digest(entry["manifest"]) != entry["manifest_sha256"]:
            raise ValueError("Pilot manifest changed after collection")
        pilot = load(entry["manifest"])
        session = pilot["sessions"][0]
        batch_path = Path(session["batch_path"]) / "batch_report.json"
        if digest(batch_path) != session["batch_sha256"]:
            raise ValueError("Batch report changed after collection")
        batch = load(batch_path)
        # Fixed before viewing predictions: first coordinate flight, first and
        # last RGB observation. Frames remain clustered by flight and session.
        raw_path = Path(batch["runs"][0]["report_path"])
        raw = load(raw_path)
        if raw["policy"] != "coordinate":
            raise ValueError("Expected the first coordinate flight")
        if raw["configuration"]["condition_id"] != entry["condition_id"]:
            raise ValueError("Condition mismatch")
        observations = raw["observations"]
        for index in (0, len(observations) - 1):
            row = observations[index]
            image = Path(row["raw_image"])
            if digest(image) != row["raw_image_sha256"]:
                raise ValueError("RGB hash mismatch")
            frames.append({"condition_id": entry["condition_id"],
                "session": session["engine_process_id"], "source_report": str(raw_path),
                "source_report_sha256": digest(raw_path), "frame_index": index + 1,
                "capture_time": row["time"], "image": str(image),
                "image_sha256": row["raw_image_sha256"],
                "processing_budget_s": raw["configuration"]["visual"]["frame_max_age_s"]})
    return frames


def checked_box(box, width, height):
    if len(box) != 4:
        raise ValueError("Expected four box coordinates")
    values = [float(x) for x in box]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite box")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box outside image or empty")
    return values


def cloud_contract(result, width, height):
    boxes = [checked_box([item[key] for key in ("xmin", "ymin", "xmax", "ymax")],
                         width, height) for item in result.get("result_bbox_list", [])]
    return {"selection_status": "SINGLE_BOX" if len(boxes) == 1 else (
                "NO_BOX" if not boxes else "AMBIGUOUS"), "boxes": boxes,
            "answer": result.get("answer", "")}


def latency_summary(rows, method):
    subset = [row[method] for row in rows if method in row]
    completed = [row["wall_seconds"] for row in subset if row["status"] == "RETURNED"]
    return {"attempted": len(subset), "returned": len(completed),
        "error_count": sum(row["status"] == "ERROR" for row in subset),
        "wall_latency_p50_s": float(np.median(completed)) if completed else None,
        "wall_latency_p95_s": float(np.percentile(completed, 95)) if completed else None,
        "processing_budget_met": sum(row.get("processing_budget_met") is True for row in subset),
        "interpretation": "Tiny offline timing probe; includes first call, excludes acquisition and queue delay"}


def run(args):
    frames = select_frames(args.sweep)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing timing results: {output}")
    output.mkdir(parents=True)
    from yolo_detector import YoloTargetDetector
    model_started = time.perf_counter()
    detector = YoloTargetDetector(args.weights, confidence=0.25, imgsz=640, device="cpu")
    model_load_seconds = time.perf_counter() - model_started
    worker = None
    if args.run_vlx:
        from dotenv import load_dotenv
        from vlx_api_client import VLXSeekAPIClient
        load_dotenv(ROOT / ".env")
        worker = VLXSeekAPIClient(timeout=30, max_retries=1)
    report = {"status": "RUNNING", "started": datetime.now(timezone.utc).isoformat(),
        "sweep": str(args.sweep.resolve()), "sweep_sha256": digest(args.sweep),
        "sampling": "First coordinate flight in each condition, first and last unannotated RGB",
        "label_status": "NOT_HUMAN_VERIFIED_NO_ACCURACY_ESTIMATE",
        "weights": str(args.weights.resolve()), "weights_sha256": digest(args.weights),
        "model_load_seconds": model_load_seconds, "device": "cpu", "imgsz": 640,
        "vlx_executed": worker is not None, "vlx_description": "红色气球",
        "vlx_request_mode": worker.request_mode if worker else None,
        "vlx_proposals_transmitted": worker.proposals_transmitted if worker else None,
        "vlx_model": worker.model if worker else None, "vlx_timeout_s": 30, "vlx_retries": 1,
        "online_control_enabled": False, "frames": []}
    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        for index, frame in enumerate(frames):
            image = cv2.imdecode(np.fromfile(frame["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode saved RGB")
            height, width = image.shape[:2]
            row = dict(frame)
            report["frames"].append(row)
            for name, operation in (("color", lambda: detect_balloon(image)),
                                    ("yolo", lambda: detector.detect(image))):
                started = time.perf_counter()
                try:
                    result = operation()
                    elapsed = time.perf_counter() - started
                    row[name] = {"status": "RETURNED", "wall_seconds": elapsed,
                        "processing_budget_met": elapsed <= frame["processing_budget_s"],
                        "first_call": index == 0, "result": result}
                except Exception as exc:
                    row[name] = {"status": "ERROR", "error_type": type(exc).__name__,
                                 "wall_seconds": time.perf_counter() - started}
            if worker:
                from PIL import Image
                rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                started = time.perf_counter()
                try:
                    # Current cloud endpoint does not receive YOLO proposals.
                    result = worker.ground(rgb, [], "红色气球", lang="zh", temperature=0.0)
                    elapsed = time.perf_counter() - started
                    row["vlx_image_text"] = {"status": "RETURNED", "wall_seconds": elapsed,
                        "processing_budget_met": elapsed <= frame["processing_budget_s"],
                        "first_call": index == 0, "result": cloud_contract(result, width, height)}
                except Exception as exc:
                    # Do not serialize transport errors: they may contain URLs.
                    row["vlx_image_text"] = {"status": "ERROR", "error_type": type(exc).__name__,
                                            "wall_seconds": time.perf_counter() - started}
            save()
            print(f"Measured {index + 1}/{len(frames)}: {frame['condition_id']}", flush=True)
        methods = ["color", "yolo"] + (["vlx_image_text"] if worker else [])
        report["summaries"] = {name: latency_summary(report["frames"], name) for name in methods}
        report["status"] = "COMPLETE_WITH_ERRORS" if any(
            value["error_count"] for value in report["summaries"].values()) else "PASS"
    except BaseException as exc:
        report.update(status="INTERRUPTED", error_type=type(exc).__name__)
        raise
    finally:
        report["finished"] = datetime.now(timezone.utc).isoformat()
        save()
    lines = ["# 单机原图感知与延迟探测", "",
        f"状态：{report['status']}；3 个条件、{len(frames)} 帧；每个条件来自一场飞行的两帧。", "",
        "| 方法 | 返回 / 尝试 | 墙钟中位数 / s | 墙钟 P95 / s | 处理耗时 ≤ 0.5 s |",
        "|---|---:|---:|---:|---:|"]
    for name, counts in report["summaries"].items():
        median, p95 = counts["wall_latency_p50_s"], counts["wall_latency_p95_s"]
        lines.append(f"| {name} | {counts['returned']}/{counts['attempted']} | "
                     f"{median if median is not None else 'NA'} | {p95 if p95 is not None else 'NA'} | "
                     f"{counts['processing_budget_met']}/{counts['attempted']} |")
    lines.extend(["", "这是离线处理耗时，不含采集与排队；含第一次模型调用。"
        "六帧无独立人工真值，不报告准确率，也不能据此保证在线时延。", "",
        "YOLO 使用预训练 COCO 模型，将 sports ball 映射为 balloon；尚未进行专用气球训练。"
        "云端 VLX 使用图像＋文字请求；YOLO 与 VLX 分别测量，未启用闭环 agent。", "",
        "AI 使用说明：探测脚本和解释由 AI 辅助编制；原图、调用结果和耗时保存在 report.json。", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--run-vlx", action="store_true")
    run(parser.parse_args())
