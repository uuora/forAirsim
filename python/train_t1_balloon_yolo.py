"""Offline, validation-selected T1 balloon adaptation with bound receipts.

This launcher validates only train/validation image and label bytes. Test paths
never enter training data.yaml or training model calls. It does not install
dependencies, download weights, tune against test data, or grant flight access.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("specialist_balloon_detector.py", "train_t1_balloon_yolo.py",
                "evaluate_t1_balloon_yolo.py", "test_t1_specialist_detector.py")


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def save_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_numeric_results_csv(path: Path | str) -> dict[str, list[Any]]:
    """Read trainer numeric CSV metadata without the unavailable Polars parser.

    The local BaseTrainer method returns columns as lists and returns {} on a
    read failure. Preserve integer epoch columns and floating-point losses/
    metrics; blank fields represent nulls. This affects checkpoint metadata
    serialization only, not the optimizer, batches, metrics or selection rule.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            headers = next(reader)
            if not headers or len(headers) != len(set(headers)):
                return {}
            rows = list(reader)
        if any(len(row) != len(headers) for row in rows):
            return {}
        columns = {}
        for index, header in enumerate(headers):
            values = [row[index].strip() for row in rows]
            integer_column = all(not value or re.fullmatch(r"[+-]?\d+", value) for value in values)
            columns[header] = [None if not value else int(value) if integer_column else float(value)
                               for value in values]
        return columns
    except (OSError, csv.Error, StopIteration, ValueError):
        return {}


def csv_detection_trainer():
    """Create the narrow trainer subclass after offline runtime setup."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    class StdlibCSVDetectionTrainer(DetectionTrainer):
        def read_results_csv(self):
            return read_numeric_results_csv(self.csv)
    return StdlibCSVDetectionTrainer


def configure_offline_runtime() -> dict[str, Any]:
    """Set local caches before importing torch/Ultralytics/Matplotlib."""
    if "ultralytics" in sys.modules:
        raise RuntimeError("Configure offline runtime before importing Ultralytics")
    base = ROOT/"logs"/"t1_balloon_model_runtime"
    values = {"YOLO_CONFIG_DIR": str(base/"yolo"), "TORCH_HOME": str(base/"torch"),
        "MPLCONFIGDIR": str(base/"matplotlib"), "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false", "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED"}
    for key in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR"):
        Path(values[key]).mkdir(parents=True, exist_ok=True)
    os.environ.update(values)
    # Ultralytics appends Ultralytics to YOLO_CONFIG_DIR. Prepopulate the font
    # cache because its font checker can otherwise attempt a remote download.
    font_directory = Path(values["YOLO_CONFIG_DIR"])/"Ultralytics"
    font_directory.mkdir(parents=True, exist_ok=True)
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.is_file():
        raise FileNotFoundError("Local Arial font required for offline validation")
    destination = font_directory/"Arial.ttf"
    if not destination.exists():
        shutil.copyfile(font, destination)
    if sha256(destination) != sha256(font):
        raise ValueError("Cached local Arial differs")
    return {"environment": values, "font": str(destination),
            "font_sha256": sha256(destination),
            "network_policy": "current-process Python socket connects and DNS denied; native/subprocess networking not intercepted"}


class OfflineNetworkError(RuntimeError):
    pass


@contextmanager
def forbid_network():
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    def denied(*unused, **unused_kwargs):
        raise OfflineNetworkError("Network access prohibited by frozen training protocol")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.getaddrinfo = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex
        socket.getaddrinfo = original_getaddrinfo


def disable_integrations() -> None:
    from ultralytics.utils import SETTINGS
    keys = ("sync", "hub", "wandb", "comet", "mlflow", "clearml", "neptune",
            "tensorboard", "dvc", "raytune", "vscode_msg", "openvino_msg")
    SETTINGS.update({key: False for key in keys if key in SETTINGS})


def runtime_versions() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "clock": "perf_counter_ns",
        "clock_resolution_s": time.get_clock_info("perf_counter").resolution}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "UNAVAILABLE"
    return versions


def source_snapshot(output: Path) -> dict[str, str]:
    folder = output/"source_snapshot"
    folder.mkdir()
    result = {}
    for name in SOURCE_FILES:
        source = Path(__file__).parent/name
        if not source.is_file():
            raise FileNotFoundError("Missing owned source module")
        result[name] = sha256(source)
        shutil.copyfile(source, folder/name)
    return result


def dataset_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset rows required")
    normalised = []
    for original in rows:
        row = dict(original)
        row["image"] = row.get("image_path", row.get("image"))
        row["rgb_sha256"] = row.get("image_sha256", row.get("rgb_sha256"))
        row["label"] = row.get("label_path", row.get("label"))
        normalised.append(row)
    return normalised


def validate_binding(manifest_path: Path, plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, plan = read_json(manifest_path), read_json(plan_path)
    if manifest.get("status") != "PASS" or manifest.get("train_val_test_ready") is not True:
        raise ValueError("A verified episode-grouped dataset is required")
    if manifest.get("plan_sha256") != sha256(plan_path):
        raise ValueError("Dataset is not bound to the requested frozen plan")
    if manifest.get("class_names") != {"0": "balloon"}:
        raise ValueError("One-class visible balloon labels required")
    expected = {entry["episode_id"]: entry["split"] for entry in plan["episodes"]}
    if len(expected) != len(plan["episodes"]):
        raise ValueError("Duplicate planned episode")
    episode_splits, image_splits, frame_ids = {}, {}, set()
    counts = {key: 0 for key in ("train", "validation", "test")}
    for row in dataset_rows(manifest):
        episode, split, image_hash = row["episode_id"], row["split"], row["rgb_sha256"]
        if split not in counts or expected.get(episode) != split:
            raise ValueError("Unplanned episode or split")
        if episode in episode_splits and episode_splits[episode] != split:
            raise ValueError("Episode crosses splits")
        if image_hash in image_splits and image_splits[image_hash] != split:
            raise ValueError("Identical RGB crosses splits")
        frame_key = (episode, row["frame_id"])
        if frame_key in frame_ids:
            raise ValueError("Duplicate frame")
        frame_ids.add(frame_key)
        episode_splits[episode], image_splits[image_hash] = split, split
        counts[split] += 1
    if set(episode_splits) != set(expected):
        raise ValueError("A planned episode is missing")
    expected_per_episode = plan["cases_per_episode"]*plan["frames_per_case"]
    for episode in expected:
        if sum(row["episode_id"] == episode for row in dataset_rows(manifest)) != expected_per_episode:
            raise ValueError("Episode frame count differs from frozen plan")
    if not all(counts.values()):
        raise ValueError("All three independent splits are required")
    return manifest, plan


def validate_label_file(row: dict[str, Any], width: int, height: int) -> list[list[float]]:
    label = Path(row["label"])
    if sha256(label) != row["label_sha256"]:
        raise ValueError("Label byte hash mismatch")
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) != 5 or tokens[0] != "0":
            raise ValueError("Expected five-field class-zero YOLO label")
        cx, cy, bw, bh = [float(value) for value in tokens[1:]]
        if not all(math.isfinite(value) for value in (cx, cy, bw, bh)):
            raise ValueError("Nonfinite label")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError("Label normalization invalid")
        if cx-bw/2 < -1e-8 or cy-bh/2 < -1e-8 or cx+bw/2 > 1+1e-8 or cy+bh/2 > 1+1e-8:
            raise ValueError("Label box outside image")
        boxes.append([(cx-bw/2)*width, (cy-bh/2)*height,
                      (cx+bw/2)*width, (cy+bh/2)*height])
    if len(boxes) != row["visible_balloon_count"]:
        raise ValueError("Visible label count mismatch")
    expected_boxes = row.get("visible_boxes_xyxy_pixels")
    if expected_boxes is None:
        instances = row.get("instances_evaluation_only", [])
        expected_boxes = [instance["bbox_xyxy_pixels"] for instance in instances]
    if not isinstance(expected_boxes, list) or len(expected_boxes) != len(boxes):
        raise ValueError("Independent visible boxes required in manifest")
    for predicted, expected in zip(boxes, expected_boxes):
        if len(expected) != 4 or any(abs(a-b) > 1e-5 for a, b in zip(predicted, expected)):
            raise ValueError("YOLO labels differ from independently verified visible boxes")
    return boxes


def validate_split_files(manifest: dict[str, Any], splits: set[str]) -> list[dict[str, Any]]:
    from PIL import Image
    checked = []
    for row in dataset_rows(manifest):
        if row["split"] not in splits:
            continue
        image = Path(row["image"])
        if sha256(image) != row["rgb_sha256"]:
            raise ValueError("RGB byte hash mismatch")
        with Image.open(image) as decoded:
            width, height = decoded.size
            decoded.verify()
        # Ultralytics resolves labels by replacing the images directory.
        expected_label = str(image).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
        if Path(expected_label).with_suffix(".txt").resolve() != Path(row["label"]).resolve():
            raise ValueError("Label path incompatible with Ultralytics image-to-label resolution")
        boxes = validate_label_file(row, width, height)
        checked.append({"episode_id": row["episode_id"], "frame_id": row["frame_id"],
            "split": row["split"], "image": str(image.resolve()), "rgb_sha256": row["rgb_sha256"],
            "label": str(Path(row["label"]).resolve()), "label_sha256": row["label_sha256"],
            "width": width, "height": height, "visible_balloon_count": len(boxes)})
    if not checked:
        raise ValueError("No requested split files")
    return checked


def write_data_yaml(output: Path, manifest: dict[str, Any], *, include_test: bool = False) -> Path:
    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    lists = {}
    for split in splits:
        rows = [row for row in dataset_rows(manifest) if row["split"] == split]
        target = output/(split+"_images.txt")
        target.write_text("".join(Path(row["image"]).resolve().as_posix()+"\n" for row in rows), encoding="utf-8")
        lists["val" if split == "validation" else split] = str(target.resolve())
    path = output/"data.yaml"
    # JSON is a YAML subset, avoiding shell/path escaping and optional converters.
    path.write_text(json.dumps({"path": str(output.resolve()), **lists,
        "names": {"0": "balloon"}}, indent=2)+"\n", encoding="utf-8")
    return path


def run(manifest_path: Path, plan_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "PREFLIGHT", "kind": "T1_BALLOON_ADAPTATION_TRAINING",
        "created_utc": utc_now(), "dataset_manifest": str(manifest_path.resolve()),
        "plan": str(plan_path.resolve()), "test_images_used": 0, "test_labels_used": 0,
        "new_flight_requests": 0, "cloud_requests": 0,
        "selection_rule": "Ultralytics best.pt selected using validation fitness only",
        "failure_attempts": []}
    save_json(output/"receipt.json", receipt)
    started = time.perf_counter_ns()
    try:
        manifest, plan = validate_binding(manifest_path, plan_path)
        checked = validate_split_files(manifest, {"train", "validation"})
        initial = Path(plan["model_initial_weights"])
        if sha256(initial) != plan["model_initial_weights_sha256"]:
            raise ValueError("Frozen initial weights changed")
        receipt.update(dataset_manifest_sha256=sha256(manifest_path), plan_sha256=sha256(plan_path),
            initial_weights=str(initial.resolve()), initial_weights_sha256=sha256(initial),
            checked_train_validation_files=checked, source_sha256=source_snapshot(output),
            runtime_versions=runtime_versions(), offline_runtime=configure_offline_runtime())
        data_yaml = write_data_yaml(output, manifest)
        protocol = dict(plan["training_protocol"])
        required = {"epochs": 20, "imgsz": 640, "batch": 8, "workers": 0,
                    "device": 0, "amp": False, "optimizer": "AdamW"}
        if any(protocol.get(key) != value for key, value in required.items()):
            raise ValueError("Unsupported frozen adaptation training protocol")
        train_arguments = {**protocol, "data": str(data_yaml.resolve()),
            "project": str((output/"training").resolve()), "name": "fixed_protocol",
            "exist_ok": False, "save": True, "val": True, "cache": False,
            "resume": False, "verbose": True}
        receipt.update(status="READY", training_arguments=train_arguments,
            data_yaml=str(data_yaml.resolve()), data_yaml_sha256=sha256(data_yaml))
        save_json(output/"receipt.json", receipt)
        with forbid_network():
            import torch
            from ultralytics import YOLO
            from specialist_balloon_detector import require_balloon_names
            disable_integrations()
            if not torch.cuda.is_available() or torch.cuda.device_count() <= int(protocol["device"]):
                raise RuntimeError("Frozen GPU device unavailable")
            receipt["device_receipt"] = {"cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(), "cuda_available": True,
                "index": int(protocol["device"]),
                "name": torch.cuda.get_device_name(int(protocol["device"])),
                "total_memory_bytes": torch.cuda.get_device_properties(int(protocol["device"])).total_memory}
            model = YOLO(str(initial))
            def actual_arguments(trainer):
                save_json(output/"actual_training_args.json", vars(trainer.args))
                receipt["actual_training_args"] = str((output/"actual_training_args.json").resolve())
                receipt["actual_training_args_sha256"] = sha256(output/"actual_training_args.json")
                save_json(output/"receipt.json", receipt)
            model.add_callback("on_pretrain_routine_start", actual_arguments)
            trainer_class = csv_detection_trainer()
            import ultralytics.engine.trainer as upstream_trainer
            receipt["checkpoint_csv_adapter"] = {
                "class": "StdlibCSVDetectionTrainer(DetectionTrainer)",
                "overridden_methods": ["read_results_csv"],
                "replacement": "stdlib csv; numeric dict-of-lists for checkpoint metadata",
                "reason": "installed Polars CPU-feature parser fails on sse3 flag",
                "upstream_module": str(Path(upstream_trainer.__file__).resolve()),
                "upstream_module_sha256": sha256(upstream_trainer.__file__),
                "optimization_data_seed_epoch_and_selection_changes": False,
                "dependency_install_or_modification": False}
            receipt.update(status="TRAINING", training_started_utc=utc_now())
            save_json(output/"receipt.json", receipt)
            train_start = time.perf_counter_ns()
            metrics = model.train(trainer=trainer_class, **train_arguments)
            receipt["training_elapsed_ns"] = time.perf_counter_ns()-train_start
            trainer = model.trainer
            save_json(output/"actual_training_args.json", vars(trainer.args))
            receipt["actual_training_args"] = str((output/"actual_training_args.json").resolve())
            receipt["actual_training_args_sha256"] = sha256(output/"actual_training_args.json")
            best, last = Path(trainer.best), Path(trainer.last)
            if not best.is_file() or not last.is_file():
                raise FileNotFoundError("Completed training requires best and last checkpoints")
            require_balloon_names(YOLO(str(best)).names)
            receipt.update(best_weights=str(best.resolve()), best_weights_sha256=sha256(best),
                last_weights=str(last.resolve()), last_weights_sha256=sha256(last),
                completed_epochs=int(trainer.epoch)+1, best_validation_fitness=float(trainer.best_fitness),
                final_best_validation_metrics={key: float(value) for key, value in metrics.results_dict.items()},
                checkpoint_selection_data="validation", best_class_names={"0": "balloon"})
        # Detect edits during the run rather than silently blessing different data.
        validate_split_files(manifest, {"train", "validation"})
        if sha256(manifest_path) != receipt["dataset_manifest_sha256"] or sha256(plan_path) != receipt["plan_sha256"]:
            raise ValueError("Plan or dataset manifest changed during training")
        if any(sha256(Path(__file__).parent/name) != value for name, value in receipt["source_sha256"].items()):
            raise ValueError("Owned training/evaluation source changed during training")
        receipt.update(status="PASS", completed_utc=utc_now())
    except Exception as error:
        previous_status = receipt["status"]
        receipt.update(status="FAIL", completed_utc=utc_now())
        receipt["failure_attempts"].append({"error_type": type(error).__name__,
            "stage": previous_status,
            "exception_text_retained": False})
        receipt["elapsed_ns"] = time.perf_counter_ns()-started
        save_json(output/"receipt.json", receipt)
        raise
    receipt["elapsed_ns"] = time.perf_counter_ns()-started
    save_json(output/"receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_manifest, args.plan, args.output)
    print(json.dumps({"status": result["status"], "receipt": str((args.output/"receipt.json").resolve())}))
