"""Read both live AirSim cameras and run the read-only YOLO baseline."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np
import torch
import ultralytics

from yolo_detector import YoloTargetDetector, annotate


ROOT = Path(__file__).resolve().parents[1]
VEHICLES = ("DroneA", "DroneB")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    settings = json.loads(
        (ROOT / "configs" / "maritime_harbor_settings.json").read_text(encoding="utf-8")
    )
    weights = ROOT / "models" / "yolo11n.pt"
    detector = YoloTargetDetector(weights, confidence=0.25, imgsz=1280, device=0)
    output = ROOT / "logs" / datetime.now().strftime("yolo_live_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    client = airsim.MultirotorClient(port=settings["ApiServerPort"], timeout_value=10)
    if not client.ping() or not set(VEHICLES).issubset(client.listVehicles()):
        raise RuntimeError("Live maritime AirSim instance with both vehicles is required")
    states = {
        vehicle: {
            "landed_state": int(client.getMultirotorState(vehicle_name=vehicle).landed_state),
            "api_control_enabled": bool(client.isApiControlEnabled(vehicle_name=vehicle)),
        }
        for vehicle in VEHICLES
    }
    records = []
    annotated_frames = []
    for vehicle in VEHICLES:
        capture_started = time.perf_counter()
        payload = client.simGetImage(
            "front_center", airsim.ImageType.Scene, vehicle_name=vehicle
        )
        capture_ms = (time.perf_counter() - capture_started) * 1000
        if not payload:
            raise RuntimeError(f"No camera payload from {vehicle}")
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot decode camera payload from {vehicle}")
        raw_path = output / f"{vehicle}_rgb.png"
        raw_path.write_bytes(bytes(payload))
        detection = detector.detect(frame)
        preview = annotate(frame, detection)
        cv2.putText(
            preview,
            f"{vehicle} LIVE YOLO11n 0.25",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )
        preview_path = output / f"{vehicle}_predictions.jpg"
        cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 94])
        annotated_frames.append(preview)
        records.append(
            {
                "vehicle": vehicle,
                "camera": "front_center",
                "image": raw_path.name,
                "image_sha256": sha256(raw_path),
                "capture_host_ms": capture_ms,
                "detection": detection,
                "preview": preview_path.name,
            }
        )
    contact = np.hstack(annotated_frames)
    cv2.imwrite(str(output / "live_yolo_contact_sheet.jpg"), contact, [cv2.IMWRITE_JPEG_QUALITY, 92])
    report = {
        "status": "PASS",
        "mode": "read-only live AirSim camera snapshot with pretrained YOLO11n",
        "flight_commands_issued": False,
        "api_control_requested": False,
        "states_before_capture": states,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "records": records,
        "limitations": [
            "COCO boat and sports ball are provisional mappings to vessel and balloon.",
            "This two-frame live snapshot has no synchronized ground-truth query.",
            "No flight command, tracking, control decision, or model fine-tuning was performed.",
        ],
    }
    (output / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "logs" / "yolo_live_latest.json").write_text(
        json.dumps({"output": str(output)}, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "detections": {
                    row["vehicle"]: row["detection"]["candidate_count"] for row in records
                },
                "states": states,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
