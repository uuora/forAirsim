"""Reusable YOLO image detector with explicit COCO-to-project class mapping."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


APPLICATION_CLASS_MAP = {"boat": "vessel", "sports ball": "balloon"}


class YoloTargetDetector:
    def __init__(self, weights, confidence=0.25, imgsz=1280, device=0):
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = device
        self.model = YOLO(str(self.weights))

    def detect(self, bgr):
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image")
        height, width = bgr.shape[:2]
        result = self.model.predict(
            source=bgr,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=0.5,
            device=self.device,
            verbose=False,
        )[0]
        candidates = []
        if result.boxes is not None:
            for xyxy, class_id, confidence in zip(
                result.boxes.xyxy.detach().cpu().numpy(),
                result.boxes.cls.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy(),
            ):
                class_id = int(class_id)
                coco_name = str(result.names[class_id])
                mapped = APPLICATION_CLASS_MAP.get(coco_name)
                if mapped is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                candidates.append(
                    {
                        "class_name": mapped,
                        "source_class_id": class_id,
                        "source_class_name": coco_name,
                        "confidence": float(confidence),
                        "bbox_xyxy_pixels": [x1, y1, x2, y2],
                        "centre_px": [cx, cy],
                        "normalised_error_xy": [
                            (cx - width / 2) / (width / 2),
                            (cy - height / 2) / (height / 2),
                        ],
                        "truncated": bool(x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height),
                    }
                )
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "source": "Ultralytics YOLO pretrained COCO mapping",
            "confidence_threshold": self.confidence,
            "image_size_wh": [width, height],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "speed_ms": {key: float(value) for key, value in result.speed.items()},
        }


def annotate(bgr, detection):
    output = bgr.copy()
    colors = {"vessel": (0, 255, 0), "balloon": (255, 255, 0)}
    for candidate in detection["candidates"]:
        x1, y1, x2, y2 = [int(round(value)) for value in candidate["bbox_xyxy_pixels"]]
        color = colors[candidate["class_name"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{candidate['class_name']} {candidate['confidence']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )
    return output
