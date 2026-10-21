"""One-class balloon detector, compatible with the bounded YOLO/VLX adapter.

The learned class is appearance only. A detector score does not establish
designated-instance identity and is not a calibrated identity probability.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def require_balloon_names(names: Any) -> dict[int, str]:
    """Reject an unadapted COCO head rather than translating sports ball."""
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or names != {0: "balloon"} or any(type(key) is not int for key in names):
        raise ValueError("Specialist weights must have exactly {0: 'balloon'}")
    return names


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class SpecialistBalloonDetector:
    """An explicit one-class model; accepts an injected predictor for tests."""

    def __init__(self, weights: Path | str | None = None, confidence: float = 0.25,
                 imgsz: int = 640, device: str | int = "cpu", *,
                 nms_iou: float = 0.5, model: Any = None):
        if (isinstance(confidence, bool) or not math.isfinite(float(confidence))
                or not 0 < float(confidence) <= 1):
            raise ValueError("confidence must be in (0,1]")
        if (isinstance(nms_iou, bool) or not math.isfinite(float(nms_iou))
                or not 0 < float(nms_iou) <= 1):
            raise ValueError("nms_iou must be in (0,1]")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("imgsz must be a positive integer")
        self.weights = Path(weights).resolve() if weights is not None else None
        if model is None:
            if self.weights is None or not self.weights.is_file():
                raise FileNotFoundError("Explicit local specialist weights required")
            from ultralytics import YOLO
            model = YOLO(str(self.weights))
        require_balloon_names(model.names)
        self.model, self.confidence = model, float(confidence)
        self.imgsz, self.device, self.nms_iou = imgsz, device, float(nms_iou)

    def detect(self, bgr: np.ndarray) -> dict[str, Any]:
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8
                or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) <= 0):
            raise ValueError("Expected a nonempty uint8 BGR image")
        require_balloon_names(self.model.names)
        results = self.model.predict(source=bgr, imgsz=self.imgsz,
            conf=self.confidence, iou=self.nms_iou, device=self.device,
            verbose=False, save=False)
        if len(results) != 1:
            raise ValueError("Expected one image result")
        result = results[0]
        require_balloon_names(result.names)
        height, width = bgr.shape[:2]
        candidates = []
        if result.boxes is not None:
            boxes, classes, scores = (_array(result.boxes.xyxy),
                _array(result.boxes.cls), _array(result.boxes.conf))
            if classes.ndim != 1 or boxes.shape != (len(classes), 4) or classes.shape != scores.shape:
                raise ValueError("Malformed detector arrays")
            for box, class_id, score in zip(boxes, classes, scores):
                if not np.isfinite(box).all() or not np.isfinite([class_id, score]).all():
                    raise ValueError("Nonfinite detector result")
                if float(class_id) != 0 or not 0 <= float(score) <= 1:
                    raise ValueError("Invalid specialist class or detector score")
                x1, y1, x2, y2 = [float(value) for value in box]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("Empty or out-of-image detector box")
                cx, cy = (x1+x2)/2, (y1+y2)/2
                candidates.append({"class_name": "balloon", "source_class_id": 0,
                    "source_class_name": "balloon", "confidence": float(score),
                    "bbox_xyxy_pixels": [x1, y1, x2, y2], "centre_px": [cx, cy],
                    "normalised_error_xy": [(cx-width/2)/(width/2), (cy-height/2)/(height/2)],
                    "truncated": bool(x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height)})
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return {"source": "Ultralytics specialist one-class balloon detector",
            "confidence_threshold": self.confidence, "nms_iou": self.nms_iou,
            "image_size_wh": [width, height], "candidate_count": len(candidates),
            "candidates": candidates,
            "speed_ms": {key: float(value) for key, value in getattr(result, "speed", {}).items()},
            "score_interpretation": "detector score; NOT calibrated identity probability",
            "designated_instance_identity_established": False}
