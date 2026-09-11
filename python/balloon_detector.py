"""Image-only red/ellipse candidate detector. Scores are not probabilities."""
import cv2
import numpy as np


def detect_balloon(bgr):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image")
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 65), (12, 255, 255)) | cv2.inRange(hsv, (168, 100, 65), (179, 255, 255))
    # The configured front camera can see red parts of the vehicle at the bottom.
    mask[int(height * 0.85):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(200, width * height * 0.0008) or len(contour) < 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if y + h >= int(height * 0.85):
            continue
        circularity = float(4 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1))
        fill = float(area / (w * h))
        if not (0.45 < w / h < 1.35 and circularity > 0.65 and fill > 0.6):
            continue
        moments = cv2.moments(contour)
        cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        candidates.append({"bbox_xywh": [x, y, w, h], "centre_px": [cx, cy],
                           "truncated": bool(x <= 0 or y <= 0 or x+w >= width),
                           "normalised_error_xy": [(cx-width/2)/(width/2), (cy-height/2)/(height/2)],
                           "area_px": float(area), "shape_score": circularity * fill})
    candidates.sort(key=lambda item: item["area_px"], reverse=True)
    return {"detected": bool(candidates), "candidate_count": len(candidates),
            "selected": candidates[0] if candidates else None,
            "candidates": candidates, "image_size_wh": [width, height]}


def annotate(bgr, result):
    output = bgr.copy()
    for index, candidate in enumerate(result["candidates"]):
        x, y, w, h = candidate["bbox_xywh"]
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = map(round, candidate["centre_px"])
        cv2.circle(output, (cx, cy), 4, (0, 255, 255), -1)
    cv2.putText(output, "RED BALLOON CANDIDATE" if result["detected"] else "NO BALLOON CANDIDATE",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if result["detected"] else (0,200,255), 2)
    return output
