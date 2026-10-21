"""Replay saved maritime frames through YOLO proposals and optional VLX-Seek.

Simulator boxes are used only for evaluation, never passed to a detector or VLX.
VLX-Seek weights and its source tree must be supplied explicitly.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# Load environment variables from .env file
load_dotenv(ROOT / ".env")

DESCRIPTION = "the rightmost red balloon in the image"


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def valid_box(box, width, height):
    if len(box) != 4:
        raise ValueError("Proposal must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = [float(value) for value in box]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Proposal outside image or empty: {box}")
    return [x1, y1, x2, y2]


def red_region_proposals(image, minimum_area=40):
    """Permissive visible-red components, including partly occluded objects."""
    import cv2
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 65), (12, 255, 255)) | cv2.inRange(
        hsv, (168, 100, 65), (179, 255, 255))
    mask[int(height * 0.85):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return sorted(boxes, key=lambda box: (box[0], box[1]))


def build_proposals(yolo_result, image, mode="yolo", limit=100):
    """Keep detector order stable; add color boxes only in the named ablation."""
    height, width = image.shape[:2]
    proposals = []
    for candidate in yolo_result["candidates"]:
        box = valid_box(candidate["bbox_xyxy_pixels"], width, height)
        proposals.append({"bbox_xyxy": box, "source": "yolo",
                          "class_name": candidate["class_name"],
                          "confidence": candidate["confidence"]})
    if mode in ("yolo_color", "yolo_red_regions"):
        if mode == "yolo_color":
            from balloon_detector import detect_balloon
            color_boxes = []
            for candidate in detect_balloon(image)["candidates"]:
                x, y, w, h = candidate["bbox_xywh"]
                color_boxes.append([x, y, x + w, y + h])
        else:
            color_boxes = red_region_proposals(image)
        for candidate_box in color_boxes:
            box = valid_box(candidate_box, width, height)
            if any(box_iou(box, item["bbox_xyxy"]) >= 0.5 for item in proposals):
                continue
            proposals.append({"bbox_xyxy": box, "source": mode,
                              "class_name": "balloon", "confidence": None})
    if mode not in ("yolo", "yolo_color", "yolo_red_regions"):
        raise ValueError(f"Unknown proposal mode: {mode}")
    return proposals[:limit]


def evaluate_proposals(proposals, truth_boxes, target_id):
    """Use saved AirSim boxes strictly after image-only proposal generation."""
    target = truth_boxes.get(target_id)
    scores = [box_iou(item["bbox_xyxy"], target) for item in proposals] if target else []
    best_index = max(range(len(scores)), key=scores.__getitem__) if scores else None
    return {"target_box_returned": target is not None,
            "target_best_proposal_iou": scores[best_index] if best_index is not None else 0.0,
            "target_best_proposal_index": best_index,
            "target_proposal_recalled_iou_0_1": any(score >= 0.1 for score in scores),
            "target_proposal_recalled_iou_0_5": any(score >= 0.5 for score in scores)}


def selected_vlx_box(result):
    boxes = result.get("result_bbox_list", [])
    if len(boxes) != 1:
        return None
    item = boxes[0]
    return [float(item[key]) for key in ("xmin", "ymin", "xmax", "ymax")]


def validated_vlx_box(result, proposals, tolerance=10.0):
    """Associate returned box with the closest input proposal.

    VLX-Seek API may return boxes with slight pixel differences from
    the input proposals due to refinement. We accept boxes within tolerance
    and record large deviations for analysis.

    Args:
        result: VLX API response.
        proposals: Input proposal list.
        tolerance: Maximum pixel difference to accept (default 10.0).

    Returns:
        Selected box [x1, y1, x2, y2], or None if no selection.
    """
    selected = selected_vlx_box(result)
    if selected is None:
        return None

    # Find closest matching proposal
    min_diff = float('inf')
    closest = None
    for proposal in proposals:
        diff = sum(abs(left - right)
                   for left, right in zip(selected, proposal["bbox_xyxy"]))
        if diff < min_diff:
            min_diff = diff
            closest = proposal["bbox_xyxy"]

    if min_diff > tolerance:
        raise ValueError(f"VLX box differs from all supplied proposals by "
                         f"{min_diff:.1f} pixels: {selected}; closest: {closest}")

    return selected


def evaluate_selection(box, truth_boxes, target_id):
    if box is None:
        return {"association": "NO_SELECTION", "associated_target_id": None, "association_iou": 0.0}
    score, identity = max(((box_iou(box, ground), name)
                           for name, ground in truth_boxes.items()), default=(0.0, None))
    if score < 0.1:
        return {"association": "UNMATCHED_SELECTION", "associated_target_id": None,
                "association_iou": score}
    return {"association": "TARGET" if identity == target_id else "OTHER_INSTANCE",
            "associated_target_id": identity, "association_iou": score}


def summarize(frames):
    profiles = sorted({entry["profile"] for entry in frames})
    summary = {}
    for profile in profiles:
        subset = [entry for entry in frames if entry["profile"] == profile]
        summary[profile] = {
            "frames": len(subset),
            "target_proposal_iou_ge_0_1": sum(entry["proposal_evaluation"][
                "target_proposal_recalled_iou_0_1"] for entry in subset),
            "target_proposal_iou_ge_0_5": sum(entry["proposal_evaluation"][
                "target_proposal_recalled_iou_0_5"] for entry in subset),
            "yolo_top_balloon_target": sum(entry["yolo_top_balloon_evaluation"][
                "association"] == "TARGET" for entry in subset),
            "rightmost_balloon_target": sum(entry["rightmost_balloon_evaluation"][
                "association"] == "TARGET" for entry in subset),
        }
        if any("vlx_selection_evaluation" in entry for entry in subset):
            summary[profile]["vlx_target"] = sum(entry["vlx_selection_evaluation"][
                "association"] == "TARGET" for entry in subset)
    return summary


def load_worker(repo, model_path, device, use_api=False, api_model=None):
    """Load VLX worker: either local inference or cloud API client.

    Args:
        repo: Path to VLX-Seek source (local mode only).
        model_path: Model path or HuggingFace ID (local mode only).
        device: CUDA device (local mode only).
        use_api: If True, use cloud API instead of local inference.
        api_model: Model ID for API ("vlx-seek" or "vlx-seek-1.5-3b").

    Returns:
        Worker instance with ground() method.
    """
    if use_api:
        from vlx_api_client import VLXSeekAPIClient
        model = api_model or "vlx-seek-1.5-3b"
        return VLXSeekAPIClient(model=model)
    else:
        repo = Path(repo).resolve()
        if not (repo / "vlx_seek_worker.py").is_file():
            raise FileNotFoundError(f"VLX-Seek source missing: {repo}")
        if model_path != "omlab/VLX-Seek-1.5-10B" and not Path(model_path).is_dir():
            raise FileNotFoundError(f"VLX-Seek checkpoint missing: {model_path}")
        sys.path.insert(0, str(repo))
        from vlx_seek_worker import VLXSeekWorker
        return VLXSeekWorker(model_path=model_path, device=device)


def run(args, detector=None, worker=None):
    source = Path(args.report).resolve()
    saved = json.loads(source.read_text(encoding="utf-8"))
    if saved.get("status") != "PASS" or saved["config"]["scene"] != "MaritimeHarborV2":
        raise ValueError("Expected a successful MaritimeHarborV2 capture report")
    target_id = saved["config"]["target_id"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cached_report = None
    cached_frames = None
    if args.proposals_report:
        if not args.run_vlx:
            raise ValueError("--proposals-report requires --run-vlx")
        cached_report = json.loads(Path(args.proposals_report).read_text(encoding="utf-8"))
        if (cached_report.get("status") != "PASS"
                or cached_report.get("source_report_sha256") != source_hash
                or cached_report.get("proposal_mode") != args.proposal_mode):
            raise ValueError("Cached proposal report does not match source capture or mode")
        cached_frames = {entry["frame"]: entry for entry in cached_report["frames"]}
        if len(cached_frames) != len(cached_report["frames"]):
            raise ValueError("Cached proposal report has duplicate frame names")
        expected_count = sum(len(condition["frames"]) for condition in saved["conditions"])
        if len(cached_frames) != expected_count:
            raise ValueError("Cached proposal report frame count differs from capture")
    elif detector is None:
        from yolo_detector import YoloTargetDetector
        detector = YoloTargetDetector(args.weights, confidence=args.confidence,
                                      imgsz=args.imgsz, device=args.device)
    if args.run_vlx and worker is None:
        if args.use_api:
            worker = load_worker(None, None, None, use_api=True, api_model=args.api_model)
        else:
            if not args.vlx_repo or not args.model_path:
                raise ValueError("--run-vlx needs --vlx-repo and --model-path (or --use-api)")
            worker = load_worker(args.vlx_repo, args.model_path, args.vlx_device)
    output = Path(args.output).resolve()
    if args.proposals_report and output == Path(args.proposals_report).resolve():
        raise ValueError("Output must not overwrite the cached proposal report")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_report": str(source), "source_report_sha256": source_hash,
              "weights": cached_report["weights"] if cached_report else str(Path(args.weights).resolve()),
              "weights_sha256": cached_report["weights_sha256"] if cached_report else
                  hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              "proposals_report": str(Path(args.proposals_report).resolve())
                  if args.proposals_report else None,
              "proposal_mode": args.proposal_mode, "yolo_confidence": args.confidence,
              "yolo_imgsz": args.imgsz, "vlx_executed": bool(args.run_vlx),
              "vlx_backend": ("cloud_api" if getattr(args, "use_api", False) else "local_worker")
                  if args.run_vlx else None,
              "vlx_request_mode": (getattr(worker, "request_mode", "region_proposals")
                  if args.run_vlx else None),
              "vlx_proposals_transmitted": (getattr(worker, "proposals_transmitted", True)
                  if args.run_vlx else None),
              "vlx_api_model": getattr(args, "api_model", None)
                  if args.run_vlx and getattr(args, "use_api", False) else None,
              "vlx_model_path": args.model_path if args.run_vlx else None,
              "description_override": args.description,
              "description_policy": "per-frame preregistered task, else fixed fallback",
              "target_id_evaluation_only": target_id,
              "frames": []}
    from PIL import Image
    for condition in saved["conditions"]:
        for frame_record in condition["frames"]:
            frame_path = source.parent / frame_record["frame"]
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            if cached_frames is not None:
                pil_image = Image.open(frame_path).convert("RGB")
            else:
                import cv2
                image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Unreadable RGB frame: {frame_path}")
            truth = frame_record["simulator_boxes_xyxy"]
            image_hash = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            if cached_frames is not None:
                if frame_record["frame"] not in cached_frames:
                    raise ValueError(f"Cached proposal missing frame: {frame_record['frame']}")
                entry = dict(cached_frames[frame_record["frame"]])
                if entry["image_sha256"] != image_hash or entry["profile"] != condition["profile"]:
                    raise ValueError(f"Cached image or profile changed: {frame_record['frame']}")
                proposals = entry["proposals"]
                if len(proposals) > 100:
                    raise ValueError("VLX supports at most 100 proposals")
                for proposal in proposals:
                    valid_box(proposal["bbox_xyxy"], *pil_image.size)
            else:
                start = time.perf_counter()
                yolo = detector.detect(image)
                yolo_wall_ms = (time.perf_counter() - start) * 1000
                proposals = build_proposals(yolo, image, args.proposal_mode)
                yolo_balloons = [item for item in yolo["candidates"]
                                 if item["class_name"] == "balloon"]
                top_balloon = yolo_balloons[0] if yolo_balloons else None
                rightmost = max(yolo_balloons,
                                key=lambda item: sum(item["bbox_xyxy_pixels"][::2])) \
                    if yolo_balloons else None
                entry = {"profile": condition["profile"], "frame": frame_record["frame"],
                         "image_sha256": image_hash,
                         "yolo_candidate_count": yolo["candidate_count"],
                         "yolo_wall_ms": yolo_wall_ms,
                         "yolo_candidates": yolo["candidates"],
                         "yolo_top_balloon_evaluation": evaluate_selection(
                             top_balloon["bbox_xyxy_pixels"] if top_balloon else None,
                             truth, target_id),
                         "rightmost_balloon_evaluation": evaluate_selection(
                             rightmost["bbox_xyxy_pixels"] if rightmost else None,
                             truth, target_id),
                         "proposals": proposals,
                         "proposal_evaluation": evaluate_proposals(proposals, truth, target_id)}
            description = (args.description or entry.get("task_description")
                           or frame_record.get("task_description") or DESCRIPTION)
            entry["task_description"] = description
            if args.run_vlx:
                if proposals:
                    if cached_frames is None:
                        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    start = time.perf_counter()
                    vlx = worker.ground(pil_image, [p["bbox_xyxy"] for p in proposals],
                                        description, lang="en", temperature=0.0,
                                        max_new_tokens=128)
                    entry["vlx_wall_ms"] = (time.perf_counter() - start) * 1000
                    entry["vlx_result"] = vlx
                    selected = (selected_vlx_box(vlx) if getattr(args, "use_api", False)
                                else validated_vlx_box(vlx, proposals))
                    entry["vlx_selection_evaluation"] = evaluate_selection(
                        selected, truth, target_id)
                else:
                    entry["vlx_result"] = None
                    entry["vlx_selection_evaluation"] = evaluate_selection(None, truth, target_id)
            result["frames"].append(entry)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int),
                              encoding="utf-8")
    result["status"] = "PASS"
    result["summary_by_profile"] = summarize(result["frames"])
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="Saved maritime_single report.json; source images must be beside it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals-report", type=Path,
                        help="Reuse a completed YOLO proposal report in a separate VLX environment")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--proposal-mode", choices=("yolo", "yolo_color", "yolo_red_regions"),
                        default="yolo")
    parser.add_argument("--description", default=None,
                        help="Optional override; otherwise use the preregistered frame task")
    parser.add_argument("--run-vlx", action="store_true")
    parser.add_argument("--use-api", action="store_true",
                        help="Use Om AI Lab cloud API instead of local inference")
    parser.add_argument("--api-model", default="vlx-seek-1.5-3b",
                        choices=["vlx-seek", "vlx-seek-1.5-3b"],
                        help="Model ID for API mode")
    parser.add_argument("--vlx-repo", type=Path,
                        help="VLX-Seek source directory (local mode only)")
    parser.add_argument("--model-path",
                        help="VLX-Seek model weights path (local mode only)")
    parser.add_argument("--vlx-device", default="cuda",
                        help="CUDA device for local inference")
    return parser.parse_args(argv)


if __name__ == "__main__":
    options = parse_args()
    completed = run(options)
    print(json.dumps({"status": completed["status"], "frames": len(completed["frames"]),
                      "vlx_executed": completed["vlx_executed"],
                      "output": str(options.output.resolve())}, ensure_ascii=False))
