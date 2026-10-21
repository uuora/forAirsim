"""Bounded YOLO / cloud VLX observation policy for T1, with no flight API.

Cloud VLX receives image + text. Boxes returned by the cloud are associated
locally with YOLO candidates; this is NOT server proposal ranking, a calibrated
identity probability, or proof that an instance survived an occlusion. Image
times here are host monotonic readback receipts, not exposure timestamps.

The pure policy is injectable for tests. ``YoloVlxBackend`` supports saved PNG
frames and the existing project detector/client, without logging credentials.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Real
from typing import Any, Callable


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _box(value: Any, width: int, height: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("box must contain four pixel coordinates")
    result = tuple(_number(item, "box coordinate") for item in value)
    x1, y1, x2, y2 = result
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("empty or out-of-image box")
    return result


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))
    union = (left[2] - left[0]) * (left[3] - left[1]) + (
        right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    bbox_xyxy_pixels: tuple[float, ...]
    detector_score: float
    source_class_name: str = "sports ball"
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id,
                "bbox_xyxy_pixels": list(self.bbox_xyxy_pixels),
                "detector_score": self.detector_score,
                "score_interpretation": "YOLO detector score; NOT calibrated identity probability",
                "source_class_name": self.source_class_name,
                "class_name": "balloon", "truncated": self.truncated}


@dataclass(frozen=True)
class SemanticFrame:
    session_id: str
    target_generation: int
    frame_id: str
    frame_sequence: int
    image_sha256: str
    width: int
    height: int
    readback_receipt_monotonic_s: float
    candidates_ready_monotonic_s: float
    candidates: tuple[Candidate, ...]
    encoded_image: bytes = field(default=b"", repr=False)
    detector_report_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        _identifier(self.session_id, "session_id")
        _identifier(self.frame_id, "frame_id")
        _integer(self.target_generation, "target_generation")
        _integer(self.frame_sequence, "frame_sequence")
        _digest(self.image_sha256, "image_sha256")
        if _integer(self.width, "width") == 0 or _integer(self.height, "height") == 0:
            raise ValueError("image dimensions must be positive")
        receipt = _number(self.readback_receipt_monotonic_s, "readback receipt")
        ready = _number(self.candidates_ready_monotonic_s, "candidates ready")
        if ready < receipt:
            raise ValueError("candidate readiness precedes readback receipt")
        if not isinstance(self.candidates, tuple) or len(self.candidates) > 100:
            raise ValueError("candidates must be an immutable tuple of at most 100 boxes")
        ids = set()
        for candidate in self.candidates:
            if not isinstance(candidate, Candidate):
                raise ValueError("invalid candidate type")
            _identifier(candidate.candidate_id, "candidate_id")
            if candidate.candidate_id in ids:
                raise ValueError("duplicate candidate IDs")
            ids.add(candidate.candidate_id)
            _box(candidate.bbox_xyxy_pixels, self.width, self.height)
            if not isinstance(candidate.bbox_xyxy_pixels, tuple):
                raise ValueError("candidate box must be immutable")
            score = _number(candidate.detector_score, "detector score")
            if score > 1:
                raise ValueError("detector score outside [0,1]")
        if not isinstance(self.encoded_image, bytes):
            raise ValueError("encoded image must be immutable bytes")
        if self.encoded_image and hashlib.sha256(self.encoded_image).hexdigest() != self.image_sha256:
            raise ValueError("encoded image differs from its bound digest")
        report = json.loads(self.detector_report_json)
        if not isinstance(report, dict):
            raise ValueError("detector report must be a JSON object")

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "target_generation": self.target_generation,
                "frame_id": self.frame_id, "frame_sequence": self.frame_sequence,
                "image_sha256": self.image_sha256, "image_size_wh": [self.width, self.height],
                "readback_receipt_monotonic_s": self.readback_receipt_monotonic_s,
                "candidates_ready_monotonic_s": self.candidates_ready_monotonic_s,
                "time_interpretation": "host readback receipt; NOT exposure timestamp",
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "detector_report": json.loads(self.detector_report_json)}


@dataclass(frozen=True)
class PolicyConfig:
    current_frame_max_age_s: float = 0.5
    semantic_max_age_s: float = 2.0
    response_timeout_s: float = 2.0
    min_request_interval_s: float = 2.0
    source_association_min_iou: float = 0.5
    current_association_min_iou: float = 0.3
    association_margin: float = 0.15

    def __post_init__(self) -> None:
        for name in ("current_frame_max_age_s", "semantic_max_age_s", "response_timeout_s",
                     "min_request_interval_s"):
            _number(getattr(self, name), name, positive=True)
        if self.semantic_max_age_s < self.current_frame_max_age_s:
            raise ValueError("semantic evidence age must allow a fresh current-frame check")
        for name in ("source_association_min_iou", "current_association_min_iou",
                     "association_margin"):
            if not 0 < _number(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} outside (0,1]")


@dataclass(frozen=True)
class SemanticRequest:
    request_id: str
    session_id: str
    target_generation: int
    description: str
    description_sha256: str
    submitted_monotonic_s: float
    frame: SemanticFrame

    def binding(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "session_id": self.session_id,
                "target_generation": self.target_generation,
                "description_sha256": self.description_sha256,
                "submitted_monotonic_s": self.submitted_monotonic_s,
                "frame_id": self.frame.frame_id, "frame_sequence": self.frame.frame_sequence,
                "image_sha256": self.frame.image_sha256,
                "readback_receipt_monotonic_s": self.frame.readback_receipt_monotonic_s}


@dataclass(frozen=True)
class SemanticReply:
    binding: dict[str, Any]
    completed_monotonic_s: float
    result: dict[str, Any]
    request_mode: str = "image_text"
    proposals_transmitted: bool = False


@dataclass
class _Pending:
    request: SemanticRequest
    future: Any
    invalidation_reason: str | None = None


class BoundedSemanticPolicy:
    """At most one outstanding transport; output is perception evidence only.

    Only the main controller thread should call ``request``, ``poll``, ``reset``.
    A timed-out running future keeps the transport slot until it exits. Reset
    invalidates old evidence and results, even if an HTTP call cannot be canceled.
    """

    def __init__(self, backend: Callable[[SemanticRequest], SemanticReply], *,
                 session_id: str, target_generation: int, description: str,
                 config: PolicyConfig | None = None, executor: Any = None,
                 clock: Callable[[], float] = time.monotonic):
        self.backend, self.config, self.clock = backend, config or PolicyConfig(), clock
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="t1-vlx")
        self._pending: _Pending | None = None
        self._evidence: tuple[SemanticRequest, Candidate, dict[str, Any]] | None = None
        self._last_now: float | None = None
        self._last_submitted: float | None = None
        self._last_requested_frame: tuple[Any, ...] | None = None
        self._last_seen_frame: tuple[int, str, str, float] | None = None
        self._serial = 0
        self._closed = False
        self.session_id = _identifier(session_id, "session_id")
        self.target_generation = _integer(target_generation, "target_generation")
        self.description = _identifier(description, "description")
        self.description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()

    def _now(self, value: float | None) -> float:
        result = _number(self.clock() if value is None else value, "policy monotonic time")
        if self._last_now is not None and result < self._last_now:
            raise ValueError("policy monotonic clock regressed")
        self._last_now = result
        return result

    def _decision(self, status: str, reason: str, **extra: Any) -> dict[str, Any]:
        return {"status": status, "reason": reason, "selected_candidate_id": None,
                "session_id": self.session_id, "target_generation": self.target_generation,
                "description_sha256": self.description_sha256,
                "transport_slot_occupied": self._pending is not None,
                "online_control_enabled": False,
                "identity_probability_calibrated": False,
                "instance_identity_guaranteed": False, **extra}

    def _frame_problem(self, frame: SemanticFrame, now: float) -> str | None:
        if not isinstance(frame, SemanticFrame):
            return "INVALID_FRAME_TYPE"
        if (frame.session_id, frame.target_generation) != (self.session_id, self.target_generation):
            return "FRAME_CONTEXT_MISMATCH"
        if frame.candidates_ready_monotonic_s > now or frame.readback_receipt_monotonic_s > now:
            return "FRAME_TIME_IN_FUTURE"
        if now - frame.readback_receipt_monotonic_s > self.config.current_frame_max_age_s:
            return "CURRENT_FRAME_EXPIRED"
        identity = (frame.frame_sequence, frame.frame_id, frame.image_sha256,
                    frame.readback_receipt_monotonic_s)
        if self._last_seen_frame:
            sequence, frame_id, digest, receipt = self._last_seen_frame
            if frame.frame_sequence < sequence or frame.readback_receipt_monotonic_s < receipt:
                return "FRAME_SEQUENCE_OR_RECEIPT_REGRESSION"
            if frame.frame_sequence == sequence and identity != self._last_seen_frame:
                return "FRAME_SEQUENCE_REUSED_WITH_DIFFERENT_BINDING"
            if frame.frame_sequence > sequence and frame.frame_id == frame_id:
                return "FRAME_ID_REUSED"
        self._last_seen_frame = identity
        return None

    def request(self, frame: SemanticFrame, *, now: float | None = None) -> dict[str, Any]:
        now = self._now(now)
        if self._closed:
            return self._decision("UNCERTAIN", "POLICY_CLOSED")
        problem = self._frame_problem(frame, now)
        if problem:
            self._evidence = None
            return self._decision("UNCERTAIN", problem)
        if self._pending is not None:
            return self._decision("UNCERTAIN", "TRANSPORT_SLOT_OCCUPIED")
        if self._last_submitted is not None and now - self._last_submitted < self.config.min_request_interval_s:
            return self._decision("NOT_SUBMITTED", "LOW_FREQUENCY_INTERVAL")
        frame_key = (frame.session_id, frame.target_generation, frame.frame_sequence,
                     frame.frame_id, frame.image_sha256)
        if frame_key == self._last_requested_frame:
            return self._decision("NOT_SUBMITTED", "FRAME_ALREADY_REQUESTED")
        if not frame.candidates:
            self._evidence = None
            return self._decision("UNCERTAIN", "NO_YOLO_BALLOON_CANDIDATE")
        self._serial += 1
        request = SemanticRequest(f"semantic-{self._serial:06d}", self.session_id,
                                  self.target_generation, self.description,
                                  self.description_sha256, now, frame)
        future = self._executor.submit(self.backend, request)
        self._pending = _Pending(request, future)
        self._evidence = None
        self._last_submitted, self._last_requested_frame = now, frame_key
        return self._decision("SUBMITTED", "ASYNC_IMAGE_TEXT_VERIFICATION",
                              request_binding=request.binding(),
                              request_mode="image_text", proposals_transmitted=False)

    def _unique_association(self, box: tuple[float, ...], frame: SemanticFrame,
                            minimum: float) -> tuple[Candidate | None, list[dict[str, Any]]]:
        scores = sorted(((box_iou(box, candidate.bbox_xyxy_pixels), candidate)
                         for candidate in frame.candidates), key=lambda item: item[0], reverse=True)
        audit = [{"candidate_id": candidate.candidate_id, "iou": score}
                 for score, candidate in scores]
        if not scores or scores[0][0] < minimum:
            return None, audit
        if len(scores) > 1 and scores[1][0] >= minimum:
            return None, audit
        if len(scores) > 1 and scores[0][0] - scores[1][0] < self.config.association_margin:
            return None, audit
        return scores[0][1], audit

    def poll(self, current_frame: SemanticFrame, *, now: float | None = None) -> dict[str, Any]:
        now = self._now(now)
        if self._closed:
            return self._decision("UNCERTAIN", "POLICY_CLOSED")
        pending = self._pending
        if pending:
            request = pending.request
            if not pending.invalidation_reason and now - request.submitted_monotonic_s > self.config.response_timeout_s:
                pending.invalidation_reason = "RESPONSE_TIMEOUT"
                self._evidence = None
                pending.future.cancel()
            if pending.invalidation_reason:
                reason = pending.invalidation_reason
                if pending.future.done():
                    # Consume terminal outcome without reading/reusing late response content.
                    try:
                        pending.future.result()
                    except BaseException:
                        pass
                    self._pending = None
                return self._decision("UNCERTAIN", reason, request_binding=request.binding())
        problem = self._frame_problem(current_frame, now)
        if problem:
            self._evidence = None
            return self._decision("UNCERTAIN", problem)
        if pending:
            request = pending.request
            if not pending.future.done():
                return self._decision("PENDING", "SEMANTIC_RESPONSE_NOT_READY", request_binding=request.binding())
            self._pending = None
            try:
                reply = pending.future.result()
            except BaseException as error:
                self._evidence = None
                return self._decision("UNCERTAIN", "SEMANTIC_BACKEND_ERROR", error_type=type(error).__name__,
                                      request_binding=request.binding())
            if not isinstance(reply, SemanticReply) or reply.binding != request.binding():
                return self._decision("UNCERTAIN", "RESPONSE_BINDING_MISMATCH", request_binding=request.binding())
            if reply.request_mode != "image_text" or reply.proposals_transmitted is not False:
                return self._decision("UNCERTAIN", "UNSUPPORTED_TRANSPORT_CONTRACT", request_binding=request.binding())
            try:
                completed = _number(reply.completed_monotonic_s, "reply completion")
                if not request.submitted_monotonic_s <= completed <= now:
                    raise ValueError("reply completion outside request/poll interval")
                if not isinstance(reply.result, dict):
                    raise ValueError("response is not a dictionary")
                boxes = reply.result.get("result_bbox_list", [])
                if not isinstance(boxes, list) or len(boxes) != 1:
                    return self._decision("UNCERTAIN", "NO_SINGLE_SEMANTIC_BOX", request_binding=request.binding())
                box = _box([boxes[0][key] for key in ("xmin", "ymin", "xmax", "ymax")],
                           request.frame.width, request.frame.height)
            except (KeyError, TypeError, ValueError):
                return self._decision("UNCERTAIN", "INVALID_SEMANTIC_RESULT", request_binding=request.binding())
            candidate, source_audit = self._unique_association(
                box, request.frame, self.config.source_association_min_iou)
            if candidate is None:
                return self._decision("UNCERTAIN", "SOURCE_BOX_ASSOCIATION_UNRESOLVED",
                                      request_binding=request.binding(), source_association=source_audit)
            trace = {"request_binding": request.binding(), "reply_completed_monotonic_s": completed,
                     "reply_binding_provenance": "CLIENT_SIDE_BINDING; NOT SERVER_ECHO",
                     "request_mode": reply.request_mode, "proposals_transmitted": False,
                     "server_proposal_ranking": False, "semantic_box_xyxy_pixels": list(box),
                     "source_candidate_id": candidate.candidate_id,
                     "source_association": source_audit}
            self._evidence = (request, candidate, trace)
        if self._evidence is None:
            return self._decision("UNCERTAIN", "NO_VALID_SEMANTIC_EVIDENCE")
        request, source_candidate, trace = self._evidence
        source_age = now - request.frame.readback_receipt_monotonic_s
        if source_age > self.config.semantic_max_age_s:
            self._evidence = None
            return self._decision("UNCERTAIN", "SEMANTIC_SOURCE_EXPIRED", **trace,
                                  semantic_source_age_proxy_s=source_age)
        if current_frame.frame_sequence <= request.frame.frame_sequence or (
                current_frame.readback_receipt_monotonic_s <= request.frame.readback_receipt_monotonic_s):
            return self._decision("UNCERTAIN", "NEWER_CURRENT_FRAME_REQUIRED", **trace)
        if (current_frame.width, current_frame.height) != (request.frame.width, request.frame.height):
            self._evidence = None
            return self._decision("UNCERTAIN", "CURRENT_FRAME_GEOMETRY_CHANGED", **trace)
        candidate, current_audit = self._unique_association(
            source_candidate.bbox_xyxy_pixels, current_frame, self.config.current_association_min_iou)
        if candidate is None:
            self._evidence = None
            return self._decision("UNCERTAIN", "CURRENT_BOX_ASSOCIATION_UNRESOLVED", **trace,
                                  current_association=current_audit)
        return self._decision("SEMANTIC_LOCAL_ASSOCIATION_VALID", "FRESH_CURRENT_YOLO_ASSOCIATION",
                              **trace, selected_candidate_id=candidate.candidate_id,
                              selected_candidate=candidate.to_dict(),
                              current_frame_id=current_frame.frame_id,
                              current_frame_sequence=current_frame.frame_sequence,
                              current_image_sha256=current_frame.image_sha256,
                              current_association=current_audit,
                              current_frame_age_proxy_s=now - current_frame.readback_receipt_monotonic_s,
                              semantic_source_age_proxy_s=source_age)

    def reset(self, *, session_id: str, target_generation: int, description: str,
              now: float | None = None) -> dict[str, Any]:
        self._now(now)
        session = _identifier(session_id, "session_id")
        generation = _integer(target_generation, "target_generation")
        description = _identifier(description, "description")
        if session == self.session_id and generation <= self.target_generation:
            raise ValueError("a same-session reset must advance target_generation")
        self.session_id, self.target_generation, self.description = session, generation, description
        self.description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
        self._evidence, self._last_seen_frame, self._last_requested_frame = None, None, None
        if self._pending:
            self._pending.invalidation_reason = "CONTEXT_RESET"
            self._pending.future.cancel()
        return self._decision("UNCERTAIN", "CONTEXT_RESET")

    def close(self) -> None:
        self._closed = True
        self._evidence = None
        if self._pending:
            self._pending.invalidation_reason = "POLICY_CLOSED"
            self._pending.future.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> BoundedSemanticPolicy:
        return self

    def __exit__(self, *unused: Any) -> None:
        self.close()


class YoloVlxBackend:
    """Public saved-image adapter; callers may inject fake detector/client.

    Default client reads existing environment variables internally. This adapter
    does not load or print .env, credentials, endpoint URLs, or exception strings.
    Warm-up must precede the timed readback; its elapsed time is returned for the
    experiment record. A warm-up image is not semantic evidence or a trial.
    """

    def __init__(self, *, weights: Path | str | None = None, detector: Any = None,
                 vlx_client: Any = None, device: str = "cpu", imgsz: int = 640,
                 confidence: float = 0.25, vlx_timeout_s: float = 3.0,
                 clock: Callable[[], float] = time.monotonic):
        if detector is None:
            from yolo_detector import YoloTargetDetector
            if weights is None:
                raise ValueError("explicit YOLO weights are required")
            detector = YoloTargetDetector(weights, confidence=confidence, imgsz=imgsz, device=device)
        if vlx_client is None:
            from vlx_api_client import VLXSeekAPIClient
            vlx_client = VLXSeekAPIClient(timeout=_number(vlx_timeout_s, "VLX timeout", positive=True), max_retries=1)
        if getattr(vlx_client, "request_mode", None) != "image_text" or getattr(
                vlx_client, "proposals_transmitted", None) is not False:
            raise ValueError("adapter requires the explicit cloud image+text contract")
        self.detector, self.vlx_client, self.clock = detector, vlx_client, clock
        self._warmed = False

    @staticmethod
    def _image(path: Path | str, expected_sha256: str | None = None) -> tuple[bytes, str, Any]:
        import cv2
        import numpy as np
        encoded = Path(path).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and digest != _digest(expected_sha256, "expected image digest"):
            raise ValueError("saved image hash changed")
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("image decode failed")
        return encoded, digest, bgr

    def warmup(self, image_path: Path | str) -> dict[str, Any]:
        _, digest, bgr = self._image(image_path)
        started = self.clock()
        self.detector.detect(bgr)
        elapsed = self.clock() - started
        self._warmed = True
        return {"status": "WARMED", "wall_seconds": elapsed, "image_sha256": digest,
                "semantic_api_called": False, "trial_count": 0}

    def detect_frame(self, image_path: Path | str, *, session_id: str, target_generation: int,
                     frame_id: str, frame_sequence: int, readback_receipt_monotonic_s: float,
                     expected_image_sha256: str | None = None) -> SemanticFrame:
        if not self._warmed:
            raise ValueError("YOLO must be explicitly warmed before timed frames")
        encoded, digest, bgr = self._image(image_path, expected_image_sha256)
        result = self.detector.detect(bgr)
        height, width = bgr.shape[:2]
        prefix = hashlib.sha256(json.dumps([session_id, target_generation, frame_id, frame_sequence,
                                           digest], ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
        candidates = []
        for candidate in result["candidates"]:
            if candidate.get("class_name") != "balloon":
                continue
            box = _box(candidate["bbox_xyxy_pixels"], width, height)
            candidates.append(Candidate(f"{prefix}-yolo-{len(candidates):03d}", box,
                                        float(candidate["confidence"]),
                                        str(candidate.get("source_class_name", "unknown")),
                                        bool(candidate.get("truncated", False))))
        if len(candidates) > 100:
            raise ValueError("more than 100 balloon proposals; explicit selection budget required")
        return SemanticFrame(session_id, target_generation, frame_id, frame_sequence, digest, width, height,
                             readback_receipt_monotonic_s, self.clock(), tuple(candidates), encoded,
                             json.dumps(result, ensure_ascii=False, allow_nan=False))

    def __call__(self, request: SemanticRequest) -> SemanticReply:
        from PIL import Image
        if (request.session_id, request.target_generation) != (
                request.frame.session_id, request.frame.target_generation) or (
                hashlib.sha256(request.description.encode("utf-8")).hexdigest()
                != request.description_sha256):
            raise ValueError("semantic request context differs from bound frame/description")
        if not request.frame.encoded_image:
            raise ValueError("bound image bytes are required by the public backend")
        if hashlib.sha256(request.frame.encoded_image).hexdigest() != request.frame.image_sha256:
            raise ValueError("request image digest mismatch")
        image = Image.open(io.BytesIO(request.frame.encoded_image)).convert("RGB")
        if image.size != (request.frame.width, request.frame.height):
            raise ValueError("decoded request dimensions differ")
        # Deliberately []: the current cloud endpoint does not send proposals.
        result = self.vlx_client.ground(image, [], request.description,
                                       lang="zh", temperature=0.0, max_new_tokens=128)
        # The request context is bound by this local worker, not echoed by VLX.
        return SemanticReply(request.binding(), self.clock(), result,
                             request_mode="image_text", proposals_transmitted=False)
