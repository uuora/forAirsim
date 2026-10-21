"""RGB-only specialist YOLO/VLX observer for the continuous ground pilot.

No simulator, truth, depth or motion APIs enter this module. A task with two
indistinguishable instances explicitly refuses semantic reanchoring after its
first lost chain: a positional description cannot recover invisible identity.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from t1_semantic_perception import BoundedSemanticPolicy, PolicyConfig, YoloVlxBackend, box_iou
from t1_temporal_identity import TemporalIdentityTracker, TrackerConfig

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_KEYS = frozenset({
    "session_id", "target_generation", "frame_id", "frame_sequence", "description",
    "rgb_path", "rgb_sha256", "image_size_wh", "readback_receipt_monotonic_s",
    "capture_requested_monotonic_s", "capture_completed_monotonic_s", "camera",
    "capture_metadata",
})
OBSERVABILITY = frozenset({"VISUAL_ATTRIBUTE", "UNRESOLVABLE_IDENTICAL"})
REVIEW_PERIOD_S = 3.0


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContinuousSemanticObserver:
    """Dependency-injected, single-main-thread observer; one cloud transport."""

    def __init__(self, backend, *, clock=time.monotonic,
                 policy_factory=BoundedSemanticPolicy,
                 tracker_factory=TemporalIdentityTracker,
                 review_period_s=REVIEW_PERIOD_S, model_receipt=None):
        if review_period_s < 2 or not review_period_s < float("inf"):
            raise ValueError("Review period must be finite and at least two seconds")
        self.backend, self.clock = backend, clock
        self.policy_factory, self.tracker_factory = policy_factory, tracker_factory
        self.review_period_s, self.model_receipt = review_period_s, model_receipt or {}
        self.policy = self.tracker = None
        self._closed = False
        self._ever_tracking = self._recovery_blocked = False
        self._ever_anchored = False
        self._chain_candidate = self._chain_receipt = None
        self._last_submitted = None
        self._last_tracker = None
        self._process_count = self._submitted_count = self._valid_count = 0
        self._observability_refusal_count = 0
        self.close_receipt = None

    def begin(self, session_id, description, warmup_rgb_path, *, recovery_observability):
        if self.policy is not None or self._closed:
            raise ValueError("Observer requires one new instance per simulator session")
        if recovery_observability not in OBSERVABILITY:
            raise ValueError("Explicit recovery observability declaration required")
        self.session_id, self.description = session_id, description
        self.recovery_observability = recovery_observability
        self.warmup_receipt = self.backend.warmup(warmup_rgb_path)
        self.policy = self.policy_factory(self.backend, session_id=session_id,
            target_generation=1, description=description, config=PolicyConfig(), clock=self.clock)
        self.tracker = self.tracker_factory(session_id, 1, description, TrackerConfig())
        return {"warmup": self.warmup_receipt, "model": self.model_receipt,
            "policy_config": asdict(PolicyConfig()), "tracker_config": asdict(TrackerConfig()),
            "review_period_s": self.review_period_s,
            "recovery_observability": recovery_observability,
            "input_scope": "RGB + self camera + constant task description only"}

    def _usage(self):
        client = getattr(self.backend, "vlx_client", None)
        raw = client.get_usage_summary() if client is not None else {}
        # Deliberately closed numeric allowlist; never serialize client/env/errors.
        return {key: value for key, value in raw.items() if key in {
            "total_requests", "total_prompt_tokens", "total_completion_tokens",
            "total_tokens", "total_took_ms", "average_took_ms"
        } and isinstance(value, (int, float)) and not isinstance(value, bool)}

    def process(self, observation: dict[str, Any]):
        if self.policy is None or self._closed:
            raise ValueError("Observer must be begun and open")
        if set(observation) != OBSERVATION_KEYS:
            raise ValueError("Observation differs from the RGB/self-camera closed allowlist")
        if (observation["session_id"], observation["target_generation"], observation["description"]) != (
                self.session_id, 1, self.description):
            raise ValueError("Task description/session/generation changed inside sequence")
        camera = observation["camera"]
        if set(camera) != {"scene_intrinsics", "pose_local_ned_m", "orientation_xyzw",
                          "world_from_local_translation_ned_m"}:
            raise ValueError("Unexpected self-camera fields")
        if set(observation["capture_metadata"]) != {
                "rgb_render_readback_timestamp_ns", "timestamp_scope"}:
            raise ValueError("Unexpected capture metadata fields")
        started = self.clock()
        frame = self.backend.detect_frame(observation["rgb_path"],
            session_id=self.session_id, target_generation=1,
            frame_id=observation["frame_id"], frame_sequence=observation["frame_sequence"],
            readback_receipt_monotonic_s=observation["readback_receipt_monotonic_s"],
            expected_image_sha256=observation["rgb_sha256"])
        if [frame.width, frame.height] != observation["image_size_wh"]:
            raise ValueError("Image geometry differs from bound readback")
        semantic = self.policy.poll(frame, now=self.clock())
        valid = semantic["status"] == "SEMANTIC_LOCAL_ASSOCIATION_VALID"
        self._valid_count += int(valid)
        # Inspect the existing chain before giving a fresh box permission to
        # start another chain on the same frame. This also catches a gap whose
        # first current frame happens to carry a new semantic reply.
        if (self.recovery_observability == "UNRESOLVABLE_IDENTICAL" and self._ever_anchored
                and not self._recovery_blocked):
            prior = self._last_tracker or {}
            selected = self._chain_candidate
            config = TrackerConfig()
            scores = sorted((box_iou(tuple(selected["bbox_xyxy_pixels"]), candidate.bbox_xyxy_pixels)
                for candidate in frame.candidates), reverse=True) if selected else []
            link_ok = (prior.get("state") in {"TRACKING", "RECOVERY_PENDING"} and scores
                and scores[0] >= config.association_min_iou
                and (len(scores) == 1 or (scores[1] < config.association_min_iou
                    and scores[0] - scores[1] >= config.association_margin))
                and self._chain_receipt is not None
                and frame.readback_receipt_monotonic_s - self._chain_receipt <= config.max_link_gap_s
                and self.clock() <= prior["semantic_review_deadline_monotonic_s"])
            if not link_ok:
                self._recovery_blocked = True
        allowed_semantic = semantic if valid and not self._recovery_blocked else None
        tracking = self.tracker.update(frame, semantic_decision=allowed_semantic, now=self.clock())
        # Observability is a task prior, not actor truth or a phase label. Loss
        # is derived only from the actual current observer association state.
        if (self.recovery_observability == "UNRESOLVABLE_IDENTICAL"
                and self._ever_anchored and tracking["state"] in {"LOST", "AMBIGUOUS"}):
            self._recovery_blocked = True
        self._ever_tracking |= tracking["state"] == "TRACKING"
        self._ever_anchored |= tracking.get("semantic_anchor") is not None
        if tracking.get("selected_candidate"):
            self._chain_candidate = tracking["selected_candidate"]
            self._chain_receipt = frame.readback_receipt_monotonic_s
        elif (tracking["state"] == "RECOVERY_PENDING" and allowed_semantic is not None
                and tracking.get("semantic_anchor") == allowed_semantic["request_binding"]["request_id"]):
            self._chain_candidate = allowed_semantic["selected_candidate"]
            self._chain_receipt = frame.readback_receipt_monotonic_s
        request = {"status": "NOT_REQUESTED", "reason": "REVIEW_NOT_DUE"}
        now = self.clock()
        review_due = (self._last_submitted is None
                      or now - self._last_submitted >= self.review_period_s)
        if self._recovery_blocked:
            self._observability_refusal_count += 1
            request = {"status": "NOT_REQUESTED", "reason": "IDENTITY_UNOBSERVABLE_AFTER_LOSS"}
        elif review_due:
            request = self.policy.request(frame, now=now)
            if request["status"] == "SUBMITTED":
                self._submitted_count += 1
                self._last_submitted = now
        self._process_count += 1
        self._last_tracker = tracking
        return {"status": "PROCESSED", "semantic_frame": frame.to_dict(),
            "semantic_poll": semantic, "semantic_request": request,
            "temporal_identity": tracking,
            "semantic_application": ("REFUSED_UNOBSERVABLE_RECOVERY" if self._recovery_blocked
                else "VALID_CERTIFICATE_APPLIED" if valid else "NO_VALID_CERTIFICATE"),
            "recovery_observability": self.recovery_observability,
            "unobservable_recovery_blocked": self._recovery_blocked,
            "process_started_host_monotonic_s": started,
            "process_completed_host_monotonic_s": self.clock(),
            "request_submission_count": self._submitted_count,
            "vlx_usage_http_response_counters": self._usage(),
            "request_mode": "image_text", "proposals_transmitted": False,
            "ground_only": True, "flight_authorized": False,
            "instance_identity_guaranteed": False,
            "identity_probability_calibrated": False}

    def close(self):
        if self._closed:
            return self.close_receipt
        started = self.clock()
        try:
            if self.policy is not None:
                self.policy.close()
        finally:
            if self.tracker is not None:
                self.tracker.close()
            self._closed = True
        self.close_receipt = {"status": "CLOSED", "processed_frames": self._process_count,
            "request_submissions": self._submitted_count,
            "valid_semantic_current_frame_certificates": self._valid_count,
            "unobservable_refusal_frames": self._observability_refusal_count,
            "final_tracker": self._last_tracker,
            "vlx_usage_http_response_counters": self._usage(),
            "model_binding": self.model_receipt,
            "policy_config": asdict(PolicyConfig()), "tracker_config": asdict(TrackerConfig()),
            "review_period_s": self.review_period_s,
            "close_wall_s": self.clock() - started,
            "ground_only": True, "flight_authorized": False}
        return self.close_receipt


def create_observer(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for field in ("weights", "training_receipt"):
        if digest(config[field]) != config[field+"_sha256"]:
            raise ValueError("Specialist artifact binding changed")
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    from dotenv import load_dotenv
    from specialist_balloon_detector import SpecialistBalloonDetector
    from vlx_api_client import VLXSeekAPIClient
    load_dotenv(ROOT / ".env")
    detector = SpecialistBalloonDetector(config["weights"], confidence=config["confidence"],
        imgsz=config["imgsz"], device=config["device"], nms_iou=config["nms_iou"])
    client = VLXSeekAPIClient(timeout=3, max_retries=1)
    backend = YoloVlxBackend(detector=detector, vlx_client=client)
    return ContinuousSemanticObserver(backend, model_receipt={
        "weights_config_path": str(config_path), "weights_config_sha256": digest(config_path),
        "weights_path": config["weights"], "weights_sha256": config["weights_sha256"],
        "training_receipt_sha256": config["training_receipt_sha256"],
        "confidence": config["confidence"], "nms_iou": config["nms_iou"],
        "imgsz": config["imgsz"], "device": config["device"],
        "model": client.model, "request_mode": "image_text", "proposals_transmitted": False})
