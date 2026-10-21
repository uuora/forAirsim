"""Pure, conservative temporal association for one semantically designated target.

An accepted VLX/YOLO decision anchors the requested instance. Subsequent boxes
are associated by geometry only. This module does not turn YOLO scores or IoU
into identity probabilities, prove appearance continuity, read scene truth, or
authorize motion. Loss breaks the association chain: recovery requires a new
semantic request and consecutive actual newer frames.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import time
from typing import Any

from t1_semantic_perception import Candidate, SemanticFrame, box_iou


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _sha(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA-256 binding")
    return value


@dataclass(frozen=True)
class TrackerConfig:
    current_frame_max_age_s: float = 0.5
    semantic_source_max_age_s: float = 2.0
    max_link_gap_s: float = 0.5
    max_unreviewed_track_s: float = 8.0
    association_min_iou: float = 0.3
    association_margin: float = 0.15
    initialization_confirm_frames: int = 2
    recovery_confirm_frames: int = 2

    def __post_init__(self) -> None:
        for key in ("current_frame_max_age_s", "semantic_source_max_age_s", "max_link_gap_s",
                    "max_unreviewed_track_s"):
            _finite(getattr(self, key), key, positive=True)
        if self.semantic_source_max_age_s < self.current_frame_max_age_s:
            raise ValueError("semantic window must allow current-frame freshness")
        if self.max_unreviewed_track_s < self.semantic_source_max_age_s:
            raise ValueError("tracking review window must cover semantic source freshness")
        for key in ("association_min_iou", "association_margin"):
            if not 0 < _finite(getattr(self, key), key) <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        for key in ("initialization_confirm_frames", "recovery_confirm_frames"):
            if _int(getattr(self, key), key) < 2:
                raise ValueError("confirmation requires at least two actual newer frames")


@dataclass(frozen=True)
class _Anchor:
    request_id: str
    source_receipt_s: float
    reply_completed_s: float
    semantic_expires_s: float
    immutable_evidence_sha256: str


class TemporalIdentityTracker:
    """Single-threaded local association, with explicit semantic recovery gates.

    ``now`` and frame receipts use the same host monotonic clock. They are not
    exposure times. A semantic certificate comes from BoundedSemanticPolicy;
    its client-side provenance is checked, but is not a server authentication
    or independent claim verification mechanism.
    """

    def __init__(self, session_id: str, target_generation: int, description: str,
                 config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._track_serial = 0
        self._closed = False
        self._set_context(session_id, target_generation, description)
        self._clear_context_state()

    def _set_context(self, session_id: str, target_generation: int, description: str) -> None:
        self.session_id = _name(session_id, "session_id")
        self.target_generation = _int(target_generation, "target_generation")
        self.description = _name(description, "description")
        self.description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()

    def _clear_context_state(self) -> None:
        self.state = "UNINITIALIZED"
        self._anchor: _Anchor | None = None
        self._candidate: Candidate | None = None
        self._link_receipt: float | None = None
        self._confirm_count = 0
        self._confirm_required = self.config.initialization_confirm_frames
        self._ever_anchored = False
        self._used_anchor_ids: set[str] = set()
        self._last_frame: SemanticFrame | None = None
        self._last_frame_fingerprint: str | None = None
        self._frame_ids: set[str] = set()
        self._last_now: float | None = None
        self._last_output_receipt: float | None = None
        self._track_id: str | None = None
        self._image_size: tuple[int, int] | None = None

    def _drop_chain(self, state: str = "LOST") -> None:
        self.state = state
        self._anchor = None
        self._candidate = None
        self._link_receipt = None
        self._confirm_count = 0
        self._track_id = None

    def _binding(self, frame: SemanticFrame | None) -> dict[str, Any] | None:
        if frame is None:
            return None
        return {"session_id": frame.session_id, "target_generation": frame.target_generation,
                "description_sha256": self.description_sha256,
                "frame_id": frame.frame_id, "frame_sequence": frame.frame_sequence,
                "image_sha256": frame.image_sha256,
                "readback_receipt_monotonic_s": frame.readback_receipt_monotonic_s,
                "candidates_ready_monotonic_s": frame.candidates_ready_monotonic_s}

    def _result(self, reason: str, frame: SemanticFrame | None, now: float | None,
                *, selected: Candidate | None = None, audit: list[dict] | None = None) -> dict[str, Any]:
        anchor = self._anchor
        review_remaining = (anchor.source_receipt_s + self.config.max_unreviewed_track_s - now
                            if anchor is not None and now is not None else None)
        return {"status": "TEMPORAL_LOCAL_ASSOCIATION_VALID" if selected else "NO_TARGET_OUTPUT",
                "state": self.state, "reason": reason,
                "session_id": self.session_id, "target_generation": self.target_generation,
                "description_sha256": self.description_sha256,
                "selected_candidate_id": selected.candidate_id if selected else None,
                "selected_candidate": selected.to_dict() if selected else None,
                "current_binding": self._binding(frame),
                "current_frame_id": frame.frame_id if frame else None,
                "current_frame_sequence": frame.frame_sequence if frame else None,
                "current_image_sha256": frame.image_sha256 if frame else None,
                "needs_semantic_review": (self.state in ("UNINITIALIZED", "LOST", "AMBIGUOUS") or
                    (self.state == "TRACKING" and review_remaining is not None and
                     review_remaining <= self.config.semantic_source_max_age_s)),
                "track_id": self._track_id, "semantic_anchor": anchor.request_id if anchor else None,
                "semantic_expires_monotonic_s": anchor.semantic_expires_s if anchor else None,
                "semantic_review_deadline_monotonic_s": (
                    anchor.source_receipt_s + self.config.max_unreviewed_track_s if anchor else None),
                "semantic_review_remaining_s": review_remaining,
                "gap_age_s": (max(0.0, now - self._last_output_receipt)
                              if now is not None and self._last_output_receipt is not None else None),
                "confirmation_count": self._confirm_count,
                "confirmation_required": self._confirm_required,
                "configured_timeouts": {key: value for key, value in asdict(self.config).items()
                                        if key.endswith("_s")},
                "association_audit": audit or [],
                "identity_probability_calibrated": False, "instance_identity_guaranteed": False,
                "appearance_continuity_verified": False, "flight_authorized": False,
                "semantic_binding_provenance": "CLIENT_SIDE_BINDING; NOT SERVER_ECHO",
                "time_interpretation": "host monotonic readback receipt; NOT exposure time"}

    def _frame_problem(self, frame: SemanticFrame, now: float) -> str | None:
        if (frame.session_id, frame.target_generation) != (self.session_id, self.target_generation):
            return "FRAME_CONTEXT_MISMATCH"
        if frame.candidates_ready_monotonic_s > now:
            return "FRAME_TIME_IN_FUTURE"
        if now - frame.readback_receipt_monotonic_s > self.config.current_frame_max_age_s:
            return "CURRENT_FRAME_EXPIRED"
        fingerprint = hashlib.sha256(json.dumps(frame.to_dict(), sort_keys=True).encode()).hexdigest()
        previous = self._last_frame
        if previous is not None:
            if frame.frame_sequence == previous.frame_sequence:
                return ("DUPLICATE_FRAME" if fingerprint == self._last_frame_fingerprint
                        else "FRAME_SEQUENCE_REUSED_WITH_DIFFERENT_BINDING")
            if (frame.frame_sequence < previous.frame_sequence or
                    frame.readback_receipt_monotonic_s <= previous.readback_receipt_monotonic_s):
                return "FRAME_SEQUENCE_OR_RECEIPT_REGRESSION"
        if frame.frame_id in self._frame_ids:
            return "FRAME_ID_REUSED"
        if self._image_size is not None and (frame.width, frame.height) != self._image_size:
            return "FRAME_GEOMETRY_CHANGED"
        self._last_frame = frame
        self._last_frame_fingerprint = fingerprint
        self._frame_ids.add(frame.frame_id)
        self._image_size = frame.width, frame.height
        return None

    def _unique_association(self, box: tuple, frame: SemanticFrame) -> tuple[Candidate | None, list[dict]]:
        scores = sorted(((box_iou(box, candidate.bbox_xyxy_pixels), candidate)
                         for candidate in frame.candidates), key=lambda pair: pair[0], reverse=True)
        audit = [{"candidate_id": candidate.candidate_id, "iou": score} for score, candidate in scores]
        if not scores or scores[0][0] < self.config.association_min_iou:
            return None, audit
        if len(scores) > 1 and (scores[1][0] >= self.config.association_min_iou or
                               scores[0][0] - scores[1][0] < self.config.association_margin):
            return None, audit
        return scores[0][1], audit

    def _validate_audit(self, rows: Any, selected_id: str, *, expected_ids: set[str] | None,
                        minimum: float) -> None:
        if not isinstance(rows, list) or not rows:
            raise ValueError("association audit missing")
        scores: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid audit row")
            key = _name(row.get("candidate_id"), "association candidate")
            score = _finite(row.get("iou"), "association IoU")
            if score > 1 or key in scores:
                raise ValueError("invalid or duplicate association")
            scores[key] = score
        if expected_ids is not None and set(scores) != expected_ids:
            raise ValueError("current audit does not cover current candidates")
        if selected_id not in scores or scores[selected_id] < minimum:
            raise ValueError("selected association below threshold")
        other = max((score for key, score in scores.items() if key != selected_id), default=0.0)
        if other >= minimum or scores[selected_id] - other < self.config.association_margin:
            raise ValueError("association is not unique")

    def _semantic_anchor(self, frame: SemanticFrame, decision: dict, now: float) -> tuple[_Anchor, Candidate]:
        if not isinstance(decision, dict) or decision.get("status") != "SEMANTIC_LOCAL_ASSOCIATION_VALID":
            raise ValueError("not a valid semantic association")
        for key, expected in {"session_id": self.session_id, "target_generation": self.target_generation,
                              "description_sha256": self.description_sha256,
                              "current_frame_id": frame.frame_id,
                              "current_frame_sequence": frame.frame_sequence,
                              "current_image_sha256": frame.image_sha256}.items():
            if decision.get(key) != expected or (key in ("target_generation", "current_frame_sequence")
                                                and isinstance(decision.get(key), bool)):
                raise ValueError("semantic current/context binding mismatch")
        if (decision.get("reply_binding_provenance") != "CLIENT_SIDE_BINDING; NOT SERVER_ECHO" or
                decision.get("request_mode") != "image_text" or decision.get("proposals_transmitted") is not False or
                decision.get("server_proposal_ranking") is not False or
                decision.get("identity_probability_calibrated") is not False or
                decision.get("instance_identity_guaranteed") is not False):
            raise ValueError("unsupported semantic provenance or identity claim")
        binding = decision.get("request_binding")
        if not isinstance(binding, dict):
            raise ValueError("source request binding missing")
        for key, expected in {"session_id": self.session_id, "target_generation": self.target_generation,
                              "description_sha256": self.description_sha256}.items():
            if binding.get(key) != expected or (key == "target_generation" and isinstance(binding.get(key), bool)):
                raise ValueError("semantic source context mismatch")
        request_id = _name(binding.get("request_id"), "request_id")
        _name(binding.get("frame_id"), "source frame_id")
        _sha(binding.get("image_sha256"))
        source_sequence = _int(binding.get("frame_sequence"), "source sequence")
        receipt = _finite(binding.get("readback_receipt_monotonic_s"), "source readback")
        submitted = _finite(binding.get("submitted_monotonic_s"), "request submission")
        completed = _finite(decision.get("reply_completed_monotonic_s"), "reply completion")
        if not receipt <= submitted <= completed <= now:
            raise ValueError("semantic clock ordering invalid")
        if source_sequence >= frame.frame_sequence or receipt >= frame.readback_receipt_monotonic_s:
            raise ValueError("semantic evidence requires actual newer current frame")
        if binding["frame_id"] == frame.frame_id:
            raise ValueError("source/current frame ID reused")
        semantic_box = decision.get("semantic_box_xyxy_pixels")
        if not isinstance(semantic_box, (tuple, list)) or len(semantic_box) != 4:
            raise ValueError("semantic source box missing")
        x1, y1, x2, y2 = (_finite(item, "semantic box") for item in semantic_box)
        if not 0 <= x1 < x2 <= frame.width or not 0 <= y1 < y2 <= frame.height:
            raise ValueError("semantic source box out of bounds")
        expires = receipt + self.config.semantic_source_max_age_s
        if now > expires:
            raise ValueError("semantic source expired")
        selected_id = _name(decision.get("selected_candidate_id"), "selected candidate")
        candidates = [candidate for candidate in frame.candidates if candidate.candidate_id == selected_id]
        if len(candidates) != 1:
            raise ValueError("selected current candidate absent")
        selected = candidates[0]
        certificate_box = decision.get("selected_candidate")
        if (not isinstance(certificate_box, dict) or certificate_box.get("candidate_id") != selected_id or
                certificate_box.get("bbox_xyxy_pixels") != list(selected.bbox_xyxy_pixels)):
            raise ValueError("selected candidate box binding mismatch")
        self._validate_audit(decision.get("current_association"), selected_id,
                             expected_ids={candidate.candidate_id for candidate in frame.candidates},
                             minimum=self.config.association_min_iou)
        self._validate_audit(decision.get("source_association"),
                             _name(decision.get("source_candidate_id"), "source candidate"),
                             expected_ids=None, minimum=0.5)
        immutable = {key: decision[key] for key in ("request_binding", "reply_completed_monotonic_s",
            "source_candidate_id", "semantic_box_xyxy_pixels", "source_association",
            "reply_binding_provenance", "request_mode", "proposals_transmitted", "server_proposal_ranking")}
        evidence_sha = hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
        return _Anchor(request_id, receipt, completed, expires, evidence_sha), selected

    def update(self, frame: SemanticFrame, semantic_decision: dict | None = None,
               now: float | None = None) -> dict[str, Any]:
        if not isinstance(frame, SemanticFrame):
            self._drop_chain()
            return self._result("INVALID_FRAME_TYPE", None, None)
        try:
            now_value = _finite(time.monotonic() if now is None else now, "tracker time")
        except ValueError:
            self._drop_chain()
            return self._result("INVALID_TRACKER_TIME", frame, None)
        if self._closed:
            return self._result("TRACKER_CLOSED", frame, now_value)
        if self._last_now is not None and now_value < self._last_now:
            self._drop_chain()
            return self._result("TRACKER_CLOCK_REGRESSION", frame, now_value)
        self._last_now = now_value
        problem = self._frame_problem(frame, now_value)
        if problem:
            if problem != "DUPLICATE_FRAME":
                self._drop_chain()
            return self._result(problem, frame, now_value)
        if not frame.candidates:
            self._drop_chain("LOST" if self._ever_anchored else "UNINITIALIZED")
            return self._result("NO_YOLO_CANDIDATES", frame, now_value)

        new_anchor: _Anchor | None = None
        semantic_candidate: Candidate | None = None
        if semantic_decision is not None and not isinstance(semantic_decision, dict):
            self._drop_chain()
            return self._result("INVALID_SEMANTIC_DECISION", frame, now_value)
        if semantic_decision and semantic_decision.get("status") == "SEMANTIC_LOCAL_ASSOCIATION_VALID":
            try:
                new_anchor, semantic_candidate = self._semantic_anchor(frame, semantic_decision, now_value)
            except (ValueError, TypeError, KeyError):
                self._drop_chain()
                return self._result("SEMANTIC_CERTIFICATE_INVALID_OR_EXPIRED", frame, now_value)

        # A fresh semantic result starts a new confirmation chain. Previously
        # used request IDs can continue their own intact chain, never revive it.
        if new_anchor is not None and (self._anchor is None or new_anchor.request_id != self._anchor.request_id):
            if new_anchor.request_id in self._used_anchor_ids:
                self._drop_chain()
                return self._result("SEMANTIC_ANCHOR_REUSED_AFTER_LOSS", frame, now_value)
            if (self.state == "TRACKING" and self._candidate is not None and self._link_receipt is not None
                    and frame.readback_receipt_monotonic_s - self._link_receipt <= self.config.max_link_gap_s):
                candidate, audit = self._unique_association(self._candidate.bbox_xyxy_pixels, frame)
                self._used_anchor_ids.add(new_anchor.request_id)
                if candidate is None or candidate.candidate_id != semantic_candidate.candidate_id:
                    self._drop_chain("AMBIGUOUS")
                    return self._result("SEMANTIC_REVIEW_CONFLICTS_WITH_CONTINUOUS_TRACK", frame, now_value, audit=audit)
                # A newly bound review of the same unique ongoing association
                # refreshes evidence without introducing an artificial loss.
                self._anchor, self._candidate = new_anchor, candidate
                self._link_receipt = frame.readback_receipt_monotonic_s
                self._last_output_receipt = frame.readback_receipt_monotonic_s
                return self._result("SEMANTIC_REVIEW_REFRESHED_CONTINUOUS_TRACK", frame, now_value,
                                    selected=candidate, audit=audit)
            self._confirm_required = (self.config.recovery_confirm_frames if self._ever_anchored
                                      else self.config.initialization_confirm_frames)
            self._ever_anchored = True
            self._used_anchor_ids.add(new_anchor.request_id)
            self._anchor, self._candidate = new_anchor, semantic_candidate
            self._link_receipt = frame.readback_receipt_monotonic_s
            self._confirm_count = 1
            self._track_id = None
            self.state = "RECOVERY_PENDING"
            return self._result("SEMANTIC_ANCHOR_AWAITING_NEWER_FRAME_CONFIRMATION", frame, now_value)

        if (new_anchor is not None and self._anchor is not None and
                new_anchor.immutable_evidence_sha256 != self._anchor.immutable_evidence_sha256):
            self._drop_chain()
            return self._result("SEMANTIC_REQUEST_ID_REBOUND", frame, now_value)

        if self._anchor is None or self._candidate is None or self._link_receipt is None:
            return self._result("FRESH_SEMANTIC_ANCHOR_REQUIRED", frame, now_value)
        if frame.readback_receipt_monotonic_s - self._link_receipt > self.config.max_link_gap_s:
            self._drop_chain()
            return self._result("TEMPORAL_LINK_GAP_EXCEEDED", frame, now_value)
        if now_value - self._anchor.source_receipt_s > self.config.max_unreviewed_track_s:
            self._drop_chain()
            return self._result("SEMANTIC_REVIEW_INTERVAL_EXCEEDED", frame, now_value)
        if self.state == "RECOVERY_PENDING" and now_value > self._anchor.semantic_expires_s:
            self._drop_chain()
            return self._result("SEMANTIC_ANCHOR_EXPIRED_DURING_CONFIRMATION", frame, now_value)
        candidate, audit = self._unique_association(self._candidate.bbox_xyxy_pixels, frame)
        if candidate is None:
            ambiguous = bool(audit and audit[0]["iou"] >= self.config.association_min_iou)
            self._drop_chain("AMBIGUOUS" if ambiguous else "LOST")
            return self._result("TEMPORAL_ASSOCIATION_AMBIGUOUS" if ambiguous else "TEMPORAL_ASSOCIATION_LOST",
                                frame, now_value, audit=audit)
        if semantic_candidate is not None and semantic_candidate.candidate_id != candidate.candidate_id:
            self._drop_chain("AMBIGUOUS")
            return self._result("SEMANTIC_AND_TEMPORAL_ASSOCIATION_CONFLICT", frame, now_value, audit=audit)
        self._candidate, self._link_receipt = candidate, frame.readback_receipt_monotonic_s
        if self.state == "RECOVERY_PENDING":
            self._confirm_count += 1
            if self._confirm_count < self._confirm_required:
                return self._result("AWAITING_MORE_ACTUAL_NEWER_FRAMES", frame, now_value, audit=audit)
            self._track_serial += 1
            self._track_id = f"track-{self._track_serial:06d}"
        self.state = "TRACKING"
        self._last_output_receipt = frame.readback_receipt_monotonic_s
        return self._result("UNIQUE_TEMPORAL_ASSOCIATION", frame, now_value, selected=candidate, audit=audit)

    def reset(self, *, session_id: str, target_generation: int, description: str,
              now: float | None = None) -> dict[str, Any]:
        next_session = _name(session_id, "session_id")
        next_generation = _int(target_generation, "target_generation")
        _name(description, "description")
        if next_session == self.session_id and next_generation <= self.target_generation:
            raise ValueError("same-session reset must advance target generation")
        now_value = _finite(time.monotonic() if now is None else now, "reset time")
        if self._last_now is not None and now_value < self._last_now:
            raise ValueError("reset monotonic clock regressed")
        if self._closed:
            raise ValueError("closed tracker cannot reset")
        self._set_context(next_session, next_generation, description)
        self._clear_context_state()
        self._last_now = now_value
        return self._result("CONTEXT_RESET", None, now_value)

    def close(self) -> None:
        self._closed = True
        self._drop_chain()
