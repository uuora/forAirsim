"""Bounded delayed image-text reviews, joined through actual YOLO frame chains.

All times are host receipts. IoU continuity is not appearance verification,
instance identity proof, or motion authorization. Original source times are
never rebased. Existing direct-policy defaults are unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any

from t1_semantic_perception import (Candidate, SemanticFrame, SemanticRequest,
    SemanticReply, _box, _number, box_iou)
from t1_temporal_identity import TemporalIdentityTracker, TrackerConfig


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def chain_proof_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DelayedBridgeConfig:
    current_frame_max_age_s: float = .5
    max_source_age_s: float = 8.
    response_timeout_s: float = 10.
    max_link_gap_s: float = .5
    min_request_interval_s: float = 3.
    source_association_min_iou: float = .5
    association_min_iou: float = .3
    association_margin: float = .15
    max_buffered_frames: int = 128

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "max_buffered_frames":
                _number(value, key, positive=True)
        if (self.current_frame_max_age_s > .5 or self.max_source_age_s > 8.
                or self.response_timeout_s > 10. or self.max_link_gap_s > .5
                or self.source_association_min_iou < .5 or self.source_association_min_iou > 1
                or self.association_min_iou < .3 or self.association_min_iou > 1
                or self.association_margin < .15 or self.association_margin > 1
                or isinstance(self.max_buffered_frames, bool)
                or not isinstance(self.max_buffered_frames, int)
                or not 3 <= self.max_buffered_frames <= 128):
            raise ValueError("DELAYED_BRIDGE_CONFIG_EXCEEDS_BOUND")


def _frame_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("CHAIN_FRAME_NOT_OBJECT")
    size = data.get("image_size_wh")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("CHAIN_FRAME_SIZE_MISSING")
    candidates = tuple(Candidate(row["candidate_id"], tuple(row["bbox_xyxy_pixels"]),
        row["detector_score"], row["source_class_name"], row["truncated"])
        for row in data["candidates"])
    frame = SemanticFrame(data["session_id"], data["target_generation"], data["frame_id"],
        data["frame_sequence"], data["image_sha256"], size[0], size[1],
        data["readback_receipt_monotonic_s"], data["candidates_ready_monotonic_s"], candidates,
        detector_report_json=json.dumps(data["detector_report"], allow_nan=False))
    if frame.to_dict() != data:
        raise ValueError("CHAIN_FRAME_SERIALIZATION_MISMATCH")
    return frame


def _association(box, frame, minimum, margin):
    scores = sorted(((box_iou(tuple(box), row.bbox_xyxy_pixels), row)
        for row in frame.candidates), key=lambda item: (-item[0], item[1].candidate_id))
    audit = [{"candidate_id": row.candidate_id, "iou": score} for score, row in scores]
    if not scores or scores[0][0] < minimum:
        raise ValueError("CHAIN_NO_UNIQUE_MATCH")
    if len(scores) > 1 and (scores[1][0] >= minimum or scores[0][0]-scores[1][0] < margin):
        raise ValueError("CHAIN_ASSOCIATION_AMBIGUOUS")
    return scores[0][1], audit


def _hop(previous, selected, current, config):
    if (current.session_id, current.target_generation, current.width, current.height) != (
            previous.session_id, previous.target_generation, previous.width, previous.height):
        raise ValueError("CHAIN_CONTEXT_OR_DIMENSION_CHANGED")
    if current.frame_sequence != previous.frame_sequence+1 or current.frame_id == previous.frame_id:
        raise ValueError("CHAIN_SEQUENCE_NOT_CONTIGUOUS")
    gap = current.readback_receipt_monotonic_s-previous.readback_receipt_monotonic_s
    if not 0 < gap <= config.max_link_gap_s:
        raise ValueError("CHAIN_LINK_GAP_EXCEEDED")
    match, audit = _association(selected.bbox_xyxy_pixels, current,
        config.association_min_iou, config.association_margin)
    # Forward uniqueness alone can conceal two prior boxes merging into one.
    reverse, _ = _association(match.bbox_xyxy_pixels, previous,
        config.association_min_iou, config.association_margin)
    if reverse.candidate_id != selected.candidate_id:
        raise ValueError("CHAIN_MANY_TO_ONE_OR_REVERSE_CONFLICT")
    return match, audit


def _node(frame, selected, audit):
    data = frame.to_dict()
    return {"frame": data, "frame_sha256": chain_proof_sha256(data),
        "selected_candidate_id": selected.candidate_id,
        "selected_candidate_sha256": chain_proof_sha256(selected.to_dict()),
        "association_audit": audit}


def _binding(frame, description):
    return {"session_id": frame.session_id, "target_generation": frame.target_generation,
        "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
        "frame_id": frame.frame_id, "frame_sequence": frame.frame_sequence,
        "image_sha256": frame.image_sha256,
        "readback_receipt_monotonic_s": frame.readback_receipt_monotonic_s,
        "candidates_ready_monotonic_s": frame.candidates_ready_monotonic_s}


def _reply_digest(binding, completed, semantic_box):
    return chain_proof_sha256({"binding": binding, "completed_monotonic_s": completed,
        "request_mode": "image_text", "proposals_transmitted": False,
        "semantic_box_xyxy_pixels": semantic_box})


def verify_delayed_chain_proof(proof, *, current_frame, description, now,
        proof_sha256=None, require_confirmed=False):
    """Recompute all bound metadata and selected-route IoUs; raises ValueError.

    Digests are consistency checks, not authentication or decoded-image checks.
    ``semantic_reply_sha256`` binds the parsed, accepted single-box reply fields;
    ``reply_result_sha256`` separately records the complete original result hash.
    """
    try:
        if not isinstance(current_frame, SemanticFrame) or not isinstance(description, str) or not description.strip():
            raise ValueError("INVALID_CHAIN_CURRENT_CONTEXT")
        now = _number(now, "chain_check_time")
        if not isinstance(proof, dict) or proof.get("schema_version") != 1 or proof.get("contract") != "YOLO_CANDIDATE_CHAIN_V1":
            raise ValueError("INVALID_CHAIN_PROOF_SCHEMA")
        if set(proof) != {"schema_version", "contract", "configuration", "request_binding", "request_sha256",
                "reply_completed_monotonic_s", "request_mode", "proposals_transmitted", "semantic_box_xyxy_pixels",
                "semantic_reply_sha256", "reply_result_sha256", "source_candidate_id", "source_association", "nodes",
                "adoption", "post_adoption_frame_ids"} or isinstance(proof["schema_version"], bool):
            raise ValueError("CHAIN_PROOF_FIELDS_NOT_CLOSED")
        if proof_sha256 is not None and chain_proof_sha256(proof) != proof_sha256:
            raise ValueError("CHAIN_PROOF_SHA256_MISMATCH")
        config = DelayedBridgeConfig(**proof["configuration"])
        nodes = proof["nodes"]
        if not isinstance(nodes, list) or not 2 <= len(nodes) <= config.max_buffered_frames:
            raise ValueError("CHAIN_NODE_COUNT_INVALID")
        frames = [_frame_from_dict(node["frame"]) for node in nodes]
        source = frames[0]
        source_age = now-source.readback_receipt_monotonic_s
        if not 0 <= source_age <= config.max_source_age_s:
            raise ValueError("SOURCE_CHAIN_AGE_EXPIRED")
        if current_frame.candidates_ready_monotonic_s > now or now-current_frame.readback_receipt_monotonic_s > config.current_frame_max_age_s:
            raise ValueError("CURRENT_FRAME_EXPIRED_OR_FUTURE")
        if frames[-1].to_dict() != current_frame.to_dict():
            raise ValueError("CHAIN_CURRENT_FRAME_BINDING_MISMATCH")
        binding = proof["request_binding"]
        if set(binding) != {"request_id", "session_id", "target_generation", "description_sha256",
                "submitted_monotonic_s", "frame_id", "frame_sequence", "image_sha256", "readback_receipt_monotonic_s"}:
            raise ValueError("CHAIN_REQUEST_BINDING_FIELDS_INVALID")
        if not isinstance(binding["request_id"], str) or not binding["request_id"]:
            raise ValueError("CHAIN_REQUEST_ID_INVALID")
        if binding["description_sha256"] != hashlib.sha256(description.encode("utf-8")).hexdigest():
            raise ValueError("CHAIN_DESCRIPTION_BINDING_MISMATCH")
        for key in ("session_id", "target_generation", "frame_id", "frame_sequence", "image_sha256", "readback_receipt_monotonic_s"):
            if binding[key] != source.to_dict()[key] or isinstance(binding[key], bool):
                raise ValueError("CHAIN_ORIGINAL_SOURCE_BINDING_MISMATCH")
        if chain_proof_sha256(binding) != proof["request_sha256"]:
            raise ValueError("CHAIN_REQUEST_SHA256_MISMATCH")
        submitted = _number(binding["submitted_monotonic_s"], "chain_submission")
        completed = _number(proof["reply_completed_monotonic_s"], "chain_reply_completion")
        if not source.readback_receipt_monotonic_s <= submitted <= completed <= now:
            raise ValueError("CHAIN_REQUEST_REPLY_CLOCK_ORDER_INVALID")
        if completed-submitted > config.response_timeout_s:
            raise ValueError("RESPONSE_TIMEOUT")
        if completed-source.readback_receipt_monotonic_s > config.max_source_age_s:
            raise ValueError("SOURCE_CHAIN_AGE_EXPIRED_AT_COMPLETION")
        if proof.get("request_mode") != "image_text" or proof.get("proposals_transmitted") is not False:
            raise ValueError("CHAIN_UNSUPPORTED_TRANSPORT")
        box = list(_box(proof["semantic_box_xyxy_pixels"], source.width, source.height))
        if proof["semantic_reply_sha256"] != _reply_digest(binding, completed, box):
            raise ValueError("CHAIN_SEMANTIC_REPLY_SHA256_MISMATCH")
        result_hash = proof["reply_result_sha256"]
        if not isinstance(result_hash, str) or len(result_hash) != 64 or any(c not in "0123456789abcdef" for c in result_hash):
            raise ValueError("CHAIN_REPLY_RESULT_SHA256_INVALID")
        selected, source_audit = _association(box, source, config.source_association_min_iou, config.association_margin)
        if proof["source_candidate_id"] != selected.candidate_id or proof["source_association"] != source_audit:
            raise ValueError("CHAIN_SOURCE_ASSOCIATION_MISMATCH")
        seen_ids = set()
        for index, (node, frame) in enumerate(zip(nodes, frames)):
            if (frame.session_id, frame.target_generation, frame.width, frame.height) != (source.session_id, source.target_generation, source.width, source.height):
                raise ValueError("CHAIN_CONTEXT_OR_DIMENSION_CHANGED")
            if frame.frame_id in seen_ids or frame.candidates_ready_monotonic_s > now:
                raise ValueError("CHAIN_REUSED_ID_OR_FUTURE_FRAME")
            seen_ids.add(frame.frame_id)
            audit = source_audit
            if index:
                selected, audit = _hop(frames[index-1], selected, frame, config)
            if node != _node(frame, selected, audit):
                raise ValueError("CHAIN_NODE_CONTENT_OR_AUDIT_MISMATCH")
        adoption = proof["adoption"]
        if not isinstance(adoption, dict) or set(adoption) != {"accepted_monotonic_s", "accepted_frame_id",
                "accepted_frame_sequence", "accepted_readback_receipt_monotonic_s"}:
            raise ValueError("CHAIN_ADOPTION_FIELDS_NOT_CLOSED")
        if isinstance(adoption["accepted_frame_sequence"], bool):
            raise ValueError("CHAIN_ADOPTION_SEQUENCE_INVALID")
        matches = [i for i, frame in enumerate(frames) if frame.frame_id == adoption["accepted_frame_id"]]
        if len(matches) != 1 or matches[0] == 0:
            raise ValueError("CHAIN_ADOPTION_NODE_MISSING")
        adoption_index = matches[0]
        adopted = frames[adoption_index]
        accepted = _number(adoption["accepted_monotonic_s"], "chain_adoption_time")
        if (adoption["accepted_frame_sequence"] != adopted.frame_sequence
                or adoption["accepted_readback_receipt_monotonic_s"] != adopted.readback_receipt_monotonic_s
                or not max(completed, adopted.candidates_ready_monotonic_s) <= accepted <= now
                or accepted-adopted.readback_receipt_monotonic_s > config.current_frame_max_age_s
                or accepted-source.readback_receipt_monotonic_s > config.max_source_age_s):
            raise ValueError("CHAIN_ADOPTION_BINDING_OR_TIME_INVALID")
        newer = frames[adoption_index+1:]
        if any(frame.readback_receipt_monotonic_s <= accepted for frame in newer):
            raise ValueError("CHAIN_CONFIRMATION_NOT_CAPTURED_AFTER_ADOPTION")
        expected_confirmations = [frame.frame_id for frame in newer]
        if proof["post_adoption_frame_ids"] != expected_confirmations:
            raise ValueError("CHAIN_CONFIRMATION_LIST_MISMATCH")
        if require_confirmed and len(newer) < 2:
            raise ValueError("CHAIN_TWO_POST_ADOPTION_FRAMES_REQUIRED")
        return {"status": "CHAIN_PROOF_VERIFIED", "source_age_proxy_s": source_age,
            "source_readback_receipt_monotonic_s": source.readback_receipt_monotonic_s,
            "reply_completed_monotonic_s": completed, "selected_candidate_id": selected.candidate_id,
            "selected_candidate": selected.to_dict(), "node_count": len(nodes),
            "post_adoption_confirmation_count": len(newer),
            "semantic_review_deadline_monotonic_s": source.readback_receipt_monotonic_s+config.max_source_age_s,
            "image_content_hash_verified": False, "certificate_authentication_verified": False,
            "appearance_continuity_verified": False, "instance_identity_guaranteed": False}
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError("MALFORMED_CHAIN_PROOF") from error


def verify_delayed_selection(frame, certificate, description, now):
    if not isinstance(certificate, dict) or certificate.get("status") != "TEMPORAL_LOCAL_ASSOCIATION_VALID" or certificate.get("state") != "TRACKING":
        raise ValueError("CHAIN_SELECTION_NOT_TRACKING")
    audit = verify_delayed_chain_proof(certificate.get("delayed_chain_proof"),
        current_frame=frame, description=description, now=now,
        proof_sha256=certificate.get("delayed_chain_proof_sha256"), require_confirmed=True)
    proof = certificate["delayed_chain_proof"]
    if not isinstance(certificate.get("delayed_chain_proof_sha256"), str):
        raise ValueError("CHAIN_SELECTION_SHA256_MISSING")
    expected = {"session_id": frame.session_id, "target_generation": frame.target_generation,
        "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
        "current_frame_id": frame.frame_id, "current_frame_sequence": frame.frame_sequence,
        "current_image_sha256": frame.image_sha256, "current_binding": _binding(frame, description),
        "selected_candidate_id": audit["selected_candidate_id"], "selected_candidate": audit["selected_candidate"],
        "semantic_anchor": proof["request_binding"]["request_id"],
        "semantic_expires_monotonic_s": audit["semantic_review_deadline_monotonic_s"],
        "semantic_review_deadline_monotonic_s": audit["semantic_review_deadline_monotonic_s"]}
    if any(certificate.get(key) != value or isinstance(certificate.get(key), bool) for key, value in expected.items()):
        raise ValueError("CHAIN_SELECTION_BINDING_MISMATCH")
    if (isinstance(certificate.get("confirmation_required"), bool)
            or isinstance(certificate.get("confirmation_count"), bool)
            or certificate.get("confirmation_required") != 3 or certificate.get("confirmation_count") != 3
            or certificate.get("confirmation_count_semantics") != "capped_at_three_adoption_plus_two_newer_frames"):
        raise ValueError("CHAIN_SELECTION_CONFIRMATION_INCOMPLETE")
    if any(certificate.get(key) is not False for key in ("flight_authorized", "appearance_continuity_verified", "identity_probability_calibrated", "instance_identity_guaranteed")):
        raise ValueError("CHAIN_SELECTION_EXCEEDS_EVIDENCE_SCOPE")
    if certificate.get("semantic_binding_provenance") != "CLIENT_SIDE_BINDING; NOT SERVER_ECHO":
        raise ValueError("CHAIN_SELECTION_PROVENANCE_MISMATCH")
    return audit


@dataclass
class _Pending:
    request: SemanticRequest
    future: Any
    frames: list
    invalidation_reason: str | None = None
    terminal_recorded: bool = False


class DelayedSemanticPolicy:
    """Single caller thread, one occupied transport until the future settles."""

    def __init__(self, backend, *, session_id, target_generation, description,
            config=None, executor=None, clock=time.monotonic):
        self.config = config or DelayedBridgeConfig()
        if not isinstance(self.config, DelayedBridgeConfig):
            raise ValueError("INVALID_DELAYED_BRIDGE_CONFIG")
        self.backend, self.clock = backend, clock
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="t1-chain-vlx")
        self._pending = self._evidence = self._last_frame = None
        self._last_now = self._last_submitted = None
        self._serial = 0
        self._closed = False
        self._terminal_trace = []
        self._set_context(session_id, target_generation, description)

    def _set_context(self, session_id, generation, description):
        if not isinstance(session_id, str) or not session_id.strip() or not isinstance(description, str) or not description.strip():
            raise ValueError("INVALID_DELAYED_CONTEXT")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("INVALID_DELAYED_GENERATION")
        self.session_id, self.target_generation, self.description = session_id, generation, description
        self.description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()

    def _now(self, now):
        instant = _number(self.clock() if now is None else now, "bridge_now")
        if self._last_now is not None and instant < self._last_now:
            raise ValueError("DELAYED_CLOCK_REGRESSION")
        self._last_now = instant
        return instant

    def _decision(self, status, reason, **extra):
        return {"status": status, "reason": reason, "selected_candidate_id": None,
            "session_id": self.session_id, "target_generation": self.target_generation,
            "description_sha256": self.description_sha256,
            "transport_slot_occupied": self._pending is not None,
            "online_control_enabled": False, "identity_probability_calibrated": False,
            "instance_identity_guaranteed": False, "flight_authorized": False,
            "semantic_evidence_mode": "CHAIN_REVIEW", "terminal_request_trace": _copy(self._terminal_trace), **extra}

    def _invalidate(self, reason):
        self._evidence = None
        if self._pending and not self._pending.invalidation_reason:
            self._pending.invalidation_reason = reason

    def _observe(self, frame, now):
        if not isinstance(frame, SemanticFrame):
            self._invalidate("INVALID_FRAME_TYPE")
            return "INVALID_FRAME_TYPE"
        if (frame.session_id, frame.target_generation) != (self.session_id, self.target_generation):
            self._invalidate("FRAME_CONTEXT_MISMATCH")
            return "FRAME_CONTEXT_MISMATCH"
        if frame.candidates_ready_monotonic_s > now or now-frame.readback_receipt_monotonic_s > self.config.current_frame_max_age_s:
            self._invalidate("CURRENT_FRAME_EXPIRED_OR_FUTURE")
            return "CURRENT_FRAME_EXPIRED_OR_FUTURE"
        previous = self._last_frame
        if previous is not None:
            if frame.frame_sequence == previous.frame_sequence:
                if frame.to_dict() == previous.to_dict():
                    return None
                self._invalidate("FRAME_SEQUENCE_REBOUND")
                return "FRAME_SEQUENCE_REBOUND"
            if frame.frame_sequence != previous.frame_sequence+1 or frame.frame_id == previous.frame_id:
                self._invalidate("CHAIN_SEQUENCE_NOT_CONTIGUOUS")
                # The occupied request remains invalid. A later fresh request
                # may start at this observed frame after that slot settles.
                # Advancing this baseline never repairs the broken old route.
                if (frame.frame_sequence > previous.frame_sequence and frame.frame_id != previous.frame_id
                        and (frame.width, frame.height) == (previous.width, previous.height)):
                    self._last_frame = frame
                return "CHAIN_SEQUENCE_NOT_CONTIGUOUS"
            if (frame.width, frame.height) != (previous.width, previous.height):
                self._invalidate("CHAIN_DIMENSION_CHANGED")
                return "CHAIN_DIMENSION_CHANGED"
            gap = frame.readback_receipt_monotonic_s-previous.readback_receipt_monotonic_s
            if not 0 < gap <= self.config.max_link_gap_s:
                self._invalidate("CHAIN_LINK_GAP_EXCEEDED")
                self._last_frame = frame
                return "CHAIN_LINK_GAP_EXCEEDED"
        self._last_frame = frame
        if self._pending and not self._pending.invalidation_reason:
            self._pending.frames.append(frame)
            if len(self._pending.frames) > self.config.max_buffered_frames:
                self._invalidate("CHAIN_BUFFER_LIMIT_EXCEEDED")
        if self._evidence:
            self._evidence["frames"].append(frame)
            if len(self._evidence["frames"]) > self.config.max_buffered_frames:
                self._invalidate("CHAIN_BUFFER_LIMIT_EXCEEDED")
        return None

    def append(self, frame, *, now=None):
        instant = self._now(now)
        if self._closed:
            return self._decision("UNCERTAIN", "POLICY_CLOSED")
        problem = self._observe(frame, instant)
        return self._decision("UNCERTAIN" if problem else "APPENDED", problem or "ACTUAL_FRAME_JOURNALED")

    def request(self, frame, *, now=None):
        instant = self._now(now)
        if self._closed:
            return self._decision("UNCERTAIN", "POLICY_CLOSED")
        problem = self._observe(frame, instant)
        if problem:
            return self._decision("UNCERTAIN", problem)
        if self._pending:
            return self._decision("UNCERTAIN", "TRANSPORT_SLOT_OCCUPIED")
        if self._last_submitted is not None and instant-self._last_submitted < self.config.min_request_interval_s:
            return self._decision("NOT_SUBMITTED", "LOW_FREQUENCY_INTERVAL")
        if not frame.candidates:
            return self._decision("UNCERTAIN", "NO_YOLO_BALLOON_CANDIDATE")
        self._serial += 1
        request = SemanticRequest(f"chain-semantic-{self._serial:06d}", self.session_id,
            self.target_generation, self.description, self.description_sha256, instant, frame)
        self._pending = _Pending(request, self._executor.submit(self.backend, request), [frame])
        self._evidence = None
        self._last_submitted = instant
        return self._decision("SUBMITTED", "ASYNC_CHAIN_IMAGE_TEXT_VERIFICATION", request_binding=request.binding(),
            request_mode="image_text", proposals_transmitted=False)

    def _trace(self, pending, status, reason, now, **extra):
        if pending.terminal_recorded:
            return
        pending.terminal_recorded = True
        row = {"status": status, "reason": reason, "request_binding": pending.request.binding(),
            "terminal_observed_monotonic_s": now,
            "source_age_at_terminal_proxy_s": now-pending.request.frame.readback_receipt_monotonic_s,
            "journal_frame_count": len(pending.frames), "transport_done": bool(pending.future.done()), **extra}
        self._terminal_trace.append(_copy(row))

    def _proof(self, request, reply, frames, now, adoption=None):
        if not isinstance(reply, SemanticReply) or reply.binding != request.binding():
            raise ValueError("RESPONSE_BINDING_MISMATCH")
        if reply.request_mode != "image_text" or reply.proposals_transmitted is not False:
            raise ValueError("UNSUPPORTED_TRANSPORT_CONTRACT")
        completed = _number(reply.completed_monotonic_s, "reply_completion")
        if not request.submitted_monotonic_s <= completed <= now:
            raise ValueError("REPLY_CLOCK_ORDER_INVALID")
        if completed-request.submitted_monotonic_s > self.config.response_timeout_s:
            raise ValueError("RESPONSE_TIMEOUT")
        if now-request.frame.readback_receipt_monotonic_s > self.config.max_source_age_s:
            raise ValueError("SOURCE_CHAIN_AGE_EXPIRED")
        if not isinstance(reply.result, dict):
            raise ValueError("INVALID_SEMANTIC_RESULT")
        boxes = reply.result.get("result_bbox_list")
        if not isinstance(boxes, list) or len(boxes) != 1:
            raise ValueError("NO_SINGLE_SEMANTIC_BOX")
        box = list(_box([boxes[0][key] for key in ("xmin", "ymin", "xmax", "ymax")],
            request.frame.width, request.frame.height))
        if len(frames) < 2:
            raise ValueError("NEWER_CURRENT_FRAME_REQUIRED")
        selected, audit = _association(box, frames[0], self.config.source_association_min_iou, self.config.association_margin)
        source_id, source_audit = selected.candidate_id, audit
        nodes = [_node(frames[0], selected, audit)]
        for previous, current in zip(frames, frames[1:]):
            selected, audit = _hop(previous, selected, current, self.config)
            nodes.append(_node(current, selected, audit))
        current = frames[-1]
        adoption = adoption or {"accepted_monotonic_s": now, "accepted_frame_id": current.frame_id,
            "accepted_frame_sequence": current.frame_sequence,
            "accepted_readback_receipt_monotonic_s": current.readback_receipt_monotonic_s}
        adoption_index = next(i for i, node in enumerate(nodes) if node["frame"]["frame_id"] == adoption["accepted_frame_id"])
        proof = {"schema_version": 1, "contract": "YOLO_CANDIDATE_CHAIN_V1",
            "configuration": asdict(self.config), "request_binding": request.binding(),
            "request_sha256": chain_proof_sha256(request.binding()), "reply_completed_monotonic_s": completed,
            "request_mode": "image_text", "proposals_transmitted": False,
            "semantic_box_xyxy_pixels": box, "semantic_reply_sha256": _reply_digest(request.binding(), completed, box),
            "reply_result_sha256": chain_proof_sha256(reply.result), "source_candidate_id": source_id,
            "source_association": source_audit, "nodes": nodes, "adoption": adoption,
            "post_adoption_frame_ids": [node["frame"]["frame_id"] for node in nodes[adoption_index+1:]]}
        verify_delayed_chain_proof(proof, current_frame=current, description=self.description, now=now)
        return proof, selected, audit

    def poll(self, current_frame, *, now=None):
        instant = self._now(now)
        if self._closed:
            return self._decision("UNCERTAIN", "POLICY_CLOSED")
        problem = self._observe(current_frame, instant)
        pending = self._pending
        if pending:
            # Done first: worker completion and main-thread polling are different events.
            if pending.future.done():
                self._pending = None
                try:
                    reply = pending.future.result()
                except BaseException as error:
                    self._evidence = None
                    self._trace(pending, "REJECTED", pending.invalidation_reason or "SEMANTIC_BACKEND_ERROR", instant,
                        error_type=type(error).__name__)
                    return self._decision("UNCERTAIN", pending.invalidation_reason or "SEMANTIC_BACKEND_ERROR",
                        request_binding=pending.request.binding())
                completed = reply.completed_monotonic_s if isinstance(reply, SemanticReply) else None
                if pending.invalidation_reason:
                    extra = {"reply_completed_monotonic_s": completed} if isinstance(completed, (int, float)) and not isinstance(completed, bool) else {}
                    self._trace(pending, "REJECTED", pending.invalidation_reason, instant, **extra)
                    return self._decision("UNCERTAIN", pending.invalidation_reason, request_binding=pending.request.binding())
                try:
                    proof, _, _ = self._proof(pending.request, reply, pending.frames, instant)
                except (ValueError, TypeError, KeyError, StopIteration) as error:
                    self._evidence = None
                    reason = str(error) if isinstance(error, ValueError) else "INVALID_SEMANTIC_RESULT"
                    if not reason or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in reason):
                        reason = "INVALID_SEMANTIC_RESULT"
                    extra = {"reply_completed_monotonic_s": completed} if isinstance(completed, (int, float)) and not isinstance(completed, bool) else {}
                    self._trace(pending, "REJECTED", reason, instant, **extra)
                    return self._decision("UNCERTAIN", reason, request_binding=pending.request.binding())
                self._evidence = {"request": pending.request, "reply": reply, "frames": pending.frames,
                    "adoption": proof["adoption"]}
                self._trace(pending, "ACCEPTED_CHAIN", "SOURCE_TO_CURRENT_CHAIN_ACCEPTED", instant,
                    reply_completed_monotonic_s=completed,
                    transport_completion_elapsed_s=completed-pending.request.submitted_monotonic_s,
                    completed_to_poll_delay_s=instant-completed,
                    request_sha256=proof["request_sha256"], semantic_reply_sha256=proof["semantic_reply_sha256"],
                    reply_result_sha256=proof["reply_result_sha256"], delayed_chain_proof_sha256=chain_proof_sha256(proof))
            elif instant-pending.request.submitted_monotonic_s > self.config.response_timeout_s:
                pending.invalidation_reason = pending.invalidation_reason or "RESPONSE_TIMEOUT"
                pending.future.cancel()
                self._evidence = None
                self._trace(pending, "REJECTED", pending.invalidation_reason, instant)
                if pending.future.done():
                    self._pending = None
                return self._decision("UNCERTAIN", pending.invalidation_reason, request_binding=pending.request.binding())
            else:
                return self._decision("PENDING", pending.invalidation_reason or "SEMANTIC_RESPONSE_NOT_READY",
                    request_binding=pending.request.binding())
        if problem:
            return self._decision("UNCERTAIN", problem)
        if self._evidence is None:
            return self._decision("UNCERTAIN", "NO_VALID_SEMANTIC_EVIDENCE")
        evidence = self._evidence
        try:
            proof, selected, audit = self._proof(evidence["request"], evidence["reply"], evidence["frames"],
                instant, evidence["adoption"])
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            self._evidence = None
            reason = str(error) if isinstance(error, ValueError) else "INVALID_SEMANTIC_RESULT"
            if not reason or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in reason):
                reason = "INVALID_CHAIN_EVIDENCE"
            return self._decision("UNCERTAIN", reason)
        return self._decision("SEMANTIC_LOCAL_ASSOCIATION_VALID", "ACTUAL_SOURCE_TO_CURRENT_YOLO_CHAIN",
            request_binding=proof["request_binding"], reply_completed_monotonic_s=proof["reply_completed_monotonic_s"],
            reply_binding_provenance="CLIENT_SIDE_BINDING; NOT SERVER_ECHO", request_mode="image_text",
            proposals_transmitted=False, server_proposal_ranking=False,
            semantic_box_xyxy_pixels=proof["semantic_box_xyxy_pixels"], source_candidate_id=proof["source_candidate_id"],
            source_association=proof["source_association"], selected_candidate_id=selected.candidate_id,
            selected_candidate=selected.to_dict(), current_frame_id=current_frame.frame_id,
            current_frame_sequence=current_frame.frame_sequence, current_image_sha256=current_frame.image_sha256,
            current_association=audit, current_frame_age_proxy_s=instant-current_frame.readback_receipt_monotonic_s,
            semantic_source_age_proxy_s=instant-proof["request_binding"]["readback_receipt_monotonic_s"],
            delayed_chain_proof=proof, delayed_chain_proof_sha256=chain_proof_sha256(proof))

    def reset(self, *, session_id, target_generation, description, now=None):
        self._now(now)
        if session_id == self.session_id and target_generation <= self.target_generation:
            raise ValueError("RESET_REQUIRES_NEW_GENERATION")
        self._set_context(session_id, target_generation, description)
        self._invalidate("CONTEXT_RESET")
        self._last_frame = None
        self._last_submitted = None
        return self._decision("UNCERTAIN", "CONTEXT_RESET")

    def drain(self, timeout_s=12.):
        """Bounded post-close transport collection; caller must first land safely."""
        timeout_s = _number(timeout_s, "drain_timeout", positive=True)
        if timeout_s > 12. or not self._closed:
            raise ValueError("DRAIN_REQUIRES_CLOSED_POLICY_AND_BOUND_AT_MOST_TWELVE_SECONDS")
        pending = self._pending
        if pending is None:
            return {"status": "NO_PENDING_TRANSPORT", "terminal_request_trace": _copy(self._terminal_trace)}
        try:
            reply = pending.future.result(timeout=timeout_s)
        except FutureTimeout:
            return {"status": "DRAIN_TIMEOUT_TRANSPORT_STILL_RUNNING", "timeout_s": timeout_s,
                "request_binding": pending.request.binding(), "terminal_request_trace": _copy(self._terminal_trace)}
        except BaseException as error:
            self._terminal_trace.append({"status": "POST_CLOSE_TRANSPORT_ERROR", "reason": "BACKEND_ERROR",
                "request_binding": pending.request.binding(), "error_type": type(error).__name__})
        else:
            completed = reply.completed_monotonic_s if isinstance(reply, SemanticReply) else None
            valid = (isinstance(reply, SemanticReply) and reply.binding == pending.request.binding()
                and reply.request_mode == "image_text" and reply.proposals_transmitted is False)
            try:
                completed = _number(completed, "post_close_completion")
                valid = valid and pending.request.submitted_monotonic_s <= completed <= self.clock()
            except ValueError:
                valid = False
                completed = None
            row = {"status": "POST_CLOSE_TRANSPORT_SETTLED", "reason": "CLOSED_RESPONSE_NOT_REUSED",
                "request_binding": pending.request.binding(), "reply_contract_binding_valid": valid,
                "reply_completed_monotonic_s": completed}
            if valid:
                row["transport_completion_elapsed_s"] = completed-pending.request.submitted_monotonic_s
                try:
                    row["reply_result_sha256"] = chain_proof_sha256(reply.result)
                except (TypeError, ValueError):
                    row["reply_result_finite_json"] = False
            self._terminal_trace.append(_copy(row))
        self._pending = None
        return {"status": "TRANSPORT_DRAINED_NO_RESPONSE_REUSE", "terminal_request_trace": _copy(self._terminal_trace)}

    def close(self, wait=False):
        if self._closed:
            return self.close_receipt
        instant = self._now(None)
        self._closed = True
        self._invalidate("POLICY_CLOSED")
        if self._pending:
            self._pending.future.cancel()
            self._trace(self._pending, "CLOSED_UNCONSUMED", self._pending.invalidation_reason, instant)
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.close_receipt = {"status": "CLOSED", "request_submissions": self._serial,
            "terminal_request_trace": _copy(self._terminal_trace),
            "transport_still_running": bool(self._pending and not self._pending.future.done()),
            "flight_authorized": False}
        if wait:
            drained = self.drain(12.)
            self.close_receipt.update(drain=drained, terminal_request_trace=_copy(self._terminal_trace),
                transport_still_running=bool(self._pending and not self._pending.future.done()))
        return self.close_receipt


class DelayedChainIdentityTracker:
    """Fresh adoption plus two subsequent actual frames, with retained proof."""

    def __init__(self, session_id, target_generation, description, config=None):
        self.config = config or TrackerConfig(semantic_source_max_age_s=8., max_unreviewed_track_s=8.,
            initialization_confirm_frames=3, recovery_confirm_frames=3)
        if (self.config.current_frame_max_age_s != .5 or self.config.semantic_source_max_age_s != 8.
                or self.config.max_unreviewed_track_s != 8. or self.config.max_link_gap_s != .5
                or self.config.initialization_confirm_frames != 3 or self.config.recovery_confirm_frames != 3):
            raise ValueError("DELAYED_TRACKER_CONTRACT_CONFIG_MISMATCH")
        self.session_id, self.target_generation, self.description = session_id, target_generation, description
        self._inner = TemporalIdentityTracker(session_id, target_generation, description, self.config)
        self._proof = None
        self._used_request_ids = set()
        self._track_serial = 0
        self._closed = False

    @property
    def state(self):
        return self._inner.state

    def _reject(self, frame, now, reason):
        self._proof = None
        self._inner._drop_chain()
        return self._inner._result(reason, frame, now)

    def update(self, frame, semantic_decision=None, now=None):
        instant = _number(time.monotonic() if now is None else now, "delayed_tracker_now")
        if self._closed:
            return self._reject(frame, instant, "TRACKER_CLOSED")
        try:
            new = isinstance(semantic_decision, dict) and semantic_decision.get("status") == "SEMANTIC_LOCAL_ASSOCIATION_VALID"
            if new:
                proof = semantic_decision["delayed_chain_proof"]
                verify_delayed_chain_proof(proof, current_frame=frame, description=self.description, now=instant,
                    proof_sha256=semantic_decision["delayed_chain_proof_sha256"])
                request_id = proof["request_binding"]["request_id"]
                prior_id = self._proof["request_binding"]["request_id"] if self._proof else None
                if request_id != prior_id:
                    if request_id in self._used_request_ids:
                        raise ValueError("SEMANTIC_ANCHOR_REUSED_AFTER_LOSS")
                    self._used_request_ids.add(request_id)
                    # New review adoption also requires two new frames; do not use
                    # the old tracker's immediate continuous-review shortcut.
                    self._inner = TemporalIdentityTracker(self.session_id, self.target_generation, self.description, self.config)
                    self._track_serial += 1
                    self._proof = _copy(proof)
                else:
                    if proof["adoption"] != self._proof["adoption"]:
                        raise ValueError("CHAIN_ADOPTION_REBOUND")
                    self._proof = _copy(proof)
            elif self._proof is not None:
                previous = _frame_from_dict(self._proof["nodes"][-1]["frame"])
                if previous.to_dict() != frame.to_dict():
                    previous_selected = next(candidate for candidate in previous.candidates
                        if candidate.candidate_id == self._proof["nodes"][-1]["selected_candidate_id"])
                    selected, audit = _hop(previous, previous_selected, frame,
                        DelayedBridgeConfig(**self._proof["configuration"]))
                    self._proof["nodes"].append(_node(frame, selected, audit))
                    self._proof["post_adoption_frame_ids"].append(frame.frame_id)
                verify_delayed_chain_proof(self._proof, current_frame=frame, description=self.description, now=instant)
            result = self._inner.update(frame, semantic_decision=semantic_decision if new else None, now=instant)
            if result["state"] in {"LOST", "AMBIGUOUS"}:
                self._proof = None
            if self._proof is not None:
                result.update(delayed_chain_proof=_copy(self._proof),
                    delayed_chain_proof_sha256=chain_proof_sha256(self._proof), semantic_evidence_mode="CHAIN_REVIEW",
                    confirmation_count_semantics="capped_at_three_adoption_plus_two_newer_frames")
                if result["status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
                    result["track_id"] = f"chain-track-{self._track_serial:06d}"
                    verify_delayed_selection(frame, result, self.description, instant)
            return result
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            reason = str(error) if isinstance(error, ValueError) else "DELAYED_CHAIN_PROOF_INVALID"
            if not reason or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in reason):
                reason = "DELAYED_CHAIN_PROOF_INVALID"
            return self._reject(frame, instant, reason)

    def reset(self, *, session_id, target_generation, description, now=None):
        result = self._inner.reset(session_id=session_id, target_generation=target_generation,
            description=description, now=now)
        self.session_id, self.target_generation, self.description = session_id, target_generation, description
        self._proof = None
        self._used_request_ids.clear()
        return result

    def close(self):
        self._closed = True
        self._proof = None
        self._inner.close()
