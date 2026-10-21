"""Independent confirmation of a new review while bounded old evidence runs.

This module consumes RGB/YOLO chain certificates only. Each output retains the
original proof and source deadline checked by the existing V3 verifier. A new
review never borrows old confirmations or extends an old source deadline.
"""
from __future__ import annotations

from copy import deepcopy
import time

from t1_delayed_semantic_bridge import (DelayedChainIdentityTracker,
    verify_delayed_chain_proof, verify_delayed_selection)
from t1_semantic_perception import SemanticFrame, _number
from t1_temporal_identity import TemporalIdentityTracker, TrackerConfig


HANDOFF_EVENTS = frozenset({"NO_NEW_REVIEW", "INITIAL_ADOPTION", "INITIAL_CONFIRMED",
    "REVIEW_PENDING", "REVIEW_ADVANCED", "REVIEW_COMMITTED", "REVIEW_REPLACED",
    "REVIEW_REJECTED", "REVIEW_CONFLICT", "CHAIN_LOST", "DUPLICATE_FRAME",
    "CONTEXT_RESET", "TRACKER_CLOSED"})
HANDOFF_STATUSES = frozenset({"IDLE", "ACQUIRING", "PENDING", "COMMITTED",
    "REJECTED", "CONFLICT", "LOST", "RESET", "CLOSED"})


def _closed_reason(error):
    reason = str(error) if isinstance(error, ValueError) else "HANDOFF_CERTIFICATE_INVALID"
    if not reason or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in reason):
        return "HANDOFF_CERTIFICATE_INVALID"
    return reason


class ReviewHandoffIdentityTracker:
    """Two independent confirmation chains with an atomic same-candidate swap.

    Duplicate frames produce no output and no confirmation. A sequence gap,
    failed association, context change or conflict drops both chains. Recovery
    then requires a never-used request whose source and submission follow the
    observed loss. Backend rejection can leave the old valid chain running,
    but cancels a pending replacement and does not rebase either deadline.
    """

    def __init__(self, session_id, target_generation, description, config=None):
        self.config = config or TrackerConfig(semantic_source_max_age_s=8.,
            max_unreviewed_track_s=8., initialization_confirm_frames=3,
            recovery_confirm_frames=3)
        # Validate the exact existing delayed contract and context now.
        contract = DelayedChainIdentityTracker(session_id, target_generation, description,
            self.config)
        contract.close()
        self.session_id, self.target_generation, self.description = (
            session_id, target_generation, description)
        self._gate = TemporalIdentityTracker(session_id, target_generation, description,
            self.config)
        self._active = self._pending = None
        self._used_request_ids = set()
        self._last_now = None
        self._loss_receipt = self._loss_now = None
        self._closed = False
        self._track_serial = 0
        self._active_track_id = None

    @property
    def state(self):
        track = self._active or self._pending
        return track.state if track is not None else self._gate.state

    def _new_tracker(self):
        return DelayedChainIdentityTracker(self.session_id, self.target_generation,
            self.description, self.config)

    @staticmethod
    def _request_id(track):
        return track._proof["request_binding"]["request_id"] if track and track._proof else None

    def _metadata(self, result, event, status, reason, *, count=None):
        if event not in HANDOFF_EVENTS or status not in HANDOFF_STATUSES:
            raise ValueError("HANDOFF_METADATA_ENUM_INVALID")
        if count is None:
            track = self._pending or self._active
            count = min(3, 1 + len(track._proof["post_adoption_frame_ids"])) if (
                track and track._proof) else 0
        result["review_handoff"] = {"event": event, "status": status,
            "active_request_id": self._request_id(self._active),
            "pending_request_id": self._request_id(self._pending),
            "confirmation_count": count, "reason": reason}
        return result

    def _empty(self, frame, instant, reason):
        self._gate.state = self.state
        return self._gate._result(reason, frame if isinstance(frame, SemanticFrame) else None,
            instant)

    def _clear_pending(self):
        if self._pending is not None:
            self._pending.close()
        self._pending = None

    def _drop(self, frame, instant, reason, *, conflict=False):
        if self._active is not None:
            self._active.close()
        self._active = None
        self._active_track_id = None
        self._clear_pending()
        self._gate.state = "AMBIGUOUS" if conflict else "LOST"
        if isinstance(frame, SemanticFrame):
            receipt = frame.readback_receipt_monotonic_s
            self._loss_receipt = max(receipt, self._loss_receipt or 0.)
        if instant is not None:
            self._loss_now = max(instant, self._loss_now or 0.)
        return self._metadata(self._empty(frame, instant, reason),
            "REVIEW_CONFLICT" if conflict else "CHAIN_LOST",
            "CONFLICT" if conflict else "LOST", reason)

    def _validate_semantic(self, frame, decision, instant):
        proof = decision["delayed_chain_proof"]
        audit = verify_delayed_chain_proof(proof, current_frame=frame,
            description=self.description, now=instant,
            proof_sha256=decision["delayed_chain_proof_sha256"])
        # Existing semantic validator checks closed provenance, selected box,
        # source/current binding, association audits and original clocks.
        self._gate._semantic_anchor(frame, decision, instant)
        return proof, audit

    @staticmethod
    def _same_proof(track, proof):
        # Both proofs must be the same independently extended actual journal.
        if track is None or track._proof != proof:
            raise ValueError("HANDOFF_EXISTING_REQUEST_PROOF_REBOUND")

    def update(self, frame, semantic_decision=None, now=None):
        try:
            instant = _number(time.monotonic() if now is None else now, "handoff_now")
        except ValueError:
            return self._drop(frame, None, "INVALID_TRACKER_TIME")
        if self._closed:
            return self._metadata(self._empty(frame, instant, "TRACKER_CLOSED"),
                "TRACKER_CLOSED", "CLOSED", "TRACKER_CLOSED")
        if self._last_now is not None and instant < self._last_now:
            return self._drop(frame, instant, "TRACKER_CLOCK_REGRESSION")
        self._last_now = self._gate._last_now = instant
        if not isinstance(frame, SemanticFrame):
            return self._drop(frame, instant, "INVALID_FRAME_TYPE")
        previous = self._gate._last_frame
        problem = self._gate._frame_problem(frame, instant)
        if problem == "DUPLICATE_FRAME":
            return self._metadata(self._empty(frame, instant, problem),
                "DUPLICATE_FRAME", "PENDING" if self._pending else "IDLE", problem)
        if problem:
            return self._drop(frame, instant, problem)
        if previous is not None and (frame.frame_sequence != previous.frame_sequence + 1
                or frame.readback_receipt_monotonic_s - previous.readback_receipt_monotonic_s
                > self.config.max_link_gap_s):
            return self._drop(frame, instant, "HANDOFF_FRAME_CHAIN_GAP")
        if not frame.candidates:
            return self._drop(frame, instant, "NO_YOLO_CANDIDATES")
        if semantic_decision is not None and not isinstance(semantic_decision, dict):
            return self._drop(frame, instant, "INVALID_SEMANTIC_DECISION")

        # Extend each retained proof to this actual current frame independently.
        active_result = self._active.update(frame, now=instant) if self._active else None
        if active_result is not None and active_result["status"] != "TEMPORAL_LOCAL_ASSOCIATION_VALID":
            return self._drop(frame, instant, active_result["reason"])
        pending_result = self._pending.update(frame, now=instant) if self._pending else None
        if pending_result is not None and pending_result["state"] in {"LOST", "AMBIGUOUS"}:
            return self._drop(frame, instant, pending_result["reason"])

        new_valid = isinstance(semantic_decision, dict) and semantic_decision.get(
            "status") == "SEMANTIC_LOCAL_ASSOCIATION_VALID"
        event, status, reason = "NO_NEW_REVIEW", "IDLE", "BOUNDED_CURRENT_ANCHOR_CONTINUES"
        try:
            if new_valid:
                proof, audit = self._validate_semantic(frame, semantic_decision, instant)
                request_id = proof["request_binding"]["request_id"]
                if request_id == self._request_id(self._active):
                    self._same_proof(self._active, proof)
                elif request_id == self._request_id(self._pending):
                    self._same_proof(self._pending, proof)
                else:
                    if request_id in self._used_request_ids:
                        raise ValueError("SEMANTIC_ANCHOR_REUSED_AFTER_LOSS_OR_REPLACEMENT")
                    self._used_request_ids.add(request_id)
                    binding = proof["request_binding"]
                    if self._loss_receipt is not None and (binding["readback_receipt_monotonic_s"]
                            <= self._loss_receipt or (self._loss_now is not None and
                                binding["submitted_monotonic_s"] <= self._loss_now)):
                        raise ValueError("HANDOFF_FRESH_SOURCE_AFTER_LOSS_REQUIRED")
                    if active_result is not None and audit["selected_candidate_id"] != active_result[
                            "selected_candidate_id"]:
                        return self._drop(frame, instant,
                            "SEMANTIC_REVIEW_CONFLICTS_WITH_CONTINUOUS_TRACK", conflict=True)
                    replaced = self._pending is not None
                    self._clear_pending()
                    self._pending = self._new_tracker()
                    pending_result = self._pending.update(frame, semantic_decision, now=instant)
                    if pending_result["state"] in {"LOST", "AMBIGUOUS"}:
                        raise ValueError("HANDOFF_NEW_REVIEW_REJECTED")
                    event = "REVIEW_REPLACED" if replaced else (
                        "REVIEW_PENDING" if self._active else "INITIAL_ADOPTION")
                    status = "PENDING" if self._active else "ACQUIRING"
                    reason = "NEW_REVIEW_REQUIRES_ADOPTION_PLUS_TWO_ACTUAL_NEWER_FRAMES"
            elif semantic_decision is not None and (semantic_decision.get("status") == "REJECTED"
                    or semantic_decision.get("reason") in {"SEMANTIC_BACKEND_ERROR",
                        "SOURCE_ATTRIBUTE_REJECTED", "RESPONSE_TIMEOUT"}):
                self._clear_pending()
                pending_result = None
                event, status, reason = "REVIEW_REJECTED", "REJECTED", "NEW_REVIEW_REJECTED_OLD_DEADLINE_RETAINED"

            if self._pending is not None and pending_result is not None:
                if self._active is not None and pending_result["status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID" and (
                        pending_result["selected_candidate_id"] != active_result["selected_candidate_id"]):
                    return self._drop(frame, instant,
                        "PENDING_REVIEW_CONFLICTS_WITH_CONTINUOUS_TRACK", conflict=True)
                if pending_result["status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
                    verify_delayed_selection(frame, pending_result, self.description, instant)
                    was_active = self._active is not None
                    if self._active is not None:
                        self._active.close()
                    self._active, self._pending = self._pending, None
                    self._track_serial += 1
                    self._active_track_id = f"handoff-track-{self._track_serial:06d}"
                    active_result = pending_result
                    event = "REVIEW_COMMITTED" if was_active else "INITIAL_CONFIRMED"
                    status, reason = "COMMITTED", "INDEPENDENT_NEW_REVIEW_CONFIRMED_AT_CURRENT_FRAME"
                elif event == "NO_NEW_REVIEW":
                    event, status, reason = "REVIEW_ADVANCED", "PENDING" if self._active else "ACQUIRING", "AWAITING_TWO_ACTUAL_NEWER_FRAMES"
            result = active_result or pending_result or self._empty(frame, instant,
                "FRESH_SEMANTIC_ANCHOR_REQUIRED")
            result = deepcopy(result)
            if result["status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
                result["track_id"] = self._active_track_id
                verify_delayed_selection(frame, result, self.description, instant)
            return self._metadata(result, event, status, reason)
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            return self._drop(frame, instant, _closed_reason(error))

    def reset(self, *, session_id, target_generation, description, now=None):
        # Validate reset before changing either live chain.
        result = self._gate.reset(session_id=session_id, target_generation=target_generation,
            description=description, now=now)
        if self._active is not None:
            self._active.close()
        self._active = None
        self._active_track_id = None
        self._clear_pending()
        self.session_id, self.target_generation, self.description = (
            session_id, target_generation, description)
        self._used_request_ids.clear()
        self._last_now = self._gate._last_now
        self._loss_receipt = self._loss_now = None
        return self._metadata(result, "CONTEXT_RESET", "RESET", "CONTEXT_RESET")

    def close(self):
        if self._active is not None:
            self._active.close()
        self._active = None
        self._active_track_id = None
        self._clear_pending()
        self._closed = True
        self._gate.close()
