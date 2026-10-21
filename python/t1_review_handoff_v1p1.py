"""Loss-aware review scheduling, preserving the frozen V5 evidence contract.

V1P1 distinguishes a broken evidence chain from rejection while already lost.
It never advances a loss watermark merely because an old reply is polled again.
All original freshness, proof, confirmation and conflict gates remain active.
"""
from __future__ import annotations

from t1_delayed_semantic_bridge import DelayedSemanticPolicy
from t1_review_handoff import ReviewHandoffIdentityTracker
from t1_semantic_perception import SemanticFrame, _number


class ReviewHandoffIdentityTrackerV1P1(ReviewHandoffIdentityTracker):
    """Frozen V5 tracker plus rejection/true-chain-loss distinction."""

    revision = "LOSS_AWARE_REVIEW_HANDOFF_V1P1"

    @property
    def loss_watermark(self):
        """Read-only snapshot of host receipts; no target truth enters it."""
        return {"source_readback_receipt_monotonic_s": self._loss_receipt,
            "observed_monotonic_s": self._loss_now}

    def _drop(self, frame, instant, reason, *, conflict=False):
        rejected_without_chain = (self._active is None and self._pending is None
            and not conflict and reason in {
                "SEMANTIC_ANCHOR_REUSED_AFTER_LOSS_OR_REPLACEMENT",
                "HANDOFF_FRESH_SOURCE_AFTER_LOSS_REQUIRED"})
        if rejected_without_chain:
            # There is no retained chain to break. The request remains banned
            # by the parent's used-ID ledger, and both source/submission gates
            # continue to use the last actual loss barrier without rebasing.
            return self._metadata(self._empty(frame, instant, reason),
                "REVIEW_REJECTED", "REJECTED", reason)
        return super()._drop(frame, instant, reason, conflict=conflict)


class LossAwareDelayedSemanticPolicy(DelayedSemanticPolicy):
    """Preflight new source receipt before original single-slot submission.

    ``tracker_supplier`` returns the current live V1P1 tracker (or None before
    initialization). A rejected preflight consumes neither request serial,
    cooldown nor transport slot. Time ordering still uses the original clock
    checks, and eligible requests go through the unchanged parent policy.
    """

    def __init__(self, backend, *, tracker_supplier=None, **kwargs):
        if tracker_supplier is not None and not callable(tracker_supplier):
            raise ValueError("INVALID_TRACKER_SUPPLIER")
        self.tracker_supplier = tracker_supplier
        super().__init__(backend, **kwargs)

    def request(self, frame, *, now=None):
        if self._closed or not isinstance(frame, SemanticFrame):
            return super().request(frame, now=now)
        instant = self._now(now)
        # Preserve the original context/current-frame/gap checks and journal
        # invalidation even when loss preflight refuses this source. The
        # parent's repeated observation of the same bound frame is idempotent.
        problem = self._observe(frame, instant)
        if problem:
            return self._decision("UNCERTAIN", problem)
        try:
            tracker = self.tracker_supplier() if self.tracker_supplier else None
            watermark = tracker.loss_watermark if tracker is not None else None
            if watermark is not None:
                if not isinstance(watermark, dict) or set(watermark) != {
                        "source_readback_receipt_monotonic_s", "observed_monotonic_s"}:
                    raise ValueError("INVALID_LOSS_WATERMARK")
                receipt = watermark["source_readback_receipt_monotonic_s"]
                observed = watermark["observed_monotonic_s"]
                if receipt is not None:
                    receipt = _number(receipt, "loss_readback")
                if observed is not None:
                    observed = _number(observed, "loss_observed")
                if ((receipt is not None and frame.readback_receipt_monotonic_s <= receipt)
                        or (observed is not None and instant <= observed)):
                    return self._decision("NOT_SUBMITTED", "FRESH_SOURCE_AFTER_LOSS_REQUIRED",
                        loss_watermark=dict(watermark), loss_aware_revision="V1P1")
        except (ValueError, TypeError, AttributeError):
            return self._decision("NOT_SUBMITTED", "LOSS_WATERMARK_UNAVAILABLE",
                loss_aware_revision="V1P1")
        return super().request(frame, now=instant)
