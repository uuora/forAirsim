"""Real policy/tracker loops with manual Futures for observed V5 failure."""
from dataclasses import replace
import unittest

import test_t1_review_handoff as handoff_fixtures
import test_t1_candidate_surface_v3 as fixtures
from t1_delayed_semantic_bridge import DelayedBridgeConfig
from t1_review_handoff_v1p1 import (ReviewHandoffIdentityTrackerV1P1,
    LossAwareDelayedSemanticPolicy)


class LossAwareTests(unittest.TestCase):
    # Reuse controlled-future fixture helpers, not the old implementation.
    frame = handoff_fixtures.HandoffTests.frame
    complete = handoff_fixtures.HandoffTests.complete
    step = handoff_fixtures.HandoffTests.step
    initial = handoff_fixtures.HandoffTests.initial
    advance = handoff_fixtures.HandoffTests.advance
    review = handoff_fixtures.HandoffTests.review
    submit = handoff_fixtures.HandoffTests.submit

    def setUp(self):
        self.executor = fixtures.ManualExecutor()
        self.tracker = ReviewHandoffIdentityTrackerV1P1("surface-session", 2,
            fixtures.DESCRIPTION)
        self.policy = LossAwareDelayedSemanticPolicy(lambda request: None,
            tracker_supplier=lambda: self.tracker, session_id="surface-session",
            target_generation=2, description=fixtures.DESCRIPTION,
            executor=self.executor, clock=lambda: 100., config=DelayedBridgeConfig())
        self.results = []

    def tearDown(self):
        self.policy.close(); self.tracker.close()

    def test_loss_frame_source_is_skipped_then_next_actual_frame_reacquires(self):
        self.initial(); self.advance(4, 15)
        lost = self.step(17)
        self.assertEqual(lost["review_handoff"]["event"], "CHAIN_LOST")
        before = (self.policy._serial, self.policy._last_submitted,
            self.policy._pending, len(self.executor.calls))
        skipped = self.policy.request(self.frame(17), now=13.45)
        self.assertEqual(skipped["status"], "NOT_SUBMITTED")
        self.assertEqual(skipped["reason"], "FRESH_SOURCE_AFTER_LOSS_REQUIRED")
        self.assertEqual(before, (self.policy._serial, self.policy._last_submitted,
            self.policy._pending, len(self.executor.calls)))
        self.step(18); self.submit(self.frame(18)); self.complete()
        adopted = self.step(19)
        self.assertEqual(adopted["state"], "RECOVERY_PENDING")
        self.assertEqual(self.step(20)["status"], "NO_TARGET_OUTPUT")
        restored = self.step(21)
        self.assertEqual(restored["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")
        self.assertEqual(restored["semantic_anchor"], "chain-semantic-000002")
        self.assertEqual(restored["delayed_chain_proof"]["post_adoption_frame_ids"],
            ["frame-20", "frame-21"])

    def test_repeated_used_reply_rejection_never_moves_loss_barrier(self):
        self.initial(); self.advance(4, 15)
        self.tracker.update(self.frame(15), now=float("nan"))
        barrier = self.tracker.loss_watermark
        for seq in (16, 17):
            result = self.step(seq)
            self.assertEqual(result["review_handoff"]["event"], "REVIEW_REJECTED")
            self.assertEqual(result["state"], "LOST")
            self.assertEqual(self.tracker.loss_watermark, barrier)
        # These rejected replies cannot make this genuinely later source stale.
        self.submit(self.frame(17)); self.complete()
        self.assertEqual(self.step(18)["state"], "RECOVERY_PENDING")
        self.step(19)
        self.assertEqual(self.step(20)["status"], "TEMPORAL_LOCAL_ASSOCIATION_VALID")

    def test_inflight_ineligible_source_rejected_once_then_no_watermark_drift(self):
        self.initial(); self.advance(4, 15)
        self.submit(self.frame(15))
        self.tracker.update(self.frame(15), now=float("nan"))
        barrier = self.tracker.loss_watermark
        self.complete()
        first = self.step(16)
        self.assertEqual(first["reason"], "HANDOFF_FRESH_SOURCE_AFTER_LOSS_REQUIRED")
        self.assertEqual(first["review_handoff"]["event"], "REVIEW_REJECTED")
        self.assertEqual(self.tracker.loss_watermark, barrier)
        second = self.step(17)
        self.assertEqual(second["reason"], "SEMANTIC_ANCHOR_REUSED_AFTER_LOSS_OR_REPLACEMENT")
        self.assertEqual(self.tracker.loss_watermark, barrier)

    def test_second_actual_break_after_recovery_advances_the_loss_barrier(self):
        self.test_loss_frame_source_is_skipped_then_next_actual_frame_reacquires()
        old = self.tracker.loss_watermark
        second_loss = self.step(23)
        self.assertEqual(second_loss["review_handoff"]["event"], "CHAIN_LOST")
        new = self.tracker.loss_watermark
        self.assertGreater(new["source_readback_receipt_monotonic_s"],
            old["source_readback_receipt_monotonic_s"])
        self.assertGreater(new["observed_monotonic_s"], old["observed_monotonic_s"])

    def test_actual_gap_while_lost_still_updates_freshness_barrier(self):
        self.initial(); self.advance(4, 15); self.step(17)
        old = self.tracker.loss_watermark
        result = self.step(19)
        self.assertEqual(result["review_handoff"]["event"], "CHAIN_LOST")
        self.assertGreater(self.tracker.loss_watermark["source_readback_receipt_monotonic_s"],
            old["source_readback_receipt_monotonic_s"])

    def test_old_anchor_expiry_drops_active_and_pending_without_new_authority(self):
        self.initial(); self.advance(4, 38); self.review(38)
        expired = self.step(40)
        self.assertEqual(expired["review_handoff"]["event"], "CHAIN_LOST")
        self.assertEqual(expired["status"], "NO_TARGET_OUTPUT")
        watermark = self.tracker.loss_watermark
        repeated = self.step(41)
        self.assertEqual(repeated["review_handoff"]["event"], "REVIEW_REJECTED")
        self.assertEqual(watermark, self.tracker.loss_watermark)
        self.assertIsNone(repeated["review_handoff"]["pending_request_id"])

    def test_conflict_still_drops_active_and_pending_immediately(self):
        self.initial(); self.advance(4, 15)
        conflict = self.review(15, box=(70., 30., 90., 50.))
        self.assertEqual(conflict["review_handoff"]["event"], "REVIEW_CONFLICT")
        self.assertEqual(conflict["state"], "AMBIGUOUS")
        self.assertIsNone(conflict["selected_candidate_id"])
        barrier = self.tracker.loss_watermark
        repeated = self.step(17)
        self.assertEqual(repeated["review_handoff"]["event"], "REVIEW_REJECTED")
        self.assertEqual(self.tracker.loss_watermark, barrier)

    def test_original_cooldown_and_single_transport_slot_remain_enforced(self):
        self.initial()
        self.assertEqual(self.policy.request(self.frame(3), now=10.65)["reason"],
            "LOW_FREQUENCY_INTERVAL")
        self.advance(4, 15); self.submit(self.frame(15))
        self.step(16)
        self.assertEqual(self.policy.request(self.frame(16), now=13.25)["reason"],
            "TRANSPORT_SLOT_OCCUPIED")
        self.assertEqual(len(self.executor.calls), 2)

    def test_submission_time_must_follow_observed_loss_even_with_newer_source(self):
        self.initial(); self.advance(4, 15); self.step(17)
        before = self.policy._serial
        # Clock is advanced to the loss observer time while no slot is consumed.
        now = self.tracker.loss_watermark["observed_monotonic_s"]
        current = replace(self.frame(18), readback_receipt_monotonic_s=13.401,
            candidates_ready_monotonic_s=13.411)
        result = self.policy.request(current, now=now)
        self.assertEqual(result["reason"], "FRESH_SOURCE_AFTER_LOSS_REQUIRED")
        self.assertEqual(self.policy._serial, before)

    def test_preflight_does_not_bypass_original_frame_context_checks(self):
        self.initial(); self.advance(4, 15); self.step(17)
        wrong = replace(self.frame(17), target_generation=3)
        self.assertEqual(self.policy.request(wrong, now=13.45)["reason"],
            "FRAME_CONTEXT_MISMATCH")
        self.assertIsNone(self.policy._pending)

    def test_watermark_snapshot_cannot_mutate_internal_loss_receipts(self):
        self.initial(); self.advance(4, 15); self.step(17)
        snapshot = self.tracker.loss_watermark
        snapshot["source_readback_receipt_monotonic_s"] = 0.
        self.assertEqual(self.tracker.loss_watermark["source_readback_receipt_monotonic_s"], 13.4)

    def test_context_reset_clears_loss_watermark_and_closed_policy_cannot_submit(self):
        self.initial(); self.advance(4, 15); self.step(17)
        self.tracker.reset(session_id="surface-session", target_generation=3,
            description=fixtures.DESCRIPTION, now=13.5)
        self.policy.reset(session_id="surface-session", target_generation=3,
            description=fixtures.DESCRIPTION, now=13.5)
        self.assertEqual(self.tracker.loss_watermark, {
            "source_readback_receipt_monotonic_s": None, "observed_monotonic_s": None})
        current = replace(self.frame(18), target_generation=3)
        self.assertEqual(self.policy.request(current, now=13.64)["status"], "SUBMITTED")
        self.policy.close()
        self.assertEqual(self.policy.request(current, now=100.)["reason"], "POLICY_CLOSED")


if __name__ == "__main__":
    unittest.main()
