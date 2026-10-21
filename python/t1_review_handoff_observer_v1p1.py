"""Background review observer with fresh-source recovery scheduling.

The tracker establishes the loss watermark before the policy decides whether
the same actual frame can submit. Source ages and motion budgets are unchanged.
"""
from t1_review_handoff_v1p1 import (
    ReviewHandoffIdentityTrackerV1P1, LossAwareDelayedSemanticPolicy,
)
from t1_source_attribute_guard import AttributeGuardedFlightObserver


class ReacquisitionAwareFlightObserver(AttributeGuardedFlightObserver):
    def __init__(self, detector_config_path):
        super().__init__(detector_config_path)
        core = self.inner.core
        core.tracker_factory = ReviewHandoffIdentityTrackerV1P1
        core.policy_factory = lambda backend, **kwargs: LossAwareDelayedSemanticPolicy(
            backend, tracker_supplier=lambda: core.tracker, **kwargs)

    @staticmethod
    def _declare(result):
        result['review_handoff_policy'] = 'CONFIRM_IN_BACKGROUND_THEN_ATOMIC_CURRENT_CANDIDATE_HANDOFF'
        result['recovery_submission_policy'] = 'STRICTLY_NEW_ACTUAL_SOURCE_AFTER_OBSERVED_LOSS'
        result['prior_source_deadline_extended'] = False
        return result

    def begin(self, *args, **kwargs):
        return self._declare(super().begin(*args, **kwargs))

    def process(self, observation):
        return self._declare(super().process(observation))

    def close(self):
        super().close()
        self._declare(self.close_receipt)
