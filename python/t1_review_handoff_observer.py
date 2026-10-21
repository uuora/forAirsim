"""Descriptor-guarded RGB observer with independently confirmed review handoff.

The prior source retains its original expiry. Background review is not motion
authority; selected current frames still pass the unchanged surface and gate.
"""
from t1_review_handoff import ReviewHandoffIdentityTracker
from t1_source_attribute_guard import AttributeGuardedFlightObserver


class ReviewHandoffFlightObserver(AttributeGuardedFlightObserver):
    def __init__(self, detector_config_path):
        super().__init__(detector_config_path)
        self.inner.core.tracker_factory = ReviewHandoffIdentityTracker

    def begin(self, *args, **kwargs):
        result = super().begin(*args, **kwargs)
        result['review_handoff_policy'] = 'CONFIRM_IN_BACKGROUND_THEN_ATOMIC_CURRENT_CANDIDATE_HANDOFF'
        result['prior_source_deadline_extended'] = False
        return result

    def process(self, observation):
        result = super().process(observation)
        result['review_handoff_policy'] = 'CONFIRM_IN_BACKGROUND_THEN_ATOMIC_CURRENT_CANDIDATE_HANDOFF'
        return result

    def close(self):
        super().close()
        self.close_receipt['review_handoff_policy'] = 'CONFIRM_IN_BACKGROUND_THEN_ATOMIC_CURRENT_CANDIDATE_HANDOFF'
