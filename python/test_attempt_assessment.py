import unittest

from attempt_assessment import (
    AttemptAction,
    AttemptState,
    ContactAttemptAssessment,
)


class ContactAttemptAssessmentTests(unittest.TestCase):
    def test_requires_start_and_distinguishes_temporary_loss(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        with self.assertRaises(ValueError):
            evaluator.update(tracking_state="DETECTED", observation_fresh=True)
        evaluator.start()
        tracked = evaluator.update(tracking_state="DETECTED", observation_fresh=True,
                                   elapsed_s=1)
        self.assertEqual(tracked.state, AttemptState.TRACKING)
        self.assertEqual(tracked.action, AttemptAction.CONTINUE)
        lost = evaluator.update(tracking_state="TEMPORARILY_LOST",
                                observation_fresh=True, elapsed_s=2)
        self.assertEqual(lost.state, AttemptState.TEMPORARILY_LOST)
        self.assertEqual(lost.action, AttemptAction.REACQUIRE)

    def test_long_loss_is_handoff_eligible_but_not_failure(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        evaluator.start()
        assessment = evaluator.update(tracking_state="LOST",
                                       observation_fresh=False, elapsed_s=3)
        self.assertEqual(assessment.state, AttemptState.UNCONFIRMED)
        self.assertEqual(assessment.action, AttemptAction.HANDOFF_ELIGIBLE)
        self.assertNotEqual(assessment.state, AttemptState.FAILED)

    def test_timeout_without_collision_is_failure(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        evaluator.start()
        assessment = evaluator.update(tracking_state="LOST",
                                      observation_fresh=False, elapsed_s=10)
        self.assertEqual(assessment.state, AttemptState.FAILED)
        self.assertEqual(assessment.action, AttemptAction.HANDOFF_ELIGIBLE)
        self.assertEqual(assessment.reason,
                         "attempt_timeout_without_target_collision")

    def test_target_collision_is_authoritative_and_terminal(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        evaluator.start()
        assessment = evaluator.update(tracking_state="LOST",
                                      observation_fresh=False,
                                      collision_kind="TARGET", elapsed_s=4)
        self.assertEqual(assessment.state, AttemptState.CONTACT_CONFIRMED)
        self.assertEqual(assessment.action, AttemptAction.COMPLETE)
        with self.assertRaises(ValueError):
            evaluator.update(tracking_state="DETECTED", observation_fresh=True,
                             elapsed_s=5)

    def test_non_target_collision_stops_attempt_without_confirming_contact(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        evaluator.start()
        assessment = evaluator.update(tracking_state="DETECTED",
                                      observation_fresh=True,
                                      collision_kind="OTHER", elapsed_s=2)
        self.assertEqual(assessment.state, AttemptState.FAILED)
        self.assertEqual(assessment.action, AttemptAction.STOP)
        self.assertEqual(assessment.reason, "non_target_collision")

    def test_stale_observation_cannot_continue(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        evaluator.start()
        assessment = evaluator.update(tracking_state="DETECTED",
                                      observation_fresh=False, elapsed_s=1)
        self.assertEqual(assessment.state, AttemptState.TEMPORARILY_LOST)
        self.assertEqual(assessment.action, AttemptAction.REACQUIRE)

    def test_caller_safety_bound_is_recorded_as_failure(self):
        evaluator = ContactAttemptAssessment(timeout_s=10)
        evaluator.start()
        assessment = evaluator.fail("dash_bound_exceeded_without_target_collision",
                                    elapsed_s=4, tracking_state="TEMPORARILY_LOST")
        self.assertEqual(assessment.state, AttemptState.FAILED)
        self.assertEqual(assessment.action, AttemptAction.HANDOFF_ELIGIBLE)
        self.assertEqual(evaluator.summary()["transitions"][-1]["reason"],
                         "dash_bound_exceeded_without_target_collision")

    def test_rejects_invalid_inputs(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    ContactAttemptAssessment(timeout)
        evaluator = ContactAttemptAssessment()
        evaluator.start()
        with self.assertRaises(ValueError):
            evaluator.update(collision_kind="UNKNOWN")
        with self.assertRaises(ValueError):
            evaluator.update(elapsed_s=-1)


if __name__ == "__main__":
    unittest.main()
