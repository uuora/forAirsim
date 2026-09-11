"""Offline checks for evidence rejection and both mission state paths."""
import unittest
from types import SimpleNamespace

from hit_judge import classify_collision
from mission_state import MissionState as S, transition


class EvidenceTests(unittest.TestCase):
    def collision(self, **changes):
        values = dict(has_collided=True, time_stamp=101, object_name="BalloonTarget")
        values.update(changes)
        return SimpleNamespace(**values)

    def test_stale_target_contact_cannot_retrigger(self):
        self.assertEqual(classify_collision(self.collision(), 101, "BalloonTarget"), "NONE")

    def test_non_contact_cannot_trigger(self):
        self.assertEqual(classify_collision(self.collision(has_collided=False), 100, "BalloonTarget"), "NONE")

    def test_only_exact_body_contact_counts(self):
        for name in ("Ground_4", "BalloonTarget_String", "BalloonTarget2"):
            self.assertEqual(classify_collision(self.collision(object_name=name), 100, "BalloonTarget"), "OTHER")
        self.assertEqual(classify_collision(self.collision(), 100, "BalloonTarget"), "TARGET")

    def test_a_success_terminates(self):
        state = transition(transition(S.READY, "start"), "hit")
        self.assertEqual(state, S.HIT_CONFIRMED)
        with self.assertRaises(ValueError):
            transition(state, "fallback_ready")

    def test_b_requires_clearance_event(self):
        state = transition(S.READY, "start")
        with self.assertRaises(ValueError):
            transition(state, "fallback_ready")
        state = transition(transition(state, "miss"), "fallback_ready")
        self.assertEqual(transition(state, "hit"), S.HIT_CONFIRMED)
        self.assertEqual(transition(state, "timeout"), S.FAILED)


if __name__ == "__main__":
    unittest.main()
