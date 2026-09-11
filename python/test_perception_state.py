import unittest

from perception_state import PerceptionTracker, TrackingState


class PerceptionTrackerTests(unittest.TestCase):
    def test_requires_two_consecutive_frames_for_first_detection(self):
        tracker = PerceptionTracker(confirm_frames=2, lost_frames=3)
        self.assertEqual(tracker.update(True).state, TrackingState.LOST)
        update = tracker.update(True)
        self.assertEqual(update.state, TrackingState.DETECTED)
        self.assertTrue(update.changed)
        self.assertEqual(update.detection_streak, 2)

    def test_short_dropout_and_immediate_reacquisition(self):
        tracker = PerceptionTracker(confirm_frames=2, lost_frames=3)
        tracker.update(True)
        tracker.update(True)
        missed = tracker.update(False)
        self.assertEqual(missed.state, TrackingState.TEMPORARILY_LOST)
        self.assertEqual(missed.miss_streak, 1)
        reacquired = tracker.update(True)
        self.assertEqual(reacquired.state, TrackingState.DETECTED)
        self.assertTrue(reacquired.changed)

    def test_three_consecutive_misses_lose_track(self):
        tracker = PerceptionTracker(confirm_frames=2, lost_frames=3)
        tracker.update(True)
        tracker.update(True)
        self.assertEqual(tracker.update(False).state, TrackingState.TEMPORARILY_LOST)
        self.assertEqual(tracker.update(False).state, TrackingState.TEMPORARILY_LOST)
        lost = tracker.update(False)
        self.assertEqual(lost.state, TrackingState.LOST)
        self.assertTrue(lost.changed)
        self.assertEqual(lost.miss_streak, 3)

    def test_rejects_invalid_thresholds(self):
        for confirm_frames, lost_frames in ((0, 3), (2, 0), (-1, 3)):
            with self.subTest(confirm_frames=confirm_frames, lost_frames=lost_frames):
                with self.assertRaises(ValueError):
                    PerceptionTracker(confirm_frames, lost_frames)


if __name__ == "__main__":
    unittest.main()
