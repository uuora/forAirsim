"""A legitimate hit during a view toggle must not look like a queue reset."""
import unittest

from validate_dual_target_control import queue_transition_valid


class AcceptanceQueueTests(unittest.TestCase):
    def test_view_toggle_allows_one_ordered_hit_without_reset(self):
        before = dict(hits=2, targets=[[1, 2, 3], [4, 5, 6]])
        self.assertTrue(queue_transition_valid(before, before))
        self.assertTrue(queue_transition_valid(before, dict(hits=3, targets=[[4, 5, 6], [7, 8, 9]])))
        self.assertFalse(queue_transition_valid(before, dict(hits=3, targets=[[7, 8, 9], [4, 5, 6]])))
        self.assertFalse(queue_transition_valid(before, dict(hits=1, targets=before['targets'])))
        self.assertFalse(queue_transition_valid(before, dict(hits=2, targets=[[7, 8, 9], [4, 5, 6]])))


if __name__ == '__main__':
    unittest.main()
