import unittest

from recover_airborne import grounded


class RecoveryEvidenceTests(unittest.TestCase):
    def test_requires_ground_height_and_low_speed(self):
        self.assertTrue(grounded([-5.0, -4.0, 1.68], 0.01))
        self.assertTrue(grounded([-5.0, -4.0, 0.83], 0.01, landed_flag=True))
        self.assertFalse(grounded([-5.0, -4.0, -3.0], 0.01))
        self.assertFalse(grounded([-5.0, -4.0, 1.68], 0.2, landed_flag=True))


if __name__ == "__main__":
    unittest.main()
