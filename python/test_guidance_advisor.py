import unittest

from guidance_advisor import GuidanceAdvice, advise


class GuidanceAdvisorTests(unittest.TestCase):
    def test_holds_without_confirmed_complete_observation(self):
        self.assertEqual(advise("LOST", (0.0, 0.0), 3.0), GuidanceAdvice.HOLD)
        self.assertEqual(advise("DETECTED", None, 3.0), GuidanceAdvice.HOLD)
        self.assertEqual(advise("DETECTED", (0.0, 0.0), None), GuidanceAdvice.HOLD)

    def test_aligns_before_advancing(self):
        self.assertEqual(advise("DETECTED", (0.2, 0.0), 3.0), GuidanceAdvice.ALIGN_RIGHT)
        self.assertEqual(advise("DETECTED", (-0.2, 0.0), 3.0), GuidanceAdvice.ALIGN_LEFT)
        self.assertEqual(advise("DETECTED", (0.0, -0.2), 3.0), GuidanceAdvice.ALIGN_UP)

    def test_advances_or_stops_when_aligned(self):
        self.assertEqual(advise("DETECTED", (0.02, -0.02), 3.0), GuidanceAdvice.ADVANCE)
        self.assertEqual(advise("DETECTED", (0.02, -0.02), 1.2), GuidanceAdvice.STOP)


if __name__ == "__main__":
    unittest.main()
