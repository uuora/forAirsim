import unittest
import cv2
import numpy as np
from balloon_detector import detect_balloon


class DetectorTests(unittest.TestCase):
    def test_blank_and_small_red_noise(self):
        image = np.zeros((480,640,3), np.uint8)
        self.assertFalse(detect_balloon(image)["detected"])
        cv2.circle(image, (200,200), 5, (0,0,255), -1)
        self.assertFalse(detect_balloon(image)["detected"])

    def test_colour_and_centroid(self):
        image = np.zeros((480,640,3), np.uint8)
        cv2.ellipse(image,(320,200),(35,45),0,0,360,(0,255,0),-1)
        self.assertFalse(detect_balloon(image)["detected"])
        cv2.ellipse(image,(320,200),(35,45),0,0,360,(0,0,255),-1)
        result = detect_balloon(image)
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["selected"]["centre_px"][0],320,delta=1)

    def test_edge_candidate_discloses_truncation(self):
        image = np.zeros((480,640,3), np.uint8)
        cv2.ellipse(image,(25,200),(35,45),0,0,360,(0,0,255),-1)
        self.assertTrue(detect_balloon(image)["selected"]["truncated"])

    def test_bottom_vehicle_mask(self):
        image = np.zeros((480,640,3), np.uint8)
        cv2.circle(image,(320,450),20,(0,0,255),-1)
        self.assertFalse(detect_balloon(image)["detected"])


if __name__ == "__main__":
    unittest.main()
