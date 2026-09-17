"""Annotation integrity and camera restoration checks without a live simulator."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import airsim

from maritime_scene import ROOT, capture_view, validate_config


class MaritimeSceneTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"configs"/"maritime_scene.json").read_text())

    def test_duplicate_annotation_id_is_rejected(self):
        self.cfg["objects"][1]["segmentation_id"] = self.cfg["objects"][0]["segmentation_id"]
        with self.assertRaises(ValueError):
            validate_config(self.cfg)

    def test_invalid_class_and_nonfinite_camera_are_rejected(self):
        bad = copy.deepcopy(self.cfg)
        bad["objects"][0]["class_name"] = "unknown_label"
        with self.assertRaises(ValueError):
            validate_config(bad)
        self.cfg["views"][0]["eye_relative_m"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_config(self.cfg)

    def test_camera_restored_after_capture_error(self):
        class CameraOnlyClient:
            def __init__(self):
                self.camera = airsim.Pose(airsim.Vector3r(1, 0, -0.05), airsim.Quaternionr())
                self.set_count = 0
            def simGetVehiclePose(self, **kwargs):
                return airsim.Pose()
            def simGetCameraInfo(self, *args, **kwargs):
                return SimpleNamespace(pose=self.camera)
            def simGetObjectPose(self, *args):
                return airsim.Pose()
            def simSetCameraPose(self, name, pose, **kwargs):
                self.camera = pose
                self.set_count += 1
            def simGetImage(self, *args, **kwargs):
                raise RuntimeError("Injected capture failure")
        client = CameraOnlyClient()
        with TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, "Injected capture failure"):
                capture_view(client, self.cfg["views"][0], self.cfg["origin_world_ned_m"], Path(output))
        self.assertEqual(client.set_count, 2)
        self.assertAlmostEqual(client.camera.position.x_val, 1)
        self.assertAlmostEqual(client.camera.position.z_val, -0.05)


if __name__ == "__main__":
    unittest.main()
