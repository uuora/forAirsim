import math
import unittest
from types import SimpleNamespace

from camera_geometry import scene_intrinsics


class SceneIntrinsicsTests(unittest.TestCase):
    def info(self):
        scale = 1 / math.tan(math.radians(70)/2)
        return SimpleNamespace(fov=89.9036, proj_mat=SimpleNamespace(matrix=[
            [0, scale, 0, 0], [0, 0, -scale*1280/720, 0],
            [0, 0, 0, 10], [-1, 0, 0, 0]]))

    def test_viewer_fov_does_not_corrupt_scene_intrinsics(self):
        result = scene_intrinsics(self.info(), 1280, 720)
        self.assertAlmostEqual(result["horizontal_fov_deg"], 70)
        self.assertAlmostEqual(result["K_nominal"][0][0], 914.0147243, places=5)
        self.assertFalse(result["camera_info_fov_matches_scene"])

    def test_unexpected_projection_is_not_silently_accepted(self):
        info = self.info()
        info.proj_mat.matrix[0][0] = .2
        with self.assertRaises(ValueError):
            scene_intrinsics(info, 1280, 720)


if __name__ == "__main__":
    unittest.main()
