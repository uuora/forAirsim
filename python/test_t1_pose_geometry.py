import math
import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from t1_pose_geometry import (CameraPose, SceneCameraModel, bbox_plane_corners_world,
    project_footprint_polygon, project_world, quaternion_rotation_xyzw,
    rgb_reference_on_surface_depth, unproject_axial, unproject_range)
from replay_t1_pose_geometry import replay_observation, replay_probe_frame, replay_reports


def euler_quaternion(roll, pitch, yaw):
    """Independent Euler expression, xyzw; angles in radians."""
    cr, cp, cy = (math.cos(a/2) for a in (roll, pitch, yaw))
    sr, sp, sy = (math.sin(a/2) for a in (roll, pitch, yaw))
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


class T1PoseGeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = SceneCameraModel(640, 360, 320., 320., 320., 180.)
        self.identity = CameraPose([0, 0, 0], [0, 0, 0, 1], [0, 0, 0])

    def test_scene_matrix_instead_of_viewer_fov(self):
        info = SimpleNamespace(fov=110., proj_mat=SimpleNamespace(matrix=[
            [0, 1, 0, 0], [0, 0, -640/360, 0], [0, 0, 0, 10], [-1, 0, 0, 0]]))
        model = SceneCameraModel.from_camera_info(info, 640, 360)
        self.assertEqual(model.fx, 320)
        self.assertEqual(model.metadata()["K_nominal"][0][2], 320)

    def test_nonzero_yaw_has_independent_known_projection(self):
        pose = CameraPose([0, 0, 0], euler_quaternion(0, 0, math.pi/2), [10, 20, -3])
        result = project_world([9, 24, -2.5], pose, self.camera)
        np.testing.assert_allclose(result["point_camera_m"], [4, 1, .5], atol=1e-12)
        np.testing.assert_allclose(result["pixel_xy"], [400, 220], atol=1e-12)

    def test_roll_rotates_right_bearing_to_down(self):
        pose = CameraPose([0, 0, 0], euler_quaternion(math.pi/2, 0, 0), [0, 0, 0])
        np.testing.assert_allclose(pose.camera_to_world([4, 1, 0]), [4, 0, 1], atol=1e-12)
        np.testing.assert_allclose(project_world([4, 0, 1], pose, self.camera)["pixel_xy"],
                                   [400, 180], atol=1e-12)

    def test_pitch_rotates_forward_to_negative_down(self):
        pose = CameraPose([0, 0, 0], euler_quaternion(0, math.pi/2, 0), [0, 0, 0])
        np.testing.assert_allclose(pose.camera_to_world([4, 0, 0]), [0, 0, -4], atol=1e-12)
        np.testing.assert_allclose(project_world([0, 0, -4], pose, self.camera)["pixel_xy"],
                                   [320, 180], atol=1e-12)

    def test_mixed_yaw_pitch_roll_roundtrip_all_pixels(self):
        pose = CameraPose([1.5, -.8, -2], euler_quaternion(.35, -.25, .8), [12, -6, .4])
        for pixel in ([50, 35], [320, 180], [600, 330]):
            point = unproject_range(pixel, 7.2, pose, self.camera)
            result = project_world(point, pose, self.camera)
            np.testing.assert_allclose(result["pixel_xy"], pixel, atol=2e-12)
            self.assertAlmostEqual(result["ray_range_m"], 7.2)

    def test_translation_is_not_rotated_by_camera_orientation(self):
        pose = CameraPose([1, 2, 3], euler_quaternion(0, 0, math.pi/2), [10, 20, 30])
        np.testing.assert_allclose(pose.camera_to_world([4, 0, 0]), [11, 26, 33], atol=1e-12)
        np.testing.assert_allclose(pose.world_to_camera([11, 26, 33]), [4, 0, 0], atol=1e-12)

    def test_depth_range_and_axial_depth_are_different(self):
        point = unproject_range([640, 180], math.sqrt(32), self.identity, self.camera)
        np.testing.assert_allclose(point, [4, 4, 0])
        wrong = unproject_axial([640, 180], math.sqrt(32), self.identity, self.camera)
        self.assertGreater(np.linalg.norm(point-wrong), 2.)

    def test_legacy_halfpixel_is_detected(self):
        with self.assertRaisesRegex(ValueError, "halfpixel"):
            SceneCameraModel.from_k([[320, 0, 319.5], [0, 320, 179.5], [0, 0, 1]], 640, 360)
        # The old convention's optical axis differs by exactly .5 pixel in each axis.
        projected = project_world([4, 0, 0], self.identity, self.camera)["pixel_xy"]
        np.testing.assert_allclose(np.array(projected)-[319.5, 179.5], [.5, .5])

    def test_near_unit_quaternion_is_normalized(self):
        np.testing.assert_allclose(quaternion_rotation_xyzw([0, 0, 0, 1.01]), np.eye(3))
        np.testing.assert_allclose(quaternion_rotation_xyzw([0, 0, 0, -1]), np.eye(3))

    def test_invalid_quaternion_and_nonfinite_translation_rejected(self):
        for quaternion in ([0, 0, 0, 0], [0, 0, 0, 2], [0, 0, float("nan"), 1], [1, 0, 0]):
            with self.assertRaises(ValueError):
                CameraPose([0, 0, 0], quaternion, [0, 0, 0])
        with self.assertRaises(ValueError):
            CameraPose([0, 0, 0], [0, 0, 0, 1], [0, float("inf"), 0])

    def test_bad_range_behind_near_and_nonfinite_points_rejected(self):
        for depth in (0, -1, float("nan"), float("inf"), .01):
            with self.assertRaises(ValueError):
                unproject_range([320, 180], depth, self.identity, self.camera)
        for point in ([0, 0, 0], [-1, 1, 1], [.05, 0, 0], [2, float("inf"), 0]):
            with self.assertRaises(ValueError):
                project_world(point, self.identity, self.camera)

    def test_image_boundaries_are_half_open(self):
        for pixel, inside in (([0, 0], True), ([639.9, 359.9], True), ([640, 180], False),
                              ([320, 360], False), ([-.1, 180], False)):
            point = unproject_axial(pixel, 4, self.identity, self.camera)
            self.assertEqual(project_world(point, self.identity, self.camera)["inside_image"], inside)

    def test_centroid_depth_reference_respects_rotated_camera(self):
        pose = CameraPose([1, 2, -3], euler_quaternion(.15, -.2, .7), [10, 20, 0])
        surface = pose.camera_to_world([5, .5, -.3])
        result = rgb_reference_on_surface_depth([400, 220], surface, pose, self.camera)
        point = project_world(result["reference_world_ned_m"], pose, self.camera)
        np.testing.assert_allclose(point["pixel_xy"], [400, 220], atol=1e-12)
        self.assertAlmostEqual(point["axial_depth_m"], 5)
        self.assertIn("not_physical_target_center", result["semantics"])

    def test_bbox_plane_reprojects_and_handles_candidate_height_rotation(self):
        corners = bbox_plane_corners_world([200, 100, 100, 80], 4, self.identity, self.camera)
        result = project_footprint_polygon(corners, self.identity, self.camera)
        self.assertEqual(result["field_area_fraction"], 1.)
        self.assertAlmostEqual(result["projected_area_px2"], 8000)
        candidate = CameraPose([.2, -.5, -.6], euler_quaternion(.1, -.15, .25), [0, 0, 0])
        moved = project_footprint_polygon(corners, candidate, self.camera)
        self.assertTrue(0 <= moved["field_area_fraction"] <= 1)
        self.assertFalse(np.allclose(result["projected_vertices_pixel_xy"], moved["projected_vertices_pixel_xy"]))

    def test_polygon_clipping_and_complete_outside(self):
        original = bbox_plane_corners_world([0, 100, 80, 80], 4, self.identity, self.camera)
        candidate = CameraPose([0, .5, 0], [0, 0, 0, 1], [0, 0, 0])
        clipped = project_footprint_polygon(original, candidate, self.camera)
        self.assertAlmostEqual(clipped["field_area_fraction"], .5)
        self.assertFalse(clipped["all_vertices_in_image"])
        outside = project_footprint_polygon(original, CameraPose([0, 2, 0], [0, 0, 0, 1], [0, 0, 0]), self.camera)
        self.assertEqual(outside["field_area_fraction"], 0.)
        self.assertEqual(outside["clipped_vertices_pixel_xy"], [])

    def test_polygon_rejects_crossings_degenerate_and_behind(self):
        corners = bbox_plane_corners_world([100, 100, 80, 80], 4, self.identity, self.camera)
        for points in (corners[[0, 2, 1, 3]], np.repeat(corners[:1], 4, axis=0),
                       [[-1, 0, 0], [4, 1, 0], [4, 0, 1]]):
            with self.assertRaises(ValueError):
                project_footprint_polygon(points, self.identity, self.camera)

    def test_self_intersecting_five_point_star_is_rejected(self):
        angles = np.arange(5)*2*math.pi/5
        pixels = np.column_stack([320+80*np.cos(angles), 180+80*np.sin(angles)])
        points = [unproject_axial(pixel, 4, self.identity, self.camera)
                  for pixel in pixels[[0, 2, 4, 1, 3]]]
        with self.assertRaises(ValueError):
            project_footprint_polygon(points, self.identity, self.camera)

    def test_invalid_scene_model_and_bbox(self):
        for args in ((640.0, 360, 320, 320, 320, 180), (True, 360, 320, 320, 320, 180),
                     (640, 360, 0, 0, 320, 180), (640, 360, 320, 321, 320, 180)):
            with self.assertRaises(ValueError):
                SceneCameraModel(*args)
        for bbox in ([0, 0, 0, 10], [-1, 10, 20, 20], [630, 10, 20, 20]):
            with self.assertRaises(ValueError):
                bbox_plane_corners_world(bbox, 4, self.identity, self.camera)

    def legacy_row(self):
        pixel = [400, 200]
        ray = np.array([1, (400-319.5)/320, (200-179.5)/320])
        point = ray/np.linalg.norm(ray)*5
        return {"frame_id": "f1", "timestamped_observation": {
            "status": "OK", "rejection_reasons": [], "coordinate_frame": "VEHICLE_LOCAL_NED",
            "projection": {"depth_size_wh": [640, 360], "rgb_size_wh": [640, 360],
                           "focal_lengths_px": [320, 320], "principal_pixel_xy": [319.5, 179.5]},
            "camera_position_local_ned_m": [0, 0, 0], "camera_orientation_xyzw": [0, 0, 0, 1],
            "surface_position_local_ned_m": point.tolist(), "depth_m": 5,
            "depth_statistics": {"representative_pixel_xy": pixel},
            "centre_px": [400, 200], "bbox_xywh": [360, 160, 80, 80]}}

    def test_saved_legacy_replay_separates_migration_from_roundtrip(self):
        row = self.legacy_row()
        result = replay_observation(row, [10, 20, 30])
        np.testing.assert_allclose(result["legacy_surface_under_scene_pixel_delta_xy"], [.5, .5])
        self.assertLess(result["new_scene_roundtrip_max_axis_error_px"], 1e-10)
        self.assertGreater(result["scene_reconstructed_surface_shift_m"], .001)
        self.assertIn("not_independently_saved", result["source_focal_scope"])

    def test_saved_replay_preserves_rejected_frame_in_all_frame_denominator(self):
        good = self.legacy_row()
        bad = deepcopy(good)
        bad["frame_id"] = "f2"
        bad["timestamped_observation"]["surface_position_local_ned_m"][1] += .1
        report = {"policy": "p", "origin_calibration": {
            "world_from_vehicle_local_translation_ned_m": [0, 0, 0]}, "observations": [good, bad]}
        summary, records = replay_reports([("source.json", report)], [])
        self.assertEqual(summary["observation_count_all"], 2)
        self.assertEqual(summary["replayed_observation_count"], 1)
        self.assertEqual(summary["rejected_observation_count"], 1)
        self.assertEqual(records[1]["frame_id"], "f2")
        self.assertEqual(summary["new_simulator_flights"], 0)
        self.assertFalse(summary["formal_method_efficacy_result"])

    def fresh_probe_fixture(self):
        mask = np.zeros((360, 640), dtype=bool)
        mask[170:190, 310:330] = True
        row = {"frame_id": "probe", "case_id": "center", "image_size_wh": [640, 360],
            "scene_intrinsics": {"intrinsics_source": "AirSim_Scene_projection_matrix",
                                 "K_nominal": [[320, 0, 320], [0, 320, 180], [0, 0, 1]]},
            "response_camera_poses": [{"position_local_ned_m": [0, 0, 0],
                                       "orientation_xyzw": [0, 0, 0, 1]} for _ in range(3)],
            "world_from_local_translation_ned_m": [10, 20, 30], "camera_world_ned_m": [10, 20, 30],
            "evaluation_only": {"target_world_ned_m": [14, 20, 30], "target": {
                "visible_pixel_count": 400, "centroid_xy_pixels": [319.5, 179.5],
                "bbox_xyxy_pixels": [310, 170, 330, 190]}}, "rgb_depth_seg_readback_skew_s": .01}
        return row, np.full(mask.shape, 4.), mask

    def test_actual_scene_probe_roundtrip_keeps_actor_pivot_diagnostic_separate(self):
        row, depth, mask = self.fresh_probe_fixture()
        result = replay_probe_frame(row, depth, mask)
        self.assertEqual(result["sampled_depth_ray_count"], 9)
        self.assertLess(result["max_depth_pixel_roundtrip_error_px"], 1e-10)
        np.testing.assert_allclose(result["actor_pivot_minus_visible_centroid_xy_px"], [.5, .5])
        self.assertIn("not_empirical_lens_calibration", result["projection_centroid_comparison_scope"])
        bad = deepcopy(row)
        bad["evaluation_only"]["target"]["visible_pixel_count"] = 399
        with self.assertRaisesRegex(ValueError, "pixel count"):
            replay_probe_frame(bad, depth, mask)

    def test_actor_in_field_can_be_completely_occluded_without_depth_samples(self):
        row, depth, mask = self.fresh_probe_fixture()
        mask[:] = False
        row["evaluation_only"]["target"] = {"visible_pixel_count": 0,
            "centroid_xy_pixels": None, "bbox_xyxy_pixels": None}
        result = replay_probe_frame(row, depth, mask)
        self.assertTrue(result["actor_pivot_inside_image"])
        self.assertEqual(result["visible_target_pixel_count"], 0)
        self.assertEqual(result["sampled_depth_ray_count"], 0)
        self.assertIsNone(result["actor_pivot_centroid_distance_px"])


if __name__ == "__main__":
    unittest.main()
