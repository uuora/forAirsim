"""Batch/scalar equivalence and optional saved-schema performance receipt.

Run unittest normally. ``--benchmark-report PATH --rounds 20`` reads only a
saved development receipt and prints JSON; no simulator, API or online detector
is called. Timing is offline projection arithmetic, not flight-loop latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from perception_trajectory_planner import evaluate_quintic, quintic_coefficients
from t1_pose_geometry import CameraPose, SceneCameraModel, project_footprint_polygon, project_world
from t1_pose_projection_batch import score_pose_path


def scalar_score(reference, corners, positions, origin, pose, camera):
    centers, ranges, fractions = [], [], []
    for position in positions:
        sampled_pose = CameraPose(np.asarray(pose.position_local_ned_m)+(position-origin),
                                 pose.orientation_xyzw, pose.world_from_local_translation_ned_m)
        point = project_world(reference, sampled_pose, camera)
        footprint = project_footprint_polygon(corners, sampled_pose, camera)
        centers.append(point["centering_squared_error"])
        ranges.append(point["ray_range_m"])
        fractions.append(footprint["field_area_fraction"])
    return {"centering_squared_errors": centers, "ray_ranges_m": ranges,
            "field_area_fractions": fractions}


def camera_world_points(pose, depth=3.0, y=0.0, z=0.0, half=.15):
    reference = pose.camera_to_world([depth, y, z])
    corners = np.array([pose.camera_to_world(point) for point in
        ([depth, y-half, z-half], [depth, y+half, z-half],
         [depth, y+half, z+half], [depth, y-half, z+half])])
    return reference, corners


class ProjectionBatchTests(unittest.TestCase):
    def setUp(self):
        self.camera = SceneCameraModel(640, 480, 320.0, 320.0, 320.0, 240.0)
        self.pose = CameraPose([1.0, 2.0, -3.0], [0.0, 0.0, 0.0, 1.0], [5.0, -7.0, 4.0])
        self.origin = np.array([5.0, -3.0, -3.0])
        self.positions = self.origin+np.array([[0.0, 0.0, 0.0], [.024, 0.0, 0.0],
                                             [0.0, -.024, 0.0], [.01, .01, .0]])
        self.reference, self.corners = camera_world_points(self.pose)

    def batch(self, **overrides):
        args = dict(reference_world=self.reference, corners_world=self.corners,
            positions_ned=self.positions, origin_body_ned=self.origin,
            camera_pose=self.pose, camera=self.camera)
        args.update(overrides)
        return score_pose_path(**args)

    def assertEquivalent(self, reference, corners, positions, origin, pose, camera):
        batch = score_pose_path(reference, corners, positions, origin, pose, camera)
        scalar = scalar_score(reference, corners, positions, origin, pose, camera)
        self.assertEqual(set(batch), set(scalar))
        for key in scalar:
            self.assertIsInstance(batch[key], list)
            np.testing.assert_allclose(batch[key], scalar[key], rtol=1e-12, atol=1e-12)
        return batch

    def test_forward_pose_with_explicit_camera_offset_and_world_translation(self):
        result = self.assertEquivalent(self.reference, self.corners, self.positions,
                                      self.origin, self.pose, self.camera)
        self.assertEqual(result["field_area_fractions"], [1.0]*len(self.positions))
        self.assertAlmostEqual(result["ray_ranges_m"][0], 3.0)

    def test_all_inside_skips_original_clipping_only_after_polygon_validation(self):
        with patch("t1_pose_projection_batch.project_footprint_polygon", side_effect=AssertionError("unexpected clip")):
            result = self.batch()
        self.assertEqual(result["field_area_fractions"], [1.0]*len(self.positions))

    def test_random_camera_rotations_and_translated_body_paths(self):
        rng = np.random.default_rng(6100424)
        for case in range(30):
            raw = rng.normal(size=4)
            quaternion = raw/np.linalg.norm(raw)
            # Every axis, including roll and pitch, participates. World/local
            # translation differs from both the body and camera position.
            pose = CameraPose(rng.normal(size=3)*2, quaternion, rng.normal(size=3)*20)
            origin = rng.normal(size=3)*10
            positions = origin+rng.normal(size=(21, 3))*.01
            reference, corners = camera_world_points(pose, depth=2+rng.random()*4,
                y=float(rng.uniform(-.2, .2)), z=float(rng.uniform(-.15, .15)))
            with self.subTest(case=case):
                self.assertEquivalent(reference, corners, positions, origin, pose, self.camera)

    def test_near_unit_quaternion_normalization_matches_original(self):
        pose = CameraPose([1.0, 2.0, -3.0], [0.0, 0.0, .70710678*1.01, .70710678*1.01], [5.0, 6.0, 7.0])
        reference, corners = camera_world_points(pose)
        self.assertEquivalent(reference, corners, self.positions, self.origin, pose, self.camera)

    def test_large_world_translation_preserves_operation_order(self):
        pose = CameraPose([1.0, 2.0, -3.0], [0.0, 0.0, 0.0, 1.0], [1e8, -1e8, 1e8])
        reference, corners = camera_world_points(pose)
        self.assertEquivalent(reference, corners, self.positions, self.origin, pose, self.camera)

    def test_partial_clip_uses_original_fraction(self):
        reference, corners = camera_world_points(self.pose, depth=3.0, y=2.95, half=.25)
        with patch("t1_pose_projection_batch.project_footprint_polygon", wraps=project_footprint_polygon) as fallback:
            result = self.assertEquivalent(reference, corners, self.positions, self.origin, self.pose, self.camera)
            self.assertEqual(fallback.call_count, len(self.positions))
        self.assertTrue(all(0.0 < item < 1.0 for item in result["field_area_fractions"]))

    def test_fully_outside_polygon_returns_zero_without_clamping_projection(self):
        reference, corners = camera_world_points(self.pose, depth=3.0, y=5.0, half=.25)
        result = self.assertEquivalent(reference, corners, self.positions, self.origin, self.pose, self.camera)
        self.assertEqual(result["field_area_fractions"], [0.0]*len(self.positions))
        self.assertTrue(all(value > 1 for value in result["centering_squared_errors"]))

    def test_exact_image_edge_uses_original_clipping_contract(self):
        reference, corners = camera_world_points(self.pose, depth=3.0, y=2.75, half=.25)
        positions = self.origin[None, :]
        with patch("t1_pose_projection_batch.project_footprint_polygon", wraps=project_footprint_polygon) as fallback:
            result = self.assertEquivalent(reference, corners, positions, self.origin, self.pose, self.camera)
            self.assertEqual(fallback.call_count, 1)  # x==width is not inside_image
        self.assertEqual(result["field_area_fractions"], [1.0])

    def test_mixed_inside_and_partial_samples_keep_input_order(self):
        reference, corners = camera_world_points(self.pose, depth=3.0, y=2.7, half=.15)
        positions = self.origin+np.array([[0.0, 0.0, 0.0], [0.0, -.3, 0.0], [0.0, .3, 0.0]])
        result = self.assertEquivalent(reference, corners, positions, self.origin, self.pose, self.camera)
        self.assertEqual(result["field_area_fractions"][0], 1.0)
        self.assertLess(result["field_area_fractions"][1], 1.0)
        self.assertEqual(result["field_area_fractions"][2], 1.0)

    def test_reference_behind_and_on_near_plane_rejected(self):
        for depth in (-1.0, 0.0, .05):
            with self.subTest(depth=depth):
                reference = self.pose.camera_to_world([depth, 0.0, 0.0])
                with self.assertRaisesRegex(ValueError, "near-depth"):
                    self.batch(reference_world=reference, positions_ned=self.origin[None, :])

    def test_one_corner_near_crossing_rejects_complete_candidate(self):
        corners = self.corners.copy()
        corners[0] = self.pose.camera_to_world([.04, -.15, -.15])
        with self.assertRaisesRegex(ValueError, "near-depth"):
            self.batch(corners_world=corners)
        with self.assertRaisesRegex(ValueError, "near-depth"):
            scalar_score(self.reference, corners, self.positions, self.origin, self.pose, self.camera)

    def test_later_sample_reference_crosses_near_and_is_not_clamped(self):
        positions = self.origin+np.array([[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "near-depth"):
            self.batch(positions_ned=positions)

    def test_degenerate_inframe_polygon_is_not_fast_path_accepted(self):
        collinear = np.array([self.pose.camera_to_world([3.0, y, 0.0]) for y in (-.1, 0.0, .1, .2)])
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            self.batch(corners_world=collinear)
        duplicate = self.corners.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            self.batch(corners_world=duplicate)

    def test_self_intersecting_star_inside_image_is_rejected(self):
        angles = np.arange(5)*2*np.pi/5
        vertices = np.array([self.pose.camera_to_world([3.0, .15*np.cos(t), .15*np.sin(t)]) for t in angles])
        star = vertices[[0, 2, 4, 1, 3]]
        with self.assertRaisesRegex(ValueError, "self-intersecting"):
            self.batch(corners_world=star)
        with self.assertRaisesRegex(ValueError, "self-intersecting"):
            scalar_score(self.reference, star, self.positions, self.origin, self.pose, self.camera)

    def test_nonconvex_and_bowtie_polygons_rejected(self):
        for order in ((0, 2, 1, 3), (0, 3, 1, 2)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.batch(corners_world=self.corners[list(order)])

    def test_clockwise_polygon_matches_original(self):
        self.assertEquivalent(self.reference, self.corners[::-1], self.positions,
                              self.origin, self.pose, self.camera)

    def test_three_and_five_vertex_convex_polygons_match(self):
        for count in (3, 5):
            angles = np.arange(count)*2*np.pi/count
            vertices = np.array([self.pose.camera_to_world([3.0, .15*np.cos(t), .15*np.sin(t)]) for t in angles])
            with self.subTest(count=count):
                self.assertEquivalent(self.reference, vertices, self.positions, self.origin, self.pose, self.camera)

    def test_nonfinite_and_bad_shapes_rejected(self):
        for overrides in (dict(reference_world=[np.nan, 0, 0]), dict(origin_body_ned=[0, np.inf, 0]),
            dict(positions_ned=[]), dict(positions_ned=[[0, 0]]), dict(positions_ned=[[0, np.nan, 0]]),
            dict(corners_world=[[0, 0, 0]]), dict(corners_world=[[0, 0]]), dict(camera_pose={}), dict(camera=None)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.batch(**overrides)

    def test_single_sample_and_no_input_mutation(self):
        reference, corners, positions, origin = [np.array(value, copy=True) for value in
            (self.reference, self.corners, self.positions, self.origin)]
        self.assertEquivalent(reference, corners, positions[:1], origin, self.pose, self.camera)
        for before, after in zip((self.reference, self.corners, self.positions, self.origin),
                                 (reference, corners, positions, origin)):
            np.testing.assert_array_equal(before, after)


def saved_schema_inputs(report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))
    rows = [row for row in report["frames"] if row.get("surface", {}).get("status") == "SURFACE_GEOMETRY_ACCEPTED"
            and row.get("planner", {}).get("status") == "POSE_TRAJECTORY_SELECTED"]
    if not rows:
        raise ValueError("saved report has no actual surface and selected trajectory")
    inputs = []
    for row in rows:
        pose_record = row["response_camera_poses"][0]
        pose = CameraPose(pose_record["position_local_ned_m"], pose_record["orientation_xyzw"],
                          row["world_from_local_translation_ned_m"])
        camera = SceneCameraModel.from_k(row["scene_intrinsics"]["K_nominal"], *row["image_size_wh"])
        coefficients = np.asarray(row["planner"]["trajectory"]["coefficients"], dtype=float)
        origin, velocity, acceleration = coefficients[0], coefficients[1], coefficients[2]*2
        config = row["planner"]["configuration"]
        paths = []
        for candidate in row["planner"]["candidates"]:
            duration = candidate["duration_s"]
            times = np.linspace(0.0, duration, int(np.ceil(duration/config["sample_dt_s"]))+1)
            coefficients = quintic_coefficients(origin, velocity, acceleration, candidate["endpoint_ned_m"], duration)
            positions = evaluate_quintic(coefficients, times)
            paths.append((row["surface"]["rgb_reference_world_ned_m"],
                          row["surface"]["bbox_plane_corners_world_ned_m"], positions, origin, pose, camera))
        inputs.append((row["frame_id"], paths, row["planner"]["candidates"], config))
    return inputs


def benchmark_saved_schema(report_path, rounds=20):
    if isinstance(rounds, bool) or rounds < 2:
        raise ValueError("benchmark requires at least two counterbalanced rounds")
    report_path = Path(report_path).resolve()
    records = saved_schema_inputs(report_path)
    differences = dict(centering_squared_errors=0.0, ray_ranges_m=0.0, field_area_fractions=0.0)
    recorded_center_difference = 0.0
    for _, paths, candidates, _ in records:
        for arguments, candidate in zip(paths, candidates):
            original = scalar_score(*arguments)
            batch = score_pose_path(*arguments)
            for key in original:
                diff = float(np.max(np.abs(np.asarray(original[key])-batch[key])))
                differences[key] = max(differences[key], diff)
                np.testing.assert_allclose(batch[key], original[key], rtol=1e-12, atol=1e-12)
            if candidate.get("feasible"):
                recorded_center_difference = max(recorded_center_difference,
                    abs(float(np.mean(original["centering_squared_errors"]))-candidate["cost_terms"]["image_center"]))
    frame_id, paths, _, config = records[0]
    functions = {"scalar": scalar_score, "batch": score_pose_path}
    # Warm each implementation once before the paired timing; the exact saved
    # frame, eighteen candidates and .1-second samples remain the same.
    for function in functions.values():
        for arguments in paths:
            function(*arguments)
    timings = []
    for index in range(rounds):
        order = ["scalar", "batch"] if index % 2 == 0 else ["batch", "scalar"]
        timing = {"round": index+1, "order": order}
        for name in order:
            started = time.perf_counter_ns()
            for arguments in paths:
                functions[name](*arguments)
            timing[name+"_ns"] = time.perf_counter_ns()-started
        timings.append(timing)
    scalar_ms = np.array([row["scalar_ns"] for row in timings])/1e6
    batch_ms = np.array([row["batch_ns"] for row in timings])/1e6
    def binding(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"schema_version": 1, "kind": "T1_OFFLINE_BATCH_POSE_PROJECTION_BENCHMARK",
        "status": "SCALAR_BATCH_EQUIVALENCE_CHECKED", "source_report": binding(report_path),
        "source_files": [binding(Path(__file__).resolve()), binding(Path(__file__).with_name("t1_pose_projection_batch.py")),
                         binding(Path(__file__).with_name("t1_pose_geometry.py")),
                         binding(Path(__file__).with_name("perception_trajectory_planner.py"))],
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "benchmark_frame_id": frame_id, "counterbalanced_rounds": rounds,
        "equivalence_saved_frames_checked": len(records), "candidate_count_per_round": len(paths),
        "sample_dt_s": config["sample_dt_s"], "sample_count_per_round": sum(len(row[2]) for row in paths),
        "maximum_scalar_batch_absolute_differences": differences,
        "maximum_original_recorded_center_cost_difference": recorded_center_difference,
        "scalar_median_ms": float(np.median(scalar_ms)), "scalar_p95_ms": float(np.percentile(scalar_ms, 95)),
        "batch_median_ms": float(np.median(batch_ms)), "batch_p95_ms": float(np.percentile(batch_ms, 95)),
        "median_projection_speedup_ratio": float(np.median(scalar_ms)/np.median(batch_ms)),
        "timing_receipts": timings, "new_simulator_flights": 0, "new_API_calls": 0,
        "formal_efficacy_result": False, "thresholds_or_candidates_changed": False,
        "scope": "offline geometry scoring only; not onboard loop timing or proof of flight effectiveness"}


if __name__ == "__main__":
    if "--benchmark-report" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--benchmark-report", required=True, type=Path)
        parser.add_argument("--rounds", type=int, default=20)
        args = parser.parse_args()
        print(json.dumps(benchmark_saved_schema(args.benchmark_report, args.rounds), ensure_ascii=False, allow_nan=False))
    else:
        unittest.main()
