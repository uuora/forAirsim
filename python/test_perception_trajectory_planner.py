"""Numerical and failure-boundary tests for the finite quintic planner."""
import json
import unittest
from dataclasses import replace

import numpy as np

from perception_trajectory_planner import (
    PlanningParams, SphereObstacle, evaluate_plan, evaluate_quintic,
    plan_trajectory, quintic_coefficients,
)


class QuinticTests(unittest.TestCase):
    def test_boundary_position_velocity_and_acceleration(self):
        p = np.array([0.2, -0.5, -2.0])
        v = np.array([0.1, 0.2, -0.1])
        a = np.array([-0.05, 0.02, 0.03])
        endpoint = np.array([1.2, 0.7, -1.5])
        coefficients = quintic_coefficients(p, v, a, endpoint, 4.3)
        for derivative, initial in enumerate((p, v, a)):
            np.testing.assert_allclose(evaluate_quintic(coefficients, 0.0, derivative), initial, atol=1e-12)
        for derivative, terminal in enumerate((endpoint, np.zeros(3), np.zeros(3))):
            np.testing.assert_allclose(evaluate_quintic(coefficients, 4.3, derivative), terminal, atol=1e-11)

    def test_vector_evaluation_matches_scalar_evaluation(self):
        c = quintic_coefficients([0, 0, -2], [0, 0, 0], [0, 0, 0], [1, 0.2, -2], 4)
        times = np.array([0.0, 0.3, 1.7, 4.0])
        for derivative in range(4):
            np.testing.assert_allclose(evaluate_quintic(c, times, derivative),
                                       np.array([evaluate_quintic(c, t, derivative) for t in times]))


class PlannerTests(unittest.TestCase):
    def test_rgb_quality_offset_changes_only_center_cost_with_shared_constraints(self):
        from trajectory_rgb_quality_reference import replay_centering_ranking
        for mode in ('efficiency','perception'):
            legacy=self.plan(mode=mode)
            updated=self.plan(mode=mode,image_center_reference_offset_ned_m=[0,-.3,.1])
            for old,new in zip(legacy['candidates'],updated['candidates']):
                self.assertEqual(old['coefficients'],new['coefficients'])
                self.assertEqual(old['reject_reasons'],new['reject_reasons'])
                self.assertEqual(old['sampled_margins'],new['sampled_margins'])
                for name in old['cost_components']:
                    if name!='image_center':self.assertEqual(old['cost_components'][name],new['cost_components'][name])
            replay=replay_centering_ranking(legacy,[0,-.3,.1])
            self.assertEqual(updated['selected_index'],replay['proposed_selected_index'])
            if mode=='efficiency':self.assertEqual(legacy['selected_index'],updated['selected_index'])

    def test_quality_reference_cannot_replace_axial_surface_depth(self):
        self.assertEqual(self.plan(image_center_reference_offset_ned_m=[.1,0,0])['status'],'INVALID')
        self.assertEqual(self.plan(image_center_reference_offset_ned_m=[0,float('nan'),0])['status'],'INVALID')

    def setUp(self):
        self.params = PlanningParams(durations=(4.0,), max_speed_mps=1.0,
                                     max_acceleration_mps2=1.0)
        self.inputs = dict(position=[0, 0, -2], velocity=[0, 0, 0], acceleration=[0, 0, 0],
                           target_position=[3, 0, -2], target_velocity=[0, 0, 0],
                           candidate_endpoints=[[1.0, 0, -2]], params=self.params)

    def plan(self, **changes):
        return plan_trajectory(**(self.inputs | changes))

    def test_feasible_plan_and_json_log(self):
        plan = self.plan()
        self.assertEqual(plan["status"], "OK")
        json.dumps(plan, allow_nan=False)
        self.assertEqual(plan["selected_index"], 0)
        self.assertEqual(set(plan["candidates"][0]["cost_components"]),
                         {"terminal_observation", "time_efficiency", "image_center",
                          "jerk_energy", "acceleration_energy", "safety_margin"})
        sampled = evaluate_plan(plan, 0.2)
        self.assertGreater(sampled["position"][0], 0)
        with self.assertRaises(ValueError):
            evaluate_plan(plan, 4.001)

    def test_nonfinite_inputs_are_invalid(self):
        for field, value in [("position", [float("nan"), 0, -2]),
                             ("velocity", [0, float("inf"), 0]),
                             ("target_position", [3, 0, float("nan")]),
                             ("candidate_endpoints", [[1, float("nan"), -2]]),
                             ("observation_age_s", float("nan"))]:
            with self.subTest(field=field):
                plan = self.plan(**{field: value})
                self.assertEqual(plan["status"], "INVALID")
                self.assertIsNone(plan["trajectory"])
                self.assertFalse(plan["protective_action"]["validated_safe_trajectory"])

    def test_stale_estimate_has_no_trajectory_and_moving_state_requires_braking(self):
        plan = self.plan(observation_age_s=0.501, velocity=[0.1, 0, 0])
        self.assertEqual(plan["status"], "STALE")
        self.assertIsNone(plan["selected_index"])
        self.assertEqual(plan["protective_action"]["recommendation"], "CONTROLLER_BRAKE_AND_REPLAN")
        with self.assertRaises(ValueError):
            evaluate_plan(plan, 0)

    def test_initial_norm_violations_do_not_claim_hover(self):
        for field in ("velocity", "acceleration"):
            with self.subTest(field=field):
                plan = self.plan(**{field: [0.8, 0.8, 0]})
                self.assertEqual(plan["status"], "INVALID")
                self.assertEqual(plan["protective_action"]["recommendation"],
                                 "CONTROLLER_CONSTRAINT_RECOVERY_REQUIRED")

    def test_initial_out_of_bounds_and_invalid_parameters_are_explicit(self):
        plan = self.plan(position=[11, 0, -2])
        self.assertEqual(plan["status"], "INVALID")
        self.assertEqual(plan["protective_action"]["recommendation"],
                         "CONTROLLER_CONSTRAINT_RECOVERY_REQUIRED")
        for params in (replace(self.params, sample_dt_s=0),
                       replace(self.params, max_speed_mps=float("inf")),
                       replace(self.params, horizontal_fov_deg=180),
                       replace(self.params, durations=(-1.0,))):
            with self.subTest(params=params):
                rejected = self.plan(params=params)
                self.assertEqual(rejected["status"], "INVALID")
                self.assertIsNone(rejected["trajectory"])

    def test_path_speed_uses_norm_and_not_individual_axes(self):
        params = replace(self.params, durations=(2.0,), max_speed_mps=1.0,
                         max_acceleration_mps2=10.0)
        plan = self.plan(candidate_endpoints=[[0.9, 0.9, -2]], params=params)
        c = plan["candidates"][0]["coefficients"]
        velocities = evaluate_quintic(c, np.linspace(0, 2, 101), 1)
        self.assertLess(np.max(np.abs(velocities)), 1.0)
        self.assertGreater(np.max(np.linalg.norm(velocities, axis=1)), 1.0)
        self.assertEqual(plan["status"], "NO_FEASIBLE")
        self.assertIn("SPEED_LIMIT", plan["candidates"][0]["reject_reasons"])
        self.assertIsNone(plan["trajectory"])

    def test_path_acceleration_uses_norm(self):
        params = replace(self.params, durations=(2.0,), max_speed_mps=10.0,
                         max_acceleration_mps2=1.6)
        plan = self.plan(candidate_endpoints=[[1, 1, -2]], params=params)
        c = plan["candidates"][0]["coefficients"]
        acceleration = evaluate_quintic(c, np.linspace(0, 2, 101), 2)
        self.assertLess(np.max(np.abs(acceleration)), 1.6)
        self.assertGreater(np.max(np.linalg.norm(acceleration, axis=1)), 1.6)
        self.assertIn("ACCELERATION_LIMIT", plan["candidates"][0]["reject_reasons"])

    def test_no_feasible_candidate_is_not_forced(self):
        plan = self.plan(candidate_endpoints=[[4, 0, -2], [1, 11, -2]])
        self.assertEqual(plan["status"], "NO_FEASIBLE")
        self.assertIsNone(plan["selected_index"])
        self.assertTrue(all(not candidate["feasible"] for candidate in plan["candidates"]))

    def test_known_sphere_rejects_path_including_vehicle_radius(self):
        # Centerline remains 0.25 m from this sphere, but the vehicle's swept radius intersects it.
        obstacle = SphereObstacle((0.5, 0.25, -2), radius=0.1, label="known_geometry")
        plan = self.plan(obstacles=[obstacle])
        self.assertEqual(plan["status"], "NO_FEASIBLE")
        self.assertIn("SPHERE_OBSTACLE:0:known_geometry", plan["candidates"][0]["reject_reasons"])

    def test_camera_depth_fov_target_distance_and_bounds(self):
        cases = [(dict(target_position=[-1, 0, -2]), "TARGET_CAMERA_DEPTH"),
                 (dict(target_position=[3, 4, -2]), "TARGET_FOV"),
                 (dict(candidate_endpoints=[[2.7, 0, -2]]), "TARGET_MIN_DISTANCE"),
                 (dict(candidate_endpoints=[[1, 0, 2]]), "POSITION_BOUNDS")]
        for changes, reason in cases:
            with self.subTest(reason=reason):
                plan = self.plan(**changes)
                self.assertEqual(plan["status"], "NO_FEASIBLE")
                self.assertIn(reason, plan["candidates"][0]["reject_reasons"])

    def test_observation_age_advances_target_prediction(self):
        plan = self.plan(observation_age_s=0.4, target_velocity=[0.1, 0, 0])
        # At t=4 s target is 3 + 0.1*(4+0.4), endpoint is 1 m.
        expected = ((2.44 - self.params.desired_observation_distance_m)
                    / self.params.desired_observation_distance_m)**2
        self.assertAlmostEqual(plan["candidates"][0]["cost_components"]["terminal_observation"], expected)

    def test_camera_offset_changes_projection_but_not_body_center_distance(self):
        centered = self.plan(target_position=[3, 0.8, -2],
                             params=replace(self.params, camera_offset_ned_m=(0, 0.8, 0)))
        unshifted = self.plan(target_position=[3, 0.8, -2])
        self.assertEqual(centered["status"], "OK")
        self.assertEqual(unshifted["status"], "OK")
        first, second = centered["candidates"][0], unshifted["candidates"][0]
        self.assertAlmostEqual(first["cost_components"]["image_center"], 0.0)
        self.assertGreater(second["cost_components"]["image_center"], 0.0)
        self.assertEqual(first["cost_components"]["terminal_observation"],
                         second["cost_components"]["terminal_observation"])
        self.assertEqual(first["sampled_margins"]["TARGET_MIN_DISTANCE"],
                         second["sampled_margins"]["TARGET_MIN_DISTANCE"])

    def test_camera_offset_can_move_target_outside_camera_fov(self):
        normal = self.plan()
        shifted = self.plan(params=replace(self.params, camera_offset_ned_m=(0, 2.2, 0)))
        self.assertEqual(normal["status"], "OK")
        self.assertEqual(shifted["status"], "NO_FEASIBLE")
        self.assertIn("TARGET_FOV", shifted["candidates"][0]["reject_reasons"])
        self.assertEqual(shifted["candidates"][0]["sampled_margins"]["TARGET_MIN_DISTANCE"],
                         normal["candidates"][0]["sampled_margins"]["TARGET_MIN_DISTANCE"])

    def test_camera_offset_requires_finite_three_vector(self):
        for offset in ((0, 0), (0, float("nan"), 0), (0, 0, float("inf"))):
            with self.subTest(offset=offset):
                plan = self.plan(params=replace(self.params, camera_offset_ned_m=offset))
                self.assertEqual(plan["status"], "INVALID")
                self.assertIsNone(plan["trajectory"])

    def test_perception_ablation_changes_only_center_weight(self):
        params = replace(self.params, position_max=(20, 20, 1), durations=(4.0,),
                         max_speed_mps=10.0, max_acceleration_mps2=10.0,
                         horizontal_fov_deg=120.0,
                         terminal_observation_weight=0.0, time_efficiency_weight=0.0,
                         jerk_energy_weight=0.0, acceleration_energy_weight=1.0,
                         image_center_weight=10.0, safety_margin_weight=0.0,
                         acceleration_reference_mps2=10.0)
        inputs = dict(position=[0, -3, -2], target_position=[10, 0, -2],
                      candidate_endpoints=[[8, -3, -2], [8, 0, -2]], params=params)
        efficient = self.plan(mode="efficiency", **inputs)
        perceptive = self.plan(mode="perception", **inputs)
        self.assertEqual(efficient["status"], "OK")
        self.assertEqual(perceptive["status"], "OK")
        self.assertEqual(efficient["selected_index"], 0)
        self.assertEqual(perceptive["selected_index"], 1)
        for first, second in zip(efficient["candidates"], perceptive["candidates"]):
            self.assertEqual(first["coefficients"], second["coefficients"])
            self.assertEqual(first["reject_reasons"], second["reject_reasons"])
            self.assertEqual(first["sampled_margins"], second["sampled_margins"])
            self.assertEqual(first["cost_components"], second["cost_components"])
        other_weights = {key: value for key, value in efficient["cost_weights"].items() if key != "image_center"}
        self.assertEqual(other_weights,
                         {key: value for key, value in perceptive["cost_weights"].items() if key != "image_center"})
        self.assertLess(perceptive["candidates"][1]["cost_components"]["image_center"],
                        perceptive["candidates"][0]["cost_components"]["image_center"])

    def test_equal_cost_is_deterministic_and_selects_first(self):
        inputs = dict(candidate_endpoints=[[1, 0, -2], [1, 0, -2]])
        first, second = self.plan(**inputs), self.plan(**inputs)
        self.assertEqual(first, second)
        self.assertEqual(first["selected_index"], 0)
        self.assertEqual(first["candidates"][0]["total_cost"], first["candidates"][1]["total_cost"])

    def test_visible_representative_point_does_not_imply_visible_footprint(self):
        point=self.plan(target_position=[3,1.8,-2])
        footprint=self.plan(target_position=[3,1.8,-2],target_support_offsets=[[0,-.3,-.2],[0,.3,.2]])
        self.assertEqual(point['status'],'OK')
        self.assertEqual(footprint['status'],'NO_FEASIBLE')
        self.assertIn('TARGET_FOV',footprint['candidates'][0]['reject_reasons'])

    def test_footprint_and_border_margin_are_identical_hard_constraints_for_both_methods(self):
        args=dict(target_position=[3,1.5,-2],target_support_offsets=[[0,-.1,-.1],[0,.1,.1]],
                  params=replace(self.params,image_border_margin_fraction=.25))
        e=self.plan(mode='efficiency',**args);p=self.plan(mode='perception',**args)
        self.assertEqual(e['status'],'NO_FEASIBLE')
        self.assertEqual(e['candidates'][0]['reject_reasons'],p['candidates'][0]['reject_reasons'])
        self.assertEqual(e['candidates'][0]['sampled_margins'],p['candidates'][0]['sampled_margins'])

    def test_empty_or_nonfinite_footprint_is_explicitly_invalid(self):
        for supports in ([],[[0,float('nan'),0]]):
            with self.subTest(supports=supports):
                self.assertEqual(self.plan(target_support_offsets=supports)['status'],'INVALID')


if __name__ == "__main__":
    unittest.main()
