"""Failure boundaries for the new planner/controller AirSim interface."""
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from perception_trajectory_planner import PlanningParams, plan_trajectory
from trajectory_single_experiment import (
    ProtectionRequired, bounded_reference_command, candidate_endpoints,
    checked_motion_state, frame_translation, require_fixed_forward_attitude,
    TrajectorySingleExperiment,
    world_height_to_local, execution_sample,
)


class TrajectoryInterfaceTests(unittest.TestCase):
    def test_translation_keeps_camera_local_and_world_distinct(self):
        origin, error = frame_translation([-5, -.5, -3], [-5, 2.5, -2])
        np.testing.assert_allclose(origin, [0, -3, -1])
        self.assertEqual(error, 0)
        np.testing.assert_allclose(np.array([-4, 2.5, -2.05]) + origin, [-4, -.5, -3.05])
        with self.assertRaises(ProtectionRequired):
            frame_translation([-5, -.4, -3], [-5, 2.5, -2], origin, .05)

    def test_actual_body_and_camera_attitude_must_face_forward(self):
        identity = [0, 0, 0, 1]
        require_fixed_forward_attitude(identity, identity, 7.5)
        rotated = [0, 0, math.sin(math.radians(15)/2), math.cos(math.radians(15)/2)]
        for body, camera in ((rotated, identity), (identity, rotated)):
            with self.subTest(body=body):
                with self.assertRaises(ProtectionRequired):
                    require_fixed_forward_attitude(body, camera, 7.5)

    def test_motion_budget_rejects_actual_input_without_clipping(self):
        with self.assertRaises(ProtectionRequired):
            checked_motion_state([.11, .11, 0], [0, 0, 0], .15, .3)
        with self.assertRaises(ProtectionRequired):
            checked_motion_state([0, 0, 0], [.22, .22, 0], .15, .3)
        velocity, acceleration = checked_motion_state([.10, .02, 0], [.03, .01, 0], .15, .3)
        np.testing.assert_array_equal(velocity, [.10, .02, 0])
        np.testing.assert_array_equal(acceleration, [.03, .01, 0])

    def test_endpoint_grid_is_relative_identical_and_bounded(self):
        planning = {"candidate_steps_x_m": [0, .1, .2, .22],
                    "candidate_steps_y_m": [-.2, -.12, 0, .12, .2],
                    "candidate_steps_z_m": [-.04, 0, .04], "durations_s": [3]}
        first = candidate_endpoints([-5, -.5, -3], planning)
        second = candidate_endpoints([-4, -.4, -3], planning)
        self.assertEqual(len(first), 60)
        np.testing.assert_allclose(np.array(second) - first, np.tile([1, .1, 0], (60, 1)))
        with self.assertRaises(ValueError):
            candidate_endpoints([-5, -.5, -3], planning | {"durations_s": [3, 4]})

    def plan(self):
        return plan_trajectory(position=[-5, 0, -3], velocity=[0, 0, 0], acceleration=[0, 0, 0],
            target_position=[0, 0, -3], target_velocity=[0, 0, 0],
            candidate_endpoints=[[-4.8, 0, -3]],
            params=PlanningParams(durations=(3,), max_speed_mps=.15,
                                  max_acceleration_mps2=.3, desired_observation_distance_m=3.7))

    def test_positive_time_reference_moves_from_rest(self):
        plan = self.plan()
        self.assertEqual(plan["status"], "OK")
        command = bounded_reference_command(plan, [-5, 0, -3], [0, 0, 0], .75, .8, .15, .3, .35)
        self.assertGreater(command["command_velocity_ned_m_s"][0], 0)
        self.assertLessEqual(command["command_velocity_norm_m_s"], .15)
        self.assertLessEqual(command["command_change_norm_m_s"], .3 * .35 + 1e-12)

    def test_tracking_command_respects_norm_and_delta_budget(self):
        plan = self.plan()
        previous = np.array([.10, 0, 0])
        command = bounded_reference_command(plan, [-10, -10, -3], previous, .75, 10, .15, .3, .1)
        applied = np.array(command["command_velocity_ned_m_s"])
        self.assertLessEqual(np.linalg.norm(applied), .15 + 1e-12)
        self.assertLessEqual(np.linalg.norm(applied - previous), .03 + 1e-12)
        self.assertTrue(command["velocity_norm_clipped"])
        self.assertTrue(command["command_change_limited"])

    def test_planning_input_anchors_box_to_surface_estimate_not_detection_centre(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        run = self.minimal_controller()
        run.cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" /
            "trajectory_guarded_pilot_v5" / "center.json").read_text(encoding="utf-8"))
        run.planning = run.cfg["planning"]
        run.fixed_height_execution = True
        run.origin_translation = np.zeros(3)
        position = np.array([-5., -.5, -3.])
        camera = position + [1., 0., -.05]
        point = np.array([2., (517.-319.5)*2./320., (233.-239.5)*2./320.])
        def vec(values):
            return SimpleNamespace(**dict(zip(('x_val','y_val','z_val'),values)))
        state = SimpleNamespace(kinematics_estimated=SimpleNamespace(
            position=vec(position), linear_velocity=vec([0,0,0]), linear_acceleration=vec([0,0,0]),
            orientation=SimpleNamespace(x_val=0.,y_val=0.,z_val=0.,w_val=1.)))
        observation = Mock(valid=True, state_sample=None, image=np.zeros((480,640,3)),
            camera_position_local_ned_m=camera, camera_orientation_xyzw=[0,0,0,1],
            bbox_xywh=[401,145,183,185], centre_px=[491.967,240.739])
        observation.age_at.return_value = .1
        observation.estimate_world.return_value = camera + point
        row = {'frame_age_s':.1}
        inputs = run._planning_inputs(observation, row, position, state)
        supports = point + np.array(inputs['target_support_offsets'])
        pixels = [319.5,239.5] + 320.*supports[:,1:]/supports[:,0,None]
        np.testing.assert_allclose(pixels, [[401,145],[584,145],[401,330],[584,330]])
        np.testing.assert_allclose(row['footprint_anchor_pixel_xy'], [517,233])
        run.client.simGetObjectPose.assert_not_called()
        run.planning['image_center_reference_mode']='rgb_detection_centroid_axial_plane'
        rgb_inputs=run._planning_inputs(observation,{'frame_age_s':.1},position,state)
        np.testing.assert_allclose(rgb_inputs['target_position'],inputs['target_position'])
        np.testing.assert_allclose(rgb_inputs['target_support_offsets'],inputs['target_support_offsets'])
        reference_point=point+np.array(rgb_inputs['image_center_reference_offset_ned_m'])
        np.testing.assert_allclose([319.5,239.5]+320.*reference_point[1:]/reference_point[0],observation.centre_px)
        run.client.simGetObjectPose.assert_not_called()

    def minimal_controller(self):
        run = object.__new__(TrajectorySingleExperiment)
        run.vehicle = "DroneA"
        run.client = Mock()
        run.cfg = {"visual": {"max_duration_s": 90, "stable_frames": 3,
            "frame_max_age_s": .5, "stop_surface_range_m": 2.5, "alignment_error_max": .08}}
        run.planning = {"lookahead_s": .75, "position_gain": .8,
                        "execution_speed_cap_m_s": .15, "max_acceleration_m_s2": .3,
                        "command_duration_s": .35}
        run.planning_mode = "perception"
        run.target_source = "sensor_surface_estimate"
        run.fixed_height_execution = False
        run.report = {"observations": [], "planning_records": [], "protective_return_reason": None}
        run.state = Mock(return_value=(np.array([-5, 0, -3.]), 0, Mock()))
        run._settle = Mock()
        run._start_iteration = Mock(return_value={"iteration_id": "test:iteration:0"})
        run._persist_planning_record = lambda item: run.report["planning_records"].append(item)
        return run

    def row(self):
        return {"detected": True, "truncated": False, "tracking_state": "DETECTED",
                "frame_age_s": .01, "depth_m": 4., "error_xy": [.0, .0]}

    def test_invalid_sensor_input_returns_without_any_velocity_command(self):
        run = self.minimal_controller()
        row = self.row()
        run._sample_timestamped = Mock(return_value=(Mock(), row, np.array([-5, 0, -3.]), Mock()))
        run._planning_inputs = Mock(side_effect=ProtectionRequired("invalid sensor timestamp"))
        run.approach()
        self.assertFalse(run.report["task_success"])
        self.assertIn("invalid sensor timestamp", run.report["protective_return_reason"])
        run.client.moveByVelocityAsync.assert_not_called()
        run.client.hoverAsync.assert_called()

    def test_infeasible_plan_and_post_solve_stale_frame_never_execute(self):
        for mode in ("infeasible", "stale"):
            with self.subTest(mode=mode):
                run = self.minimal_controller()
                observation = Mock()
                observation.age_at.return_value = .501
                run._sample_timestamped = Mock(return_value=(observation, self.row(), np.array([-5, 0, -3.]), Mock()))
                run._planning_inputs = Mock(return_value={})
                plan = {"status": "NO_FEASIBLE", "reason": "no candidates", "candidates": []} if mode == "infeasible" else self.plan()
                with patch("trajectory_single_experiment.plan_trajectory", return_value=plan):
                    run.approach()
                self.assertFalse(run.report["task_success"])
                self.assertEqual(len(run.report["planning_records"]), 1)
                run.client.moveByVelocityAsync.assert_not_called()

    def test_collision_safety_error_propagates_without_new_flight_commands(self):
        run = self.minimal_controller()
        run._settle.side_effect = RuntimeError("Unexpected collision with obstacle")
        with self.assertRaisesRegex(RuntimeError, "Unexpected collision"):
            run.approach()
        self.assertFalse(run.report["task_success"])
        run.client.moveByVelocityAsync.assert_not_called()
        run.client.hoverAsync.assert_not_called()

    def test_fixed_height_candidates_do_not_follow_body_altitude_drift(self):
        planning = {"candidate_steps_x_m": [0, .1], "candidate_steps_y_m": [0, .1],
                    "candidate_steps_z_m": [0], "durations_s": [3],
                    "execution_mode": "fixed_world_height_preview", "fixed_world_height_ned_m": -3}
        first = candidate_endpoints([-5, -.5, -2.97], planning)
        drifted = candidate_endpoints([-5, -.5, -2.60], planning)
        np.testing.assert_array_equal(first, drifted)
        self.assertTrue(all(point[2] == -3 for point in first))
        with self.assertRaises(ValueError):
            candidate_endpoints([-5, -.5, -3], planning | {"candidate_steps_z_m": [-.04, 0, .04]})

    def test_local_height_uses_measured_origin_not_settings_spawn_z(self):
        local = world_height_to_local(-3, [0, -3, -.853504002])
        self.assertAlmostEqual(local, -2.146495998)
        with self.assertRaises(ProtectionRequired):
            world_height_to_local(-3, [0, 0, float("nan")])

    def test_physical_guard_distinguishes_command_budget_from_actual_acceleration(self):
        from types import SimpleNamespace
        state = SimpleNamespace(timestamp=100,
            kinematics_estimated=SimpleNamespace(
                linear_velocity=SimpleNamespace(x_val=.05, y_val=0., z_val=0.),
                linear_acceleration=SimpleNamespace(x_val=.45, y_val=0., z_val=0.),
                orientation=SimpleNamespace(x_val=0., y_val=0., z_val=0., w_val=1.)))
        planning = {"fixed_world_height_ned_m": -3, "execution_speed_cap_m_s": .15,
                    "max_acceleration_m_s2": .3, "execution_guard_margin_m": .15,
                    "fixed_height_max_error_m": .12, "attitude_limit_deg": 7.5}
        sample = execution_sample([-5, -.5, -3], state, planning, [-5.6, -1.3, -3.6], [-2.4, .6, -2.4])
        self.assertFalse(sample["over_planning_speed_budget"])
        self.assertIn("ACTUAL_ACCELERATION_BUDGET_EXCEEDED", sample["protection_reasons"])

    def test_protection_margin_stops_remaining_execution_chunks(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        state = SimpleNamespace(timestamp=100, kinematics_estimated=SimpleNamespace(
            linear_velocity=SimpleNamespace(x_val=0., y_val=0., z_val=0.),
            linear_acceleration=SimpleNamespace(x_val=0., y_val=0., z_val=0.),
            orientation=SimpleNamespace(x_val=0., y_val=0., z_val=0., w_val=1.)))
        run = self.minimal_controller()
        run.origin_translation = np.array([0, -3, -.853504002])
        run.cfg.update(workspace_min_ned_m=[-5.6, -1.3, -3.6], workspace_max_ned_m=[-2.4, .6, -2.4])
        run.planning.update(fixed_world_height_ned_m=-3, execution_sample_dt_s=.1,
            execution_ramp_acceleration_m_s2=.15, execution_guard_margin_m=.15,
            fixed_height_max_error_m=.12, attitude_limit_deg=7.5)
        run.state.return_value = (np.array([-2.50, 0, -3]), 0, state)
        run._record_path = Mock()
        command = {"command_velocity_ned_m_s": [.05, 0, 0]}
        record = {"plan_id": "test:plan:0"}
        with TemporaryDirectory() as directory:
            run.folder = Path(directory)
            with self.assertRaisesRegex(ProtectionRequired, "PROTECTION_MARGIN"):
                run._execute_fixed_height(command, record, state)
        self.assertEqual(run.client.moveByVelocityZAsync.call_count, 1)
        run.client.moveByVelocityAsync.assert_not_called()
        run.client.cancelLastTask.assert_called_once()

    def test_continuous_goal_keeps_same_command_and_measured_height_without_join(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        run = self.minimal_controller()
        run.fixed_height_execution = True
        run.active_xy_command = np.array([.03, -.01])
        run.origin_translation = np.array([0, -3, -.853504002])
        run.planning.update(fixed_world_height_ned_m=-3, continuous_goal_keepalive_s=.5)
        run.report["goal_keepalive_records"] = []
        with TemporaryDirectory() as directory:
            run.folder = Path(directory)
            run._maintain_fixed_goal("capture")
        args, kwargs = run.client.moveByVelocityZAsync.call_args
        np.testing.assert_allclose(args, [.03, -.01, -2.146495998])
        self.assertEqual(kwargs["duration"], .5)
        run.client.moveByVelocityZAsync.return_value.join.assert_not_called()
        self.assertFalse(run.report["goal_keepalive_records"][0]["join_called"])
        run.active_xy_command = np.array([.15, .15])
        with self.assertRaises(ProtectionRequired):
            run._maintain_fixed_goal("invalid")
        self.assertEqual(run.client.moveByVelocityZAsync.call_count, 1)

    def test_rejected_post_capture_acceleration_is_preserved_before_planning(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        def state(acceleration):
            return SimpleNamespace(timestamp=100, kinematics_estimated=SimpleNamespace(
                linear_velocity=SimpleNamespace(x_val=0., y_val=0., z_val=0.),
                linear_acceleration=SimpleNamespace(x_val=acceleration, y_val=0., z_val=0.),
                orientation=SimpleNamespace(x_val=0., y_val=0., z_val=0., w_val=1.)))
        run = self.minimal_controller()
        run.fixed_height_execution = True
        run.planning.update(fixed_world_height_ned_m=-3, execution_guard_margin_m=.15,
            fixed_height_max_error_m=.12, attitude_limit_deg=7.5)
        run.cfg.update(workspace_min_ned_m=[-5.6, -1.3, -3.6], workspace_max_ned_m=[-2.4, .6, -2.4])
        run.state.side_effect = [(np.array([-5, -.5, -3]), 0, state(0)),
                                 (np.array([-5, -.5, -3]), 0, state(.45))]
        run._record_path = Mock()
        observation = Mock(image=None, depth=None, valid=True, detected=True,
            truncated=False, state_sample=None, candidate_count=1, confidence=.9,
            normalised_error_xy=[0, 0], depth_m=3.)
        observation.age_at.return_value = .1
        observation.metadata.return_value = {}
        run.timestamped_camera = Mock()
        run.timestamped_camera.capture.return_value = observation
        run.tracker = Mock()
        run.tracker.update.return_value.state.value = "DETECTED"
        with TemporaryDirectory() as directory:
            run.folder = Path(directory)
            _, row, _, _ = run._sample_timestamped()
        self.assertEqual(len(run.report["observations"]), 1)
        sample = row["planning_input_body_state"]
        self.assertAlmostEqual(sample["acceleration_norm_m_s2"], .45)
        self.assertIn("ACTUAL_ACCELERATION_BUDGET_EXCEEDED", sample["protection_reasons"])
        self.assertEqual(row["pre_capture_body_state"]["acceleration_norm_m_s2"], 0)

    def test_unaccepted_start_returns_before_any_method_observation_or_command(self):
        run = self.minimal_controller()
        run.cfg["initial_state_match"] = {"enabled": True}
        run._prepare_matched_start = Mock(side_effect=ProtectionRequired("Initial-state matching timed out"))
        run._sample_timestamped = Mock()
        run.approach()
        run._sample_timestamped.assert_not_called()
        run._start_iteration.assert_not_called()
        run.client.moveByVelocityAsync.assert_not_called()
        run.client.moveByVelocityZAsync.assert_not_called()
        run.client.hoverAsync.assert_called_once()
        self.assertFalse(run.report["task_success"])
        self.assertIsNone(run.report["metrics"]["approach_duration_s"])
        self.assertIsNone(run.report["metrics"]["approach_path_m"])

    def test_accepted_start_does_not_run_a_second_settle_before_first_observation(self):
        run = self.minimal_controller()
        run.fixed_height_execution = True
        run.cfg["initial_state_match"] = {"enabled": True}
        run._prepare_matched_start = Mock()
        run._sample_timestamped = Mock(return_value=(Mock(), self.row(), np.array([-5, 0, -3.]), Mock()))
        run._planning_inputs = Mock(side_effect=ProtectionRequired("invalid sensor timestamp"))
        run.approach()
        run._prepare_matched_start.assert_called_once()
        run._settle.assert_not_called()
        run._sample_timestamped.assert_called_once()
        self.assertFalse(run.report["task_success"])

    def guarded_controller(self):
        from types import SimpleNamespace
        run=self.minimal_controller()
        state=SimpleNamespace(timestamp=100,kinematics_estimated=SimpleNamespace(
            linear_velocity=SimpleNamespace(x_val=0.,y_val=0.,z_val=0.),
            linear_acceleration=SimpleNamespace(x_val=0.,y_val=0.,z_val=0.),
            orientation=SimpleNamespace(x_val=0.,y_val=0.,z_val=0.,w_val=1.)))
        run.guard_cfg=dict(horizon_s=.5,position_reserve_m=.02,freshness_reserve_s=.06,
            minimum_command_s=.03,brake_timeout_s=20.,max_freshness_replans=6)
        run.planning.update(execution_envelope=run.guard_cfg,fixed_world_height_ned_m=-3,
            command_duration_s=.15,execution_sample_dt_s=.05,execution_ramp_acceleration_m_s2=.15,
            execution_guard_margin_m=.15,fixed_height_max_error_m=.12,attitude_limit_deg=7.5,
            settle_speed_m_s=.025,settle_acceleration_m_s2=.08,settle_stable_s=.25)
        run.cfg.update(workspace_min_ned_m=[-5.6,-1.3,-3.6],workspace_max_ned_m=[-2.4,.6,-2.4])
        run.state.return_value=(np.array([-5.,-.5,-3.]),0.,state)
        run.origin_translation=np.array([0,-3,-.8535])
        run.active_xy_command=np.zeros(2);run.active_observation=Mock();run.freshness_replans=0
        run.report.update(execution_guard_samples=[],settling_records=[],freshness_replan_events=[],goal_keepalive_records=[])
        run._record_path=Mock()
        return run,state

    def test_expiring_frame_brakes_and_reobserves_without_target_rpc(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        run,state=self.guarded_controller()
        observation=Mock();observation.age_at.return_value=.49
        run._freshness_brake_and_reobserve=Mock(return_value=(np.array([-5.,-.5,-3.]),state))
        record={'plan_id':'expiry'}
        with TemporaryDirectory() as directory:
            run.folder=Path(directory)
            run._execute_fixed_height({'command_velocity_ned_m_s':[.05,0,0]},record,state,observation)
        run.client.moveByVelocityZAsync.assert_not_called()
        run._freshness_brake_and_reobserve.assert_called_once()
        self.assertEqual(record['execution_status'],'SELF_STATE_BRAKE_AND_REOBSERVE')

    def test_rpc_latency_spike_does_not_dispatch_another_stale_target_segment(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        run,state=self.guarded_controller()
        observation=Mock();observation.age_at.side_effect=[.1,.1,.6,.6,.6]
        run._freshness_brake_and_reobserve=Mock(return_value=(np.array([-5.,-.5,-3.]),state))
        record={'plan_id':'latency'}
        with TemporaryDirectory() as directory:
            run.folder=Path(directory)
            run._execute_fixed_height({'command_velocity_ned_m_s':[.05,0,0]},record,state,observation)
        self.assertEqual(run.client.moveByVelocityZAsync.call_count,1)
        self.assertEqual(len(record['execution_samples']),1)
        self.assertEqual(record['execution_samples'][0]['observation_age_after_rpc_and_state_s'],.6)
        run._freshness_brake_and_reobserve.assert_called_once()

    def test_self_state_brake_discards_target_and_keeps_bounded_command_ramp(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        run,_=self.guarded_controller()
        run.active_xy_command=np.array([.02,0.])
        old_observation=run.active_observation
        with TemporaryDirectory() as directory,patch('trajectory_single_experiment.time.monotonic',side_effect=np.arange(1000)*.05):
            run.folder=Path(directory)
            run._freshness_brake_and_reobserve('test',.49)
        self.assertIsNone(run.active_observation)
        old_observation.age_at.assert_not_called()
        run.client.simGetObjectPose.assert_not_called()
        self.assertEqual(run.report['freshness_replan_events'][0]['status'],'BRAKED_READY_TO_REOBSERVE')
        velocities=[.02]+[c['command_velocity_xy_m_s'][0] for c in run.report['goal_keepalive_records']]
        self.assertTrue(all(abs(a-b) <= .15*.05+1e-10 for a,b in zip(velocities,velocities[1:])))
        self.assertAlmostEqual(velocities[-1],0.)


if __name__ == "__main__":
    unittest.main()
