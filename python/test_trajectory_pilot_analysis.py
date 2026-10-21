from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_trajectory_pilot import build_overview, main, summarize_matched_pairs, summarize_run


class PilotOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.report = {"policy": "efficiency", "target_source": "geometry_oracle", "status": "PASS",
                       "configuration": {"planning": {"execution_speed_cap_m_s": .15}},
                       "task_success": False, "landing_evidence": {"landed_flag": True, "api_control_released": True}}

    def test_safe_landing_does_not_count_as_task_success(self):
        row = summarize_run(self.report)
        self.assertTrue(row["safe_landing_and_release"])
        self.assertFalse(row["task_success"])
        self.assertIsNone(row["solver_p95_s"])

    def test_guard_query_budget_exceedance_and_brake_commands_are_not_dropped(self):
        report=self.motion_report()
        report['execution_guard_samples']=[{'velocity_ned_m_s':[.2,0,0],'acceleration_ned_m_s2':[0,0,0]}]
        report['goal_keepalive_records']=[{'reason':'SELF_STATE_BRAKE_WITHOUT_TARGET','command_velocity_xy_m_s':[.12,.12],'join_called':True}]
        row=summarize_run(report)
        self.assertEqual(row['guard_input_speed_budget_exceedances'],1)
        self.assertEqual(row['observed_approach_speed_budget_exceedances'],1)
        self.assertEqual(row['physical_sampled_budget_status'],'OBSERVED_EXCEEDANCE')
        self.assertEqual(row['command_norm_violations'],1)

    def test_status_pass_without_release_evidence_is_incomplete(self):
        report = deepcopy(self.report)
        report["landing_evidence"]["api_control_released"] = False
        self.assertFalse(summarize_run(report)["safe_landing_and_release"])

    def test_vector_command_limit_counts_norm_violation(self):
        report = deepcopy(self.report)
        report["planning_records"] = [{"command": {"command_velocity_ned_m_s": [.12, .12, 0]},
                                       "solver_elapsed_s": .1, "plan": {"status": "OK"}}]
        row = summarize_run(report)
        self.assertEqual(row["command_norm_violations"], 1)
        self.assertGreater(row["max_command_norm_m_s"], .15)

    def motion_report(self):
        report = deepcopy(self.report)
        report["configuration"]["planning"]["max_acceleration_m_s2"] = .3
        report["configuration"]["planning"]["body_execution_mode"] = "moveByVelocityZAsync_constant_world_height"
        report["configuration"]["planning"]["execution_mode"] = "fixed_world_height_preview"
        return report

    def test_compliant_commands_and_safe_landing_can_have_actual_budget_exceedance(self):
        report = self.motion_report()
        report["planning_records"] = [{
            "command": {"command_velocity_ned_m_s": [.10, .0, .0]},
            "execution_status": "RPC_COMPLETED",
            "executed_body_state": {"world_ned_m": [-4.8, -.1, -3.0],
                "velocity_ned_m_s": [.12, .12, .0], "speed_norm_m_s": .1,
                "acceleration_ned_m_s2": [.22, .22, .0], "acceleration_norm_m_s2": .1,
                "over_planning_speed_budget": False, "over_planning_acceleration_budget": False},
        }]
        row = summarize_run(report)
        self.assertTrue(row["recorded_command_norm_within_budget"])
        self.assertEqual(row["command_norm_violations"], 0)
        self.assertTrue(row["safe_landing_and_release"])
        self.assertEqual(row["execution_speed_budget_exceedances"], 1)
        self.assertEqual(row["execution_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["physical_sampled_budget_status"], "OBSERVED_EXCEEDANCE")
        self.assertAlmostEqual(row["execution_max_speed_norm_m_s"], (.12 ** 2 + .12 ** 2) ** .5)

    def test_subsegment_samples_take_precedence_over_endpoint_without_double_counting(self):
        report = self.motion_report()
        report["planning_records"] = [{
            "command": {"command_velocity_ned_m_s": [.1, 0, 0]},
            "execution_samples": [
                {"speed_norm_m_s": .1, "acceleration_norm_m_s2": .2},
                {"speed_norm_m_s": .2, "acceleration_norm_m_s2": .4},
            ],
            "executed_body_state": {"speed_norm_m_s": 9., "acceleration_norm_m_s2": 9.},
        }]
        row = summarize_run(report)
        self.assertEqual(row["execution_sample_count"], 2)
        self.assertEqual(row["execution_sample_sources"]["legacy_executed_body_state"], 0)
        self.assertEqual(row["execution_speed_budget_exceedances"], 1)
        self.assertEqual(row["execution_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["execution_max_speed_norm_m_s"], .2)

    def test_distinct_endpoint_simulator_query_counts_despite_inherited_host_time(self):
        report = self.motion_report()
        report["planning_records"] = [{
            "execution_samples": [{"speed_norm_m_s": .1, "acceleration_norm_m_s2": .2,
                                   "query_state_timestamp_ns": 100, "host_monotonic_s": 10.0}],
            "executed_body_state": {"speed_norm_m_s": .2, "acceleration_norm_m_s2": .4,
                                    "query_state_timestamp_ns": 101, "host_monotonic_s": 10.0},
        }]
        row = summarize_run(report)
        self.assertEqual(row["execution_sample_count"], 2)
        self.assertEqual(row["execution_sample_sources"]["distinct_execution_end_query"], 1)
        self.assertEqual(row["execution_speed_budget_exceedances"], 1)
        self.assertEqual(row["execution_acceleration_budget_exceedances"], 1)

    def test_equal_or_missing_endpoint_timestamp_cannot_create_independent_query(self):
        for endpoint_stamp, sample_stamp in ((100, 100), (None, 100), (101, None)):
            with self.subTest(endpoint_stamp=endpoint_stamp, sample_stamp=sample_stamp):
                report = self.motion_report()
                report["planning_records"] = [{
                    "execution_samples": [{"speed_norm_m_s": .1, "acceleration_norm_m_s2": .2,
                                           "query_state_timestamp_ns": sample_stamp}],
                    "executed_body_state": {"speed_norm_m_s": 9., "acceleration_norm_m_s2": 9.,
                                            "query_state_timestamp_ns": endpoint_stamp, "host_monotonic_s": 11.},
                }]
                row = summarize_run(report)
                self.assertEqual(row["execution_sample_count"], 1)
                self.assertEqual(row["execution_sample_sources"]["distinct_execution_end_query"], 0)
                self.assertEqual(row["execution_speed_budget_exceedances"], 0)

    def matching_report(self, policy="efficiency"):
        report = self.motion_report()
        report["policy"] = policy
        specification = {"position_tolerance_m": .04, "speed_tolerance_m_s": .02,
                         "acceleration_tolerance_m_s2": .04, "attitude_tolerance_deg": 2.5,
                         "stable_s": .5, "max_sample_gap_s": .25}
        sample = {"world_ned_m": [-5., -.5, -3.], "velocity_ned_m_s": [.01, 0, 0],
                  "acceleration_ned_m_s2": [.02, 0, 0], "actual_body_rpy_deg": [0., 0., 0.],
                  "accepted": True, "rejection_reasons": [], "previous_command_velocity_xy_m_s": [0., 0.]}
        samples = []
        for index in range(4):
            value = deepcopy(sample)
            value.update(query_state_timestamp_ns=100 + index, host_monotonic_s=10. + index * .2)
            samples.append(value)
        report["initial_state_match"] = {"status": "ACCEPTED", "specification": specification,
            "desired_world_ned_m": [-5., -.5, -3.], "samples": samples,
            "accepted_state": deepcopy(samples[-1]), "elapsed_s": 1.0, "protective_reason": None}
        report["observations"] = [{"planning_input_body_state": {
            "world_ned_m": [-4.99, -.5, -3.], "velocity_ned_m_s": [.015, 0, 0],
            "acceleration_ned_m_s2": [.03, 0, 0], "query_state_timestamp_ns": 104}}]
        report["task_success"] = True
        return report

    def test_verified_initial_match_records_raw_first_observation_drift_and_separate_stage(self):
        row = summarize_run(self.matching_report())
        match = row["initial_state_match"]
        self.assertEqual(match["status"], "ACCEPTED_VERIFIED")
        self.assertTrue(match["stable_interval_verified"])
        self.assertEqual(match["sample_count"], 4)
        self.assertEqual(row["observed_approach_sample_count"], 1)
        self.assertAlmostEqual(match["first_observation_drift"]["position_difference_norm_m"], .01)
        self.assertAlmostEqual(match["first_observation_drift"]["velocity_difference_norm_m_s"], .005)
        self.assertEqual(match["first_observation_drift"]["query_timestamp_delta_ns"], 1)
        self.assertEqual(match["position_tolerance_norm_m"], .04)
        self.assertTrue(row["task_success"])

    def test_failed_status_never_trusts_leftover_accepted_state_or_landing_as_task_success(self):
        report = self.matching_report()
        report["initial_state_match"].update(status="PROTECTED_RETURN", protective_reason="POSITIONING_TIMEOUT")
        row = summarize_run(report)
        self.assertFalse(row["task_success"])
        self.assertTrue(row["safe_landing_and_release"])
        self.assertTrue(row["task_success_inconsistent_with_initial_acceptance"])
        self.assertFalse(row["initial_state_match"]["acceptance_verified"])
        self.assertIsNone(row["initial_state_match"]["accepted_state"])

    def test_incomplete_match_configuration_cannot_be_verified(self):
        report = self.matching_report()
        del report["initial_state_match"]["specification"]["speed_tolerance_m_s"]
        row = summarize_run(report)
        self.assertFalse(row["initial_state_match"]["thresholds_complete"])
        self.assertFalse(row["initial_state_match"]["acceptance_verified"])
        self.assertFalse(row["task_success"])

    def test_fake_accepted_sample_cannot_hide_nonzero_or_missing_previous_xy_command(self):
        for value in ([.001, 0], None):
            with self.subTest(value=value):
                report = self.matching_report()
                for sample in report["initial_state_match"]["samples"]:
                    if value is None:
                        del sample["previous_command_velocity_xy_m_s"]
                    else:
                        sample["previous_command_velocity_xy_m_s"] = value
                report["initial_state_match"]["accepted_state"] = deepcopy(report["initial_state_match"]["samples"][-1])
                row = summarize_run(report)
                self.assertFalse(row["initial_state_match"]["acceptance_verified"])
                self.assertFalse(row["task_success"])
                self.assertIn("FINAL_PREVIOUS_XY_COMMAND_NOT_VERIFIED_ZERO", row["initial_state_match"]["acceptance_evidence_reasons"])

    def test_final_state_recomputed_vector_cannot_be_hidden_by_saved_norm_or_accepted_flag(self):
        report = self.matching_report()
        for sample in report["initial_state_match"]["samples"]:
            sample.update(velocity_ned_m_s=[.018, .018, 0], speed_norm_m_s=.01)
        report["initial_state_match"]["accepted_state"] = deepcopy(report["initial_state_match"]["samples"][-1])
        row = summarize_run(report)
        self.assertFalse(row["initial_state_match"]["acceptance_verified"])
        self.assertGreater(row["initial_state_match"]["accepted_state_recomputed_values"]["speed_norm_m_s"], .02)

    def test_positioning_physical_budget_is_not_approach_budget_or_height(self):
        report = self.matching_report()
        report["initial_state_match"]["samples"].insert(0, {
            "world_ned_m": [-5.1, -.5, -2.5], "velocity_ned_m_s": [.2, 0, 0],
            "acceleration_ned_m_s2": [.4, 0, 0], "actual_body_rpy_deg": [0, 0, 0],
            "accepted": False, "host_monotonic_s": 9., "query_state_timestamp_ns": 99})
        row = summarize_run(report)
        self.assertEqual(row["initial_state_match"]["sampled_motion"]["speed_budget_exceedances"], 1)
        self.assertEqual(row["initial_state_match"]["physical_sampled_budget_status"], "OBSERVED_EXCEEDANCE")
        self.assertEqual(row["observed_approach_speed_budget_exceedances"], 0)
        self.assertAlmostEqual(row["min_sampled_approach_world_height_m"], 3.)

    def test_stable_suffix_requires_duration_and_fresh_gaps(self):
        for times in ([10., 10.1, 10.2, 10.3], [10., 10.2, 10.8, 11.0]):
            with self.subTest(times=times):
                report = self.matching_report()
                for sample, value in zip(report["initial_state_match"]["samples"], times):
                    sample["host_monotonic_s"] = value
                report["initial_state_match"]["accepted_state"] = deepcopy(report["initial_state_match"]["samples"][-1])
                row = summarize_run(report)
                self.assertFalse(row["initial_state_match"]["stable_interval_verified"])
                self.assertFalse(row["initial_state_match"]["acceptance_verified"])

    def test_earlier_stale_gap_does_not_discard_a_subsequently_complete_stable_suffix(self):
        report = self.matching_report()
        match = report["initial_state_match"]
        template = deepcopy(match["samples"][-1])
        match["samples"] = []
        for index, value in enumerate((10., 10.2, 10.8, 11., 11.2, 11.4)):
            sample = deepcopy(template)
            sample.update(host_monotonic_s=value, query_state_timestamp_ns=100 + index)
            match["samples"].append(sample)
        match["accepted_state"] = deepcopy(match["samples"][-1])
        row = summarize_run(report)
        self.assertTrue(row["initial_state_match"]["acceptance_verified"])
        self.assertEqual(row["initial_state_match"]["stable_suffix_sample_count"], 4)
        self.assertAlmostEqual(row["initial_state_match"]["stable_suffix_elapsed_s"], .6)

    def test_positioning_warmup_budget_and_commands_remain_outside_approach(self):
        report = self.matching_report()
        match = report["initial_state_match"]
        match["specification"]["reposition_speed_cap_m_s"] = .05
        match["warmup_settling_records"] = [{"samples": [{
            "world_ned_m": [-5., -.5, -2.8], "velocity_ned_m_s": [.2, 0, 0],
            "acceleration_ned_m_s2": [.4, 0, 0]}]}]
        match["command_records"] = [{"command_velocity_xy_m_s": [.04, .04]}]
        row = summarize_run(report)
        self.assertEqual(row["initial_state_match"]["warmup_settling_sample_count"], 1)
        self.assertEqual(row["initial_state_match"]["sample_count"], 5)
        self.assertEqual(row["initial_state_match"]["sampled_motion"]["acceleration_budget_exceedances"], 1)
        self.assertEqual(row["initial_state_match"]["command_xy_norm_violations"], 1)
        self.assertEqual(row["command_norm_assessed_count"], 0)
        self.assertEqual(row["observed_approach_sample_count"], 1)

    def test_missing_first_observation_input_does_not_substitute_a_later_frame(self):
        report = self.matching_report()
        report["observations"].insert(0, {"frame_id": "missing-body-query"})
        self.assertIsNone(summarize_run(report)["initial_state_match"]["first_observation_drift"])

    def test_paired_states_compare_vector_norms_with_twice_shared_limits(self):
        reports = [self.matching_report(policy) for policy in ("efficiency", "perception")]
        for sample in reports[1]["initial_state_match"]["samples"]:
            sample["world_ned_m"][0] += .03
            sample["velocity_ned_m_s"] = [-.01, 0, 0]
        reports[1]["initial_state_match"]["accepted_state"] = deepcopy(reports[1]["initial_state_match"]["samples"][-1])
        rows = [dict(summarize_run(report), session=1) for report in reports]
        pairs = summarize_matched_pairs(rows)
        self.assertEqual(pairs["paired_session_sample_count"], 1)
        pair = pairs["pairs"][0]
        self.assertAlmostEqual(pair["position_difference_norm_m"], .03)
        self.assertAlmostEqual(pair["velocity_difference_norm_m_s"], .02)
        self.assertEqual(pair["pair_difference_norm_limits"]["position_m"], .08)
        self.assertTrue(pair["within_shared_acceptance_envelope"])

    def test_out_of_envelope_pair_is_flagged_and_not_an_independent_accepted_pair(self):
        rows = [dict(summarize_run(self.matching_report(policy)), session=1) for policy in ("efficiency", "perception")]
        # Exercise the pair checker independently of the run acceptance guard.
        rows[1]["initial_state_match"]["accepted_state"]["world_ned_m"][0] += .09
        pairs = summarize_matched_pairs(rows)
        self.assertEqual(pairs["paired_session_sample_count"], 0)
        self.assertFalse(pairs["pairs"][0]["within_shared_acceptance_envelope"])

    def test_failed_positioning_counts_as_started_but_not_accepted_pair_or_task(self):
        reports = [self.matching_report(policy) for policy in ("efficiency", "perception")]
        reports[1]["initial_state_match"].update(status="PROTECTED_RETURN", accepted_state=None)
        rows = [dict(summarize_run(report), session=1) for report in reports]
        pairs = summarize_matched_pairs(rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["task_success"] for row in rows), 1)
        self.assertEqual(sum(row["safe_landing_and_release"] for row in rows), 2)
        self.assertEqual(pairs["started_session_count"], 1)
        self.assertEqual(pairs["paired_session_sample_count"], 0)
        self.assertEqual(build_overview(rows)["runs_with_failed_or_invalid_initial_acceptance"], 1)

    def test_legacy_missing_acceptance_retains_task_record_as_unassessed(self):
        report = self.motion_report()
        report["task_success"] = True
        row = summarize_run(report)
        self.assertEqual(row["initial_state_match"]["status"], "LEGACY_NOT_EVALUATED")
        self.assertIsNone(row["initial_state_match"]["acceptance_verified"])
        self.assertTrue(row["task_success"])

    def test_actual_sent_xy_command_norm_is_checked_without_calling_it_3d_velocity(self):
        report = self.motion_report()
        report["planning_records"] = [{
            "command": {"command_velocity_ned_m_s": [.10, 0, 0],
                        "requested_preview_velocity_ned_m_s": [4, 4, 4]},
            "execution_samples": [
                {"command_velocity_xy_m_s": [.08, 0], "speed_norm_m_s": .10, "acceleration_norm_m_s2": .1},
                {"command_velocity_xy_m_s": [.12, .12], "speed_norm_m_s": .12, "acceleration_norm_m_s2": .1},
            ],
        }]
        row = summarize_run(report)
        self.assertEqual(row["command_norm_assessed_count"], 2)
        self.assertEqual(row["command_norm_violations"], 1)
        self.assertFalse(row["recorded_command_norm_within_budget"])
        self.assertTrue(row["recorded_reference_command_norm_within_budget"])
        self.assertEqual(row["command_norm_scope"], "ACTUAL_XY_VELOCITY_COMPONENTS_ONLY_NOT_3D_VELOCITY_COMMAND")
        self.assertEqual(row["physical_sampled_budget_status"], "NO_EXCEEDANCE_IN_RECORDED_SAMPLES")

    def test_settling_and_execution_have_separate_counts_and_combined_total(self):
        report = self.motion_report()
        report["planning_records"] = [{"executed_body_state": {"speed_norm_m_s": .16, "acceleration_norm_m_s2": .31}}]
        report["settling_records"] = [{"samples": [
            {"speed_m_s": .14, "acceleration_norm_m_s2": .1},
            {"speed_m_s": .2, "acceleration_norm_m_s2": .5},
        ]}]
        row = summarize_run(report)
        self.assertEqual(row["execution_sample_count"], 1)
        self.assertEqual(row["settling_sample_count"], 2)
        self.assertEqual(row["settling_speed_budget_exceedances"], 1)
        self.assertEqual(row["settling_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["observed_approach_sample_count"], 3)
        self.assertEqual(row["observed_approach_speed_budget_exceedances"], 2)
        self.assertEqual(row["observed_approach_acceleration_budget_exceedances"], 2)
        self.assertEqual(row["max_settling_sample_speed_m_s"], .2)

    def test_missing_legacy_acceleration_budget_is_unassessed_not_compliant(self):
        report = deepcopy(self.report)
        report["settling_records"] = [{"samples": [{"speed_m_s": .1, "acceleration_norm_m_s2": 9.}]}]
        row = summarize_run(report)
        self.assertIsNone(row["settling_acceleration_budget_exceedances"])
        self.assertEqual(row["physical_sampled_budget_status"], "PARTIALLY_ASSESSED")
        self.assertEqual(row["body_execution_mode"], "NOT_RECORDED")
        self.assertEqual(row["max_settling_sample_acceleration_m_s2"], 9.)

    def test_invalid_raw_velocity_does_not_fall_back_to_a_compliant_scalar(self):
        report = self.motion_report()
        report["planning_records"] = [{"executed_body_state": {
            "velocity_ned_m_s": [float("nan"), 0, 0], "speed_norm_m_s": .1,
            "acceleration_norm_m_s2": .1,
        }}]
        row = summarize_run(report)
        self.assertEqual(row["execution_speed_assessed_sample_count"], 0)
        self.assertEqual(row["physical_sampled_budget_status"], "PARTIALLY_ASSESSED")
        json.dumps(row, allow_nan=False)

    def test_no_samples_does_not_infer_physical_compliance(self):
        row = summarize_run(self.report)
        self.assertEqual(row["physical_sampled_budget_status"], "NOT_RECORDED")
        self.assertIsNone(row["execution_max_speed_norm_m_s"])
        self.assertEqual(row["observed_approach_sample_count"], 0)

    def test_world_height_minmax_and_overview_cover_approach_samples_only(self):
        report = self.motion_report()
        report["observations"] = [{"world_ned_m": [-5, 0, -3.]}]
        report["planning_records"] = [{"execution_samples": [
            {"world_ned_m": [-4.9, 0, -2.9], "speed_norm_m_s": .1, "acceleration_norm_m_s2": .1}],
            "executed_body_state": {"world_ned_m": [-4.9, 0, -10.0]}}]
        report["settling_records"] = [{"samples": [{"world_ned_m": [-5, 0, -3.2], "speed_m_s": .1, "acceleration_norm_m_s2": .1}]}]
        report["landing_evidence"]["world_ned_m"] = [-5, -4, .8]
        row = summarize_run(report)
        self.assertAlmostEqual(row["min_sampled_approach_world_height_m"], 2.9)
        self.assertAlmostEqual(row["max_sampled_approach_world_height_m"], 3.2)
        overview = build_overview([row])
        self.assertEqual(overview["body_execution_modes"], ["moveByVelocityZAsync_constant_world_height"])
        self.assertEqual(overview["execution_modes"], ["fixed_world_height_preview"])
        self.assertAlmostEqual(overview["min_sampled_approach_world_height_m"], 2.9)
        self.assertIn("not ground clearance", overview["height_scope"])

    def test_legacy_budget_rejection_with_unrecorded_input_is_incomplete(self):
        report = self.motion_report()
        report["protective_return_reason"] = "Measured initial acceleration exceeds planner budget"
        report["planning_records"] = [{
            "command": {"command_velocity_ned_m_s": [.1, 0, 0]},
            "execution_samples": [{"speed_norm_m_s": .1, "acceleration_norm_m_s2": .1}],
        }]
        report["settling_records"] = [{"samples": [{"speed_m_s": .01, "acceleration_norm_m_s2": .02}]}]
        row = summarize_run(report)
        self.assertTrue(row["safe_landing_and_release"])
        self.assertTrue(row["recorded_command_norm_within_budget"])
        self.assertEqual(row["observed_approach_acceleration_budget_exceedances"], 0)
        self.assertEqual(row["physical_sampled_budget_status"], "INCOMPLETE_WITH_REPORTED_BUDGET_REJECTION")
        self.assertEqual(row["unquantified_budget_rejection_reasons"], [report["protective_return_reason"]])
        self.assertIsNone(row["planning_input_max_acceleration_norm_m_s2"])
        self.assertTrue(row["reported_motion_budget_rejection_present"])
        overview = build_overview([row])
        self.assertEqual(overview["runs_with_unquantified_motion_budget_rejection"], 1)
        self.assertEqual(overview["runs_with_observed_physical_budget_exceedance"], 0)

    def test_planning_input_and_command_start_count_as_distinct_physical_queries(self):
        report = self.motion_report()
        report["observations"] = [{"planning_input_body_state": {
            "world_ned_m": [-4.9, 0, -3.1], "velocity_ned_m_s": [.12, .12, 0],
            "acceleration_ned_m_s2": [.4, 0, 0], "query_state_timestamp_ns": 123,
            "protection_reasons": ["ACTUAL_SPEED_BUDGET_EXCEEDED", "ACTUAL_ACCELERATION_BUDGET_EXCEEDED"],
        }}]
        report["planning_records"] = [{"command_start_actual_body_state": {
            "world_ned_m": [-4.9, 0, -2.9], "speed_norm_m_s": .1, "acceleration_norm_m_s2": .31,
            "query_state_timestamp_ns": 123, "protection_reasons": ["ACTUAL_ACCELERATION_BUDGET_EXCEEDED"],
        }, "execution_samples": [{"speed_norm_m_s": .1, "acceleration_norm_m_s2": .1}]}]
        report["settling_records"] = [{"samples": [{"speed_m_s": .1, "acceleration_norm_m_s2": .1}]}]
        report["protective_return_reason"] = "Measured initial acceleration exceeds planner budget"
        row = summarize_run(report)
        self.assertEqual(row["planning_input_sample_count"], 1)
        self.assertEqual(row["command_start_sample_count"], 1)
        self.assertEqual(row["planning_input_speed_budget_exceedances"], 1)
        self.assertEqual(row["planning_input_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["command_start_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["observed_approach_sample_count"], 4)
        self.assertEqual(row["observed_approach_acceleration_budget_exceedances"], 2)
        self.assertEqual(row["physical_sampled_budget_status"], "OBSERVED_EXCEEDANCE")
        self.assertEqual(row["unquantified_budget_rejection_reasons"], [])
        self.assertAlmostEqual(row["min_sampled_approach_world_height_m"], 2.9)
        self.assertAlmostEqual(row["max_sampled_approach_world_height_m"], 3.1)

    def test_budget_rejection_cannot_be_quantified_by_an_earlier_input_query(self):
        report = self.motion_report()
        report["observations"] = [
            {"planning_input_body_state": {"speed_norm_m_s": .1, "acceleration_norm_m_s2": .5}},
            {"frame_id": "last-frame-without-saved-body-state"},
        ]
        report["protective_return_reason"] = "Measured initial acceleration exceeds planner budget"
        row = summarize_run(report)
        self.assertEqual(row["observed_approach_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["physical_sampled_budget_status"], "INCOMPLETE_WITH_REPORTED_BUDGET_REJECTION")
        self.assertTrue(row["observed_numeric_physical_budget_exceedance_present"])
        self.assertTrue(row["unquantified_motion_budget_rejection_present"])

    def test_invalid_saved_guard_vector_does_not_invent_a_numeric_exceedance(self):
        report = self.motion_report()
        report["protective_return_reason"] = "Measured initial acceleration exceeds planner budget"
        report["observations"] = [{"planning_input_body_state": {
            "speed_norm_m_s": .1, "acceleration_ned_m_s2": [float("nan"), 0, 0],
            "acceleration_norm_m_s2": .5,
        }}]
        row = summarize_run(report)
        self.assertEqual(row["planning_input_acceleration_assessed_sample_count"], 0)
        self.assertEqual(row["observed_approach_acceleration_budget_exceedances"], 0)
        self.assertEqual(row["physical_sampled_budget_status"], "INCOMPLETE_WITH_REPORTED_BUDGET_REJECTION")
        json.dumps(row, allow_nan=False)

    def test_command_budget_rejection_is_not_a_physical_motion_rejection(self):
        report = self.motion_report()
        report["protective_return_reason"] = "Previous velocity command exceeds current budget"
        report["planning_records"] = [{"execution_samples": [{"speed_norm_m_s": .1, "acceleration_norm_m_s2": .1}]}]
        row = summarize_run(report)
        self.assertFalse(row["reported_motion_budget_rejection_present"])
        self.assertEqual(row["physical_sampled_budget_status"], "NO_EXCEEDANCE_IN_RECORDED_SAMPLES")
        self.assertIn(report["protective_return_reason"], row["recorded_protection_reasons"])

    def test_coded_command_start_budget_rejection_has_its_own_quantified_evidence(self):
        report = self.motion_report()
        report["protective_return_reason"] = "ACTUAL_ACCELERATION_BUDGET_EXCEEDED"
        report["planning_records"] = [{"command_start_actual_body_state": {
            "speed_norm_m_s": .1, "acceleration_norm_m_s2": .41,
            "protection_reasons": "ACTUAL_ACCELERATION_BUDGET_EXCEEDED",
        }}]
        row = summarize_run(report)
        self.assertEqual(row["command_start_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["unquantified_budget_rejection_reasons"], [])
        self.assertEqual(row["physical_sampled_budget_status"], "OBSERVED_EXCEEDANCE")

    def test_generic_initial_budget_error_uses_explicit_command_start_rejection_stage(self):
        report = self.motion_report()
        report["protective_return_reason"] = "Measured initial acceleration exceeds planner budget"
        report["observations"] = [{"planning_input_body_state": {"speed_norm_m_s": .1, "acceleration_norm_m_s2": .1}}]
        report["planning_records"] = [{"execution_status": "REJECTED_COMMAND_START_MOTION_BUDGET",
            "command_start_actual_body_state": {"speed_norm_m_s": .1, "acceleration_norm_m_s2": .4}}]
        row = summarize_run(report)
        self.assertEqual(row["planning_input_acceleration_budget_exceedances"], 0)
        self.assertEqual(row["command_start_acceleration_budget_exceedances"], 1)
        self.assertEqual(row["unquantified_budget_rejection_reasons"], [])
        self.assertEqual(row["budget_rejection_evidence"][0]["stage"], "command_start")

    def test_keepalive_commands_count_separately_without_becoming_physical_samples_or_duration(self):
        report = self.motion_report()
        report["metrics"] = {"approach_duration_s": 1.23}
        report["planning_records"] = [{"execution_samples": [{
            "command_velocity_xy_m_s": [.10, 0], "speed_norm_m_s": .10, "acceleration_norm_m_s2": .1}]}]
        report["goal_keepalive_records"] = [
            {"command_velocity_xy_m_s": [.10, 0], "requested_duration_s": .5, "join_called": False,
             "host_monotonic_s": 10., "rpc_enqueue_return_host_s": 10.01},
            {"command_velocity_xy_m_s": [.12, .12], "requested_duration_s": .5, "join_called": False,
             "host_monotonic_s": 10.1, "rpc_enqueue_return_host_s": 10.11},
        ]
        row = summarize_run(report)
        self.assertEqual(row["goal_keepalive_command_count"], 2)
        self.assertEqual(row["goal_keepalive_command_norm_assessed_count"], 2)
        self.assertEqual(row["goal_keepalive_command_norm_violations"], 1)
        self.assertEqual(row["command_norm_assessed_count"], 3)
        self.assertEqual(row["command_norm_violations"], 1)
        self.assertEqual(row["observed_approach_sample_count"], 1)
        self.assertEqual(row["physical_sampled_budget_status"], "NO_EXCEEDANCE_IN_RECORDED_SAMPLES")
        self.assertEqual(row["approach_duration_s"], 1.23)
        self.assertFalse(row["goal_keepalive_requested_duration_is_executed_time"])
        self.assertEqual(row["command_norm_scope"], "ACTUAL_XY_VELOCITY_COMPONENTS_ONLY_NOT_3D_VELOCITY_COMMAND")

    def test_invalid_keepalive_vector_is_unassessed_instead_of_compliant(self):
        report = self.motion_report()
        report["goal_keepalive_records"] = [{"command_velocity_xy_m_s": [.1, 0, 0], "requested_duration_s": .5}]
        row = summarize_run(report)
        self.assertEqual(row["goal_keepalive_command_count"], 1)
        self.assertEqual(row["goal_keepalive_command_norm_assessed_count"], 0)
        self.assertEqual(row["goal_keepalive_command_norm_invalid_count"], 1)
        self.assertIsNone(row["recorded_command_norm_within_budget"])
        self.assertEqual(row["observed_approach_sample_count"], 0)

    def test_missing_terminal_report_counts_as_started_but_unstarted_policy_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.motion_report()
            report.update(task_success=True, configuration_sha256="frozen-config", source_sha256={"controller.py": "frozen-source"})
            report["planning_records"] = [{"executed_body_state": {"speed_norm_m_s": .2, "acceleration_norm_m_s2": .1}}]
            report_path = root / "completed_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            for index, runs in enumerate(([
                {"run": 1, "policy": "efficiency", "report_path": str(report_path)},
                {"run": 2, "policy": "perception", "status": "STARTED", "report_path": str(root / "missing_report.json")},
            ], []), 1):
                batch = root / f"batch_{index}"
                batch.mkdir()
                (batch / "batch_report.json").write_text(json.dumps({"target_source": "geometry_oracle", "runs": runs}), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "status": "INTERRUPTED", "target_source": "geometry_oracle", "experiment_config_sha256": "frozen-config",
                "planned_sessions": 2, "planned_flights_per_session": 2,
                "sessions": [{"session_index": index, "batch_path": str(root / f"batch_{index}")} for index in (1, 2)],
            }), encoding="utf-8")
            with patch("analyze_trajectory_pilot.render"):
                result = main(manifest_path, root / "analysis")
            self.assertEqual(result["planned_flights"], 4)
            self.assertEqual(result["started_flights"], 2)
            self.assertEqual(result["task_success_count"], 1)
            self.assertEqual(result["safe_landing_and_release_count"], 1)
            self.assertEqual(result["runs"][1]["status"], "NO_TERMINAL_REPORT")
            self.assertEqual(result["overview"]["runs_with_observed_physical_budget_exceedance"], 1)


if __name__ == "__main__":
    unittest.main()
