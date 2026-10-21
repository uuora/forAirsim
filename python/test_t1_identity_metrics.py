"""Independent evaluation accounting: gaps and failures cannot improve scores."""
import unittest

from t1_identity_metrics import (EvaluationGrid, EvaluationTick, PredictionTick, RoundRecord,
                                 TruthTick, evaluate_identity_round, round_success_summary)


class IdentityMetricsTests(unittest.TestCase):
    def evaluate(self, selections, *, times=None, end=None, truth=None, missing_predictions=(),
                 invalid_predictions=(), timeout=2.0, loss_signal="identity"):
        times = times if times is not None else list(range(len(selections)))
        end = end if end is not None else times[-1] + 1
        ids = [f"t{index}" for index in range(len(times))]
        grid = EvaluationGrid.from_times(times, end_time_s=end, tick_ids=ids)
        truth = truth if truth is not None else [TruthTick(identity, "target", True) for identity in ids]
        predictions = [PredictionTick(ids[index], None if index in invalid_predictions else selection,
                                      "INVALID" if index in invalid_predictions else
                                      "OUTPUT" if selection is not None else "NO_OUTPUT")
                       for index, selection in enumerate(selections) if index not in missing_predictions]
        return evaluate_identity_round(grid, truth, predictions, target_identity="target",
                                       reacquisition_timeout_s=timeout, loss_signal=loss_signal)

    def test_all_ticks_missing_outputs_and_wrong_identity_in_denominator(self):
        report = self.evaluate(["target", None, "other", "target"], missing_predictions=(1,))
        self.assertEqual(report["expected_ticks"], 4)
        self.assertEqual(report["prediction_missing_ticks"], 1)
        self.assertEqual(report["no_output_ticks"], 1)
        self.assertEqual(report["wrong_identity_ticks"], 1)
        self.assertEqual(report["identity_correct_time_fraction"], 0.5)
        self.assertEqual(report["longest_loss_s"], 2)

    def test_irregular_grid_uses_time_not_surviving_frame_count(self):
        report = self.evaluate(["target", None, "other", "target"], times=[0, 0.5, 3, 4], end=6)
        self.assertEqual(report["identity_correct_tick_fraction"], 0.5)
        self.assertAlmostEqual(report["identity_correct_time_fraction"], 2.5 / 6)
        self.assertEqual(report["longest_loss_s"], 3.5)

    def test_identity_and_visible_observation_are_distinct_on_occluded_tick(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t1", "target", False), TruthTick("t2", "target", True)]
        report = self.evaluate(["target", "target", "target"], truth=truth)
        self.assertEqual(report["occluded_ticks"], 1)
        self.assertEqual(report["nonvisible_target_output_ticks"], 1)
        self.assertEqual(report["identity_correct_allticks_time_fraction"], 1)
        self.assertAlmostEqual(report["effective_visible_observation_time_fraction"], 2 / 3)
        self.assertEqual(report["longest_identity_loss_s"], 0)
        self.assertEqual(report["longest_effective_visible_observation_loss_s"], 1)
        self.assertEqual(report["longest_loss_s"], 0)
        self.assertEqual(report["identity_reacquisition_episodes"], [])
        self.assertEqual(len(report["effective_visible_observation_reacquisition_episodes"]), 1)

    def test_stale_label_must_be_explicitly_invalid_independent_of_visibility(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t1", "target", False), TruthTick("t2", "target", True)]
        report = self.evaluate(["target", "target", "target"], truth=truth, invalid_predictions=(1,))
        self.assertAlmostEqual(report["identity_correct_allticks_time_fraction"], 2 / 3)
        self.assertAlmostEqual(report["effective_visible_observation_time_fraction"], 2 / 3)
        self.assertEqual(report["invalid_output_ticks"], 1)
        self.assertEqual(report["longest_identity_loss_s"], 1)

    def test_loss_signal_explicitly_selects_primary_visible_observation_sequence(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t1", "target", False), TruthTick("t2", "target", True)]
        report = self.evaluate(["target", "target", "target"], truth=truth,
                               loss_signal="effective_visible_observation")
        self.assertEqual(report["loss_signal"], "effective_visible_observation")
        self.assertEqual(report["longest_loss_s"], 1)
        self.assertEqual(report["reacquisition_episodes"], report["effective_visible_observation_reacquisition_episodes"])
        self.assertEqual(report["identity_correct_allticks_time_fraction"], 1)
        with self.assertRaises(ValueError):
            self.evaluate(["target"], loss_signal="unspecified")

    def test_unknown_truth_preserved_separate_with_bounds(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t2", None, None, "UNCERTAIN"), TruthTick("t3", "target", True)]
        report = self.evaluate(["target", None, "other", None], truth=truth)
        self.assertEqual(report["truth_missing_ticks"], 1)
        self.assertEqual(report["truth_uncertain_ticks"], 1)
        self.assertEqual(report["expected_ticks"], 4)
        self.assertIsNone(report["identity_correct_time_fraction"])
        self.assertIsNone(report["effective_visible_observation_time_fraction"])
        self.assertEqual(report["identity_correct_time_fraction_lower_bound"], 0.25)
        self.assertEqual(report["identity_correct_time_fraction_upper_bound"], 0.75)
        self.assertIsNone(report["longest_loss_s"])
        self.assertEqual(report["longest_loss_lower_bound_s"], 1)
        self.assertEqual(report["longest_loss_upper_bound_s"], 3)

    def test_direct_wrong_instance_switch_differs_from_gap(self):
        report = self.evaluate(["target", "other", "other", None, "target", "third"])
        self.assertEqual(report["direct_identity_switch_count"], 2)
        self.assertEqual(report["target_to_other_switch_count"], 2)
        self.assertEqual(report["identity_change_after_gap_count"], 1)
        self.assertEqual(report["wrong_identity_ticks"], 3)

    def test_reacquisition_within_timeout_and_after_timeout_both_retained(self):
        report = self.evaluate(["target", None, "target", None, None, None, "target"], timeout=2)
        self.assertEqual([item["status"] for item in report["reacquisition_episodes"]],
                         ["REACQUIRED_WITHIN_TIMEOUT", "REACQUIRED_AFTER_TIMEOUT"])
        self.assertEqual(report["reacquisition_failure_or_unfinished_count"], 1)
        self.assertEqual(report["longest_loss_s"], 3)

    def test_remaining_lost_not_discarded(self):
        report = self.evaluate(["target", None, None], timeout=1.5)
        episode = report["reacquisition_episodes"][0]
        self.assertEqual(episode["status"], "TIMED_OUT_STILL_LOST")
        self.assertEqual(episode["duration_s"], 2)
        self.assertFalse(episode["reacquired"])
        self.assertEqual(report["reacquisition_failure_or_unfinished_count"], 1)

    def test_short_ending_loss_is_retained_as_unfinished(self):
        report = self.evaluate(["target", None], timeout=3)
        self.assertEqual(report["reacquisition_episodes"][0]["status"], "STILL_LOST_AT_END_BEFORE_TIMEOUT")
        self.assertEqual(report["reacquisition_failure_or_unfinished_count"], 1)

    def test_never_acquired_is_initial_failure_not_fake_reacquisition(self):
        report = self.evaluate([None, "other", None])
        self.assertEqual(report["initial_acquisition_status"], "NOT_ACQUIRED")
        self.assertIsNone(report["initial_acquisition_duration_s"])
        self.assertEqual(report["longest_loss_s"], 3)
        self.assertEqual(report["reacquisition_episodes"], [])

    def test_truth_gap_marks_unresolved_episode(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t1", "target", True),
                 TruthTick("t3", "target", True)]
        report = self.evaluate(["target", None, None, "target"], truth=truth)
        self.assertEqual(report["reacquisition_episodes"][0]["status"], "TRUTH_GAP_BEFORE_REACQUISITION")
        self.assertEqual(report["reacquisition_evaluation_status"], "INCOMPLETE_TRUTH_COVERAGE")

    def test_invalid_stale_label_not_anchor_for_direct_switch(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t1", "target", False), TruthTick("t2", "target", True)]
        report = self.evaluate(["target", "target", "other"], truth=truth, invalid_predictions=(1,))
        self.assertEqual(report["direct_identity_switch_count"], 0)
        self.assertEqual(report["identity_change_after_gap_count"], 1)

    def test_valid_identity_output_stays_anchor_when_visibility_false(self):
        truth = [TruthTick("t0", "target", True), TruthTick("t1", "target", False), TruthTick("t2", "target", True)]
        report = self.evaluate(["target", "target", "other"], truth=truth)
        self.assertEqual(report["direct_identity_switch_count"], 1)
        self.assertEqual(report["identity_change_after_gap_count"], 0)

    def test_invalid_output_is_loss(self):
        grid = EvaluationGrid.from_times([0, 1], end_time_s=2, tick_ids=["a", "b"])
        report = evaluate_identity_round(grid, [TruthTick("a", "target", True), TruthTick("b", "target", True)],
            [PredictionTick("a", "target", "OUTPUT"), PredictionTick("b", status="INVALID")],
            target_identity="target", reacquisition_timeout_s=2)
        self.assertEqual(report["invalid_output_ticks"], 1)
        self.assertEqual(report["identity_correct_time_fraction"], 0.5)

    def test_grid_duplicate_nonfinite_backward_or_missing_end_rejected(self):
        for times, end in (([0, 0], 1), ([1, 0], 2), ([0, float("nan")], 2), ([0, 1], 1), ([], 1)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                EvaluationGrid.from_times(times, end_time_s=end)
        with self.assertRaises(ValueError):
            EvaluationGrid((EvaluationTick("a", 0, 1), EvaluationTick("b", 2, 3)))
        with self.assertRaises(ValueError):
            EvaluationGrid.from_times([0, 1], end_time_s=2, tick_ids=["same", "same"])

    def test_duplicate_or_unexpected_prediction_and_truth_rejected(self):
        grid = EvaluationGrid.from_times([0], end_time_s=1, tick_ids=["a"])
        for truths, predictions in (([TruthTick("a", "target", True)] * 2, []),
                                    ([], [PredictionTick("a")] * 2), ([], [PredictionTick("extra")])):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_identity_round(grid, truths, predictions, target_identity="target", reacquisition_timeout_s=1)

    def test_truth_must_match_registered_instance(self):
        with self.assertRaises(ValueError):
            self.evaluate(["target"], truth=[TruthTick("t0", "different", True)])
        with self.assertRaises(ValueError):
            TruthTick("t0", "target", None)
        with self.assertRaises(ValueError):
            PredictionTick("t0", "target", "NO_OUTPUT")


class RoundAccountingTests(unittest.TestCase):
    def test_all_confirmed_started_include_interrupted_unfeasible_and_unknown(self):
        records = [RoundRecord("success", "CONFIRMED_STARTED", "SUCCESS", True, True),
                   RoundRecord("interrupted", "CONFIRMED_STARTED", "INTERRUPTED"),
                   RoundRecord("unfeasible", "CONFIRMED_STARTED", "NO_FEASIBLE_TRAJECTORY"),
                   RoundRecord("unknown-outcome", "CONFIRMED_STARTED", "UNKNOWN"),
                   RoundRecord("prestart", "CONFIRMED_NOT_STARTED", "NOT_STARTED"),
                   RoundRecord("unknown-start", "UNKNOWN", "UNKNOWN")]
        report = round_success_summary(records)
        self.assertEqual(report["registered_rounds"], 6)
        self.assertEqual(report["confirmed_started_rounds"], 4)
        self.assertEqual(report["task_success_all_confirmed_started"], 0.25)
        self.assertEqual(report["confirmed_not_started_rounds"], 1)
        self.assertEqual(report["unknown_start_rounds"], 1)
        self.assertEqual(report["start_reconciliation_status"], "UNKNOWN_STARTS_REMAIN")

    def test_declared_success_needs_both_verification_gates(self):
        report = round_success_summary([RoundRecord("missing", "CONFIRMED_STARTED", "SUCCESS", True, None),
                                        RoundRecord("unsafe", "CONFIRMED_STARTED", "SUCCESS", False, True)])
        self.assertEqual(report["verified_successful_rounds"], 0)
        self.assertEqual(report["confirmed_started_rounds"], 2)
        self.assertEqual(report["task_success_all_confirmed_started"], 0)
        self.assertEqual(len(report["declared_success_missing_or_failed_gate_round_ids"]), 2)

    def test_no_started_rounds_no_division_or_fabricated_success(self):
        report = round_success_summary([RoundRecord("prestart", "CONFIRMED_NOT_STARTED", "NOT_STARTED")])
        self.assertIsNone(report["task_success_all_confirmed_started"])
        self.assertEqual(report["verified_successful_rounds"], 0)

    def test_missing_declared_round_remains_unknown_start(self):
        report = round_success_summary([RoundRecord("known", "CONFIRMED_STARTED", "SUCCESS", True, True)],
                                       expected_round_ids=["known", "missing"])
        self.assertEqual(report["registered_rounds"], 2)
        self.assertEqual(report["missing_round_records"], 1)
        self.assertEqual(report["unknown_start_round_ids"], ["missing"])
        self.assertEqual(report["confirmed_started_rounds"], 1)
        self.assertEqual(report["task_success_all_confirmed_started"], 1)
        self.assertEqual(report["start_reconciliation_status"], "UNKNOWN_STARTS_REMAIN")
        with self.assertRaises(ValueError):
            round_success_summary([RoundRecord("extra", "CONFIRMED_STARTED", "FAILED")], expected_round_ids=["known"])

    def test_duplicate_round_and_contradictory_start_rejected(self):
        record = RoundRecord("same", "CONFIRMED_STARTED", "FAILED")
        with self.assertRaises(ValueError):
            round_success_summary([record, record])
        with self.assertRaises(ValueError):
            RoundRecord("bad", "CONFIRMED_STARTED", "NOT_STARTED")
        with self.assertRaises(ValueError):
            RoundRecord("bad", "CONFIRMED_NOT_STARTED", "SUCCESS", True, True)


if __name__ == "__main__":
    unittest.main()
